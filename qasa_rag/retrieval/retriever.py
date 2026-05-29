from dataclasses import dataclass, field

from neo4j import Driver
from pydantic import BaseModel

from qasa_rag.client import create_genai_client
from qasa_rag.embedder import Embedder

from .debug import RetrieverDebugger


class QueryEntity(BaseModel):
    name: str
    search_hint: str

class QueryEntities(BaseModel):
    entities: list[QueryEntity]


QUERY_ENTITY_PROMPT = """Extract named entities from the question for knowledge graph lookup.

For each entity return:
- name: entity name exactly as written, lowercase
- search_hint: used for vector search, should be name + 1-2 disambiguating words from the question (e.g. "uhf film", "titanic ship").

Rules:
- Extract proper nouns: people, places, organizations, works, events
- Do NOT extract generic words or question words
- Always extract at least one entity

Question: {question}"""


_PROPAGATION_QA = """
MATCH (src:Entity) WHERE src._resource > $threshold
MATCH (src)-[:RELATES_TO]-(dst:Entity)
WHERE dst.description_embedding IS NOT NULL
  AND (dst._resource IS NULL OR dst._resource < src._resource)

WITH dst, src,
     gds.similarity.cosine(dst.description_embedding, $qe) AS sim,
     src._resource AS parent_res

WITH dst,
     sum(parent_res * $decay * CASE WHEN sim > 0 THEN sim ELSE 0 END) AS incoming

WHERE incoming > $threshold
SET dst._resource = coalesce(dst._resource, 0) + incoming
RETURN count(dst) AS active
"""

_PROPAGATION_UNIFORM = """
MATCH (src:Entity) WHERE src._resource > $threshold
MATCH (src)-[:RELATES_TO]-(dst:Entity)
WHERE dst._resource IS NULL OR dst._resource < src._resource

WITH dst,
     sum(src._resource * $decay) AS incoming

WHERE incoming > $threshold
SET dst._resource = coalesce(dst._resource, 0) + incoming
RETURN count(dst) AS active
"""


@dataclass
class RetrievalResult:
    paths: list[dict] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)
    seed_entities: list[str] = field(default_factory=list)
    steps_taken: int = 0
    extracted_entities: list[QueryEntity] = field(default_factory=list)


class QueryAwareRetriever:
    def __init__(
        self,
        driver: Driver,
        embedder: Embedder,
        llm_model: str = "gemini-2.5-flash",
        max_steps: int = 3,
        decay: float = 0.7,
        resource_threshold: float = 0.01,
        top_k_paths: int = 30,
        top_k_entities: int = 30,
        seed_vector_fallback_k: int = 5,
        query_aware: bool = True,
        min_cosine: float = 0.0,
        query_entity_cache: dict[str, list[QueryEntity]] | None = None,
        verbose: bool = False,
    ) -> None:
        self._driver = driver
        self._embedder = embedder
        self._genai = create_genai_client()
        self._llm_model = llm_model
        self._max_steps = max_steps
        self._decay = decay
        self._threshold = resource_threshold
        self._top_k_paths = top_k_paths
        self._top_k_entities = top_k_entities
        self._seed_fallback_k = seed_vector_fallback_k
        self._query_aware = query_aware
        self._min_cosine = min_cosine
        self._query_entity_cache = query_entity_cache
        self._debug = RetrieverDebugger(driver, decay, resource_threshold) if verbose else None

    def retrieve(self, question: str) -> RetrievalResult:
        query_entities = self._extract_query_entities(question)
        query_embedding = self._embedder.embed(question)
        seeds = self._find_seeds(query_entities)
        if self._debug:
            self._debug.log_seeds(seeds)

        if not seeds:
            return RetrievalResult()

        self._init_resources(seeds)

        steps = 0
        for step in range(self._max_steps):
            if self._debug:
                self._debug.log_flows(step + 1, query_embedding)

            active = self._propagate_step(query_embedding)
            steps = step + 1

            if self._debug:
                self._debug.log_state(steps, active)

            if active == 0:
                break

        entities = self._get_activated_entities()
        seed_names = [s["name"] for s in seeds]
        paths = self._collect_paths(seed_names)
        self._cleanup()

        return RetrievalResult(
            paths=paths,
            entities=entities,
            seed_entities=seed_names,
            steps_taken=steps,
            extracted_entities=query_entities,
        )

    def _extract_query_entities(self, question: str) -> list[QueryEntity]:
        if self._query_entity_cache is not None and question in self._query_entity_cache:
            return self._query_entity_cache[question]

        response = self._genai.models.generate_content(
            model=self._llm_model,
            contents=QUERY_ENTITY_PROMPT.format(question=question),
            config={
                "response_mime_type": "application/json",
                "response_schema": QueryEntities,
                "temperature": 0.0,
            },
        )

        entities = response.parsed.entities if response.parsed else []

        if self._query_entity_cache is not None:
            self._query_entity_cache[question] = entities

        return entities

    def _find_seeds(
        self,
        query_entities: list[QueryEntity],
    ) -> list[dict]:
        seeds = []
        with self._driver.session() as session:
            for entity in query_entities:
                result = session.run(
                    "MATCH (e:Entity {name: $name}) RETURN e.name AS name",
                    name=entity.name,
                ).single()

                if result:
                    seeds.append({"name": result["name"], "resource": 1.0})

            if not seeds:
                # TODO: distribute _seed_fallback_k across all query entities.
                # 99% that we're going to have only 1 query entity, but still we have to make it more robust.
                first_seed_embedding = self._embedder.embed(query_entities[0].search_hint)

                rows = session.run(
                    """
                    CALL db.index.vector.queryNodes(
                        'entity_description_embeddings', $k, $emb
                    )
                    YIELD node, score
                    RETURN node.name AS name, score
                    """,
                    k=self._seed_fallback_k,
                    emb=first_seed_embedding,
                ).data()

                max_score = max(r["score"] for r in rows)
                seeds = [{"name": r["name"], "resource": r["score"] / max_score} for r in rows]

        return seeds

    def _init_resources(self, seeds: list[dict]) -> None:
        with self._driver.session() as session:
            session.run("MATCH (e:Entity) REMOVE e._resource")
            for s in seeds:
                session.run(
                    "MATCH (e:Entity {name: $name}) SET e._resource = $res",
                    name=s["name"],
                    res=s["resource"],
                )

    def _propagate_step(self, query_embedding: list[float]) -> int:
        with self._driver.session() as session:
            if self._query_aware:
                result = session.run(
                    _PROPAGATION_QA,
                    qe=query_embedding,
                    decay=self._decay,
                    threshold=self._threshold,
                ).single()
            else:
                result = session.run(
                    _PROPAGATION_UNIFORM,
                    decay=self._decay,
                    threshold=self._threshold,
                ).single()

            return result["active"] if result else 0

    def _get_activated_entities(self) -> list[dict]:
        with self._driver.session() as session:
            return session.run(
                """
                MATCH (e:Entity) WHERE e._resource IS NOT NULL
                RETURN e.name AS name, e._resource AS resource,
                       e.type AS type, e.description AS description
                ORDER BY e._resource DESC
                LIMIT 50
                """,
            ).data()

    def _collect_paths(self, seed_names: list[str]) -> list[dict]:
        with self._driver.session() as session:
            rows = session.run(
                """
                MATCH path = (seed:Entity)-[:RELATES_TO*1..4]-(target:Entity)
                WHERE seed.name IN $seeds
                  AND target._resource IS NOT NULL
                  AND seed <> target
                  AND ALL(n IN nodes(path) WHERE n._resource IS NOT NULL)
                  AND ALL(i IN range(0, size(nodes(path))-2)
                    WHERE NOT nodes(path)[i] IN nodes(path)[(i+1)..])
                WITH path,
                     reduce(w = 0.0, n IN nodes(path) | w + coalesce(n._resource, 0))
                         / size(nodes(path)) AS path_weight,
                     [n IN nodes(path) | n.name] AS names,
                     [i IN range(0, size(relationships(path))-1) |
                         {relation: relationships(path)[i].relation,
                          forward: startNode(relationships(path)[i]) = nodes(path)[i]}
                     ] AS edges
                RETURN names, edges, path_weight
                ORDER BY path_weight DESC
                LIMIT $k
                """,
                seeds=seed_names,
                k=self._top_k_paths,
            ).data()

            return self._deduplicate_paths(rows)

    @staticmethod
    def _deduplicate_paths(rows: list[dict]) -> list[dict]:
        seen_directions: set[tuple] = set()
        candidates: list[dict] = []
        for row in rows:
            key = tuple(row["names"])
            rev_key = tuple(reversed(row["names"]))
            if key in seen_directions or rev_key in seen_directions:
                continue
            seen_directions.add(key)
            candidates.append(row)

        kept_keys: set[tuple] = set()
        result: list[dict] = []
        for path in candidates:
            names = tuple(path["names"])
            is_subpath = False
            for other in candidates:
                other_names = tuple(other["names"])
                if names == other_names:
                    continue
                if len(names) < len(other_names) and _is_prefix(names, other_names):
                    is_subpath = True
                    break
            if not is_subpath:
                kept_keys.add(names)
                result.append(path)

        return result

    def _cleanup(self) -> None:
        with self._driver.session() as session:
            session.run("MATCH (e:Entity) WHERE e._resource IS NOT NULL REMOVE e._resource")


def _is_prefix(short: tuple, long: tuple) -> bool:
    return long[:len(short)] == short or long[-len(short):] == tuple(reversed(short))
