import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from google.api_core import exceptions as google_exceptions
from pydantic import BaseModel, model_validator
from tqdm import tqdm

from qasa_rag.client import create_genai_client
from qasa_rag.extractor import Entity, ExtractionResult, Triple


class EntityGroup(BaseModel):
    canonical_name: str
    members: list[str]
    description: str

    @model_validator(mode="after")
    def validate_group(self) -> "EntityGroup":
        if len(self.members) < 2:
            raise ValueError(f"Group must have 2+ members, got {len(self.members)}")

        if self.canonical_name not in self.members:
            raise ValueError(
                f"canonical_name '{self.canonical_name}' not in members {self.members}",
            )

        if not self.description.strip():
            raise ValueError("description must not be empty")

        return self


class ResolutionResult(BaseModel):
    groups: list[EntityGroup]


RESOLUTION_PROMPT = """You are given entity names that may refer to the same real-world entity.

Group entities that refer to the SAME real-world entity. For each group:
1. Pick canonical_name — MUST be exactly one of the listed entity names, do NOT invent new names
2. List ALL member names (including canonical_name)
3. Write a single merged description combining all known facts

Rules:
- Only group entities that are truly the same (e.g. "louis xiii" and "louis xiii of france" = same)
- Do NOT group different entities (e.g. "john smith" and "john doe" = different)
- Only return groups with 2+ members
- If ALL entities are different, return empty groups list

Entities:
{entities}"""


class EntityResolver:
    def __init__(
        self,
        llm_model: str = "gemini-2.5-flash",
        max_retries: int = 5,
        base_delay: float = 2.0,
    ) -> None:
        self._genai = create_genai_client()
        self._llm_model = llm_model
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._descriptions: dict[str, str] = {}

    def resolve(self, extractions: dict[str, ExtractionResult]) -> dict[str, str]:
        entities_by_type = self._collect_entities(extractions)
        clusters = self._build_clusters(entities_by_type)

        if not clusters:
            print("[Resolver] No duplicate candidates found")
            return {}

        print(f"[Resolver] Found {len(clusters)} clusters to resolve")

        mapping: dict[str, str] = {}
        self._descriptions = {}

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(self._resolve_cluster, cluster): cluster
                for cluster in clusters
            }

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Resolving entities",
            ):
                groups = future.result()
                for group in groups:
                    canonical = _normalize(group.canonical_name)
                    self._descriptions[canonical] = group.description
                    for member in group.members:
                        normalized = _normalize(member)
                        if normalized != canonical:
                            mapping[normalized] = canonical

        print(f"[Resolver] Resolved {len(mapping)} entity aliases")
        return mapping

    def apply(
        self,
        extractions: dict[str, ExtractionResult],
        mapping: dict[str, str],
    ) -> dict[str, ExtractionResult]:
        if not mapping:
            return extractions

        resolved: dict[str, ExtractionResult] = {}
        for key, extraction in extractions.items():
            if extraction is None:
                resolved[key] = extraction
                continue
            resolved[key] = _apply_to_extraction(
                extraction, mapping, self._descriptions,
            )

        return resolved

    def _collect_entities(
        self, extractions: dict[str, ExtractionResult],
    ) -> dict[str, list[dict]]:
        entity_info: dict[str, dict] = defaultdict(
            lambda: {"types": defaultdict(int), "descriptions": set()},
        )

        for extraction in extractions.values():
            if extraction is None:
                continue

            for entity in extraction.entities:
                info = entity_info[entity.name]
                info["types"][entity.type] += 1
                if entity.description:
                    info["descriptions"].add(entity.description[:120])

            for triple in extraction.triples:
                entity_info[triple.subject]
                entity_info[triple.object]

        entities_by_type: dict[str, list[dict]] = defaultdict(list)
        for name, info in entity_info.items():
            if not info["types"]:
                continue
            most_common_type = max(info["types"], key=info["types"].get)
            entities_by_type[most_common_type].append({
                "name": name,
                "type": most_common_type,
                "descriptions": list(info["descriptions"]),
            })

        return entities_by_type

    def _build_clusters(
        self, entities_by_type: dict[str, list[dict]],
    ) -> list[list[dict]]:
        all_clusters: list[list[dict]] = []

        for _, entities in entities_by_type.items():
            if len(entities) < 2:
                continue

            names = [e["name"] for e in entities]
            name_to_info = {e["name"]: e for e in entities}
            token_sets = {name: set(name.split()) for name in names}

            parent: dict[str, str] = {}
            rank: dict[str, int] = {}
            for name in names:
                parent[name] = name
                rank[name] = 0

            for i, a in enumerate(names):
                for b in names[i + 1 :]:
                    ta, tb = token_sets[a], token_sets[b]
                    shorter = ta if len(ta) <= len(tb) else tb
                    longer = tb if len(ta) <= len(tb) else ta
                    if shorter.issubset(longer):
                        _union(parent, rank, a, b)

            groups: dict[str, list[str]] = defaultdict(list)
            for name in names:
                groups[_find(parent, name)].append(name)

            for group_names in groups.values():
                if len(group_names) >= 2:
                    all_clusters.append([name_to_info[n] for n in group_names])

        return all_clusters

    def _resolve_cluster(self, cluster: list[dict]) -> list[EntityGroup]:
        entity_lines = []
        for e in cluster:
            descs = e["descriptions"]
            desc_str = " / ".join(descs) if descs else "no description"
            entity_lines.append(
                f'- "{e["name"]}" ({e["type"]}): {desc_str}',
            )

        prompt = RESOLUTION_PROMPT.format(entities="\n".join(entity_lines))

        for attempt in range(self._max_retries):
            try:
                response = self._genai.models.generate_content(
                    model=self._llm_model,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": ResolutionResult,
                        "temperature": 0.0,
                        "thinking_config": {"thinking_budget": 200},
                    },
                )

                if response.parsed is None:
                    return []
                if not response.parsed.groups:
                    return []

                return response.parsed.groups
            except Exception as e:
                delay = self._base_delay * (2**attempt)
                print(f"Retrying with error... {e.__class__.__name__}: {e}")
                time.sleep(delay)

        # safe fallback if LLM can't decide what to do
        print(f"LLM failed to resolve cluster, returning empty list for {[e['name'] for e in cluster]}")
        return []


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _find(parent: dict[str, str], x: str) -> str:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent: dict[str, str], rank: dict[str, int], a: str, b: str) -> None:
    ra, rb = _find(parent, a), _find(parent, b)
    if ra == rb:
        return
    if rank[ra] < rank[rb]:
        ra, rb = rb, ra
    parent[rb] = ra
    if rank[ra] == rank[rb]:
        rank[ra] += 1


def _apply_to_extraction(
    extraction: ExtractionResult,
    mapping: dict[str, str],
    descriptions: dict[str, str],
) -> ExtractionResult:
    entity_map: dict[str, Entity] = {}
    for entity in extraction.entities:
        resolved_name = mapping.get(entity.name, entity.name)
        if resolved_name in entity_map:
            continue

        desc = descriptions.get(resolved_name, entity.description)
        entity_map[resolved_name] = Entity(
            name=resolved_name,
            type=entity.type,
            description=desc,
        )

    resolved_triples: list[Triple] = []
    for triple in extraction.triples:
        subject = mapping.get(triple.subject, triple.subject)
        obj = mapping.get(triple.object, triple.object)
        if subject == obj:
            continue
        resolved_triples.append(Triple(
            subject=subject,
            relation=triple.relation,
            object=obj,
        ))

    return ExtractionResult(
        entities=list(entity_map.values()),
        triples=resolved_triples,
    )
