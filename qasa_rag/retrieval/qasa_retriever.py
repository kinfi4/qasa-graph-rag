from .retriever import QueryAwareRetriever, RetrievalResult


_SEED_PREFIX = """
UNWIND $entity_data AS ent
OPTIONAL MATCH (exact:Entity {name: ent.name})

CALL (ent, exact) {
    UNWIND CASE WHEN exact IS NOT NULL THEN [exact] ELSE [] END AS e
    RETURN e.name AS seed_name, 1.0 AS seed_score
  UNION ALL
    UNWIND CASE WHEN exact IS NULL THEN [ent] ELSE [] END AS e
    CALL db.index.vector.queryNodes(
        'entity_description_embeddings', $k_per_entity, e.emb
    ) YIELD node, score
    RETURN node.name AS seed_name, score AS seed_score
}

WITH seed_name, max(seed_score) AS seed_score
WITH collect({name: seed_name, score: seed_score}) AS raw,
     max(seed_score) AS mx
WITH [s IN raw | s.name] AS seed_names,
     apoc.map.fromPairs([s IN raw | [s.name, s.score / mx]]) AS resource_map
"""


_PROPAGATION_QA = """
CALL (resource_map) {
    UNWIND keys(resource_map) AS src_name
    WITH src_name, resource_map[src_name] AS src_res, resource_map
    WHERE src_res > $threshold
    MATCH (src:Entity {name: src_name})-[:RELATES_TO]-(dst:Entity)
    WHERE dst.description_embedding IS NOT NULL
      AND (resource_map[dst.name] IS NULL OR resource_map[dst.name] < src_res)
    WITH dst, src_res,
         gds.similarity.cosine(dst.description_embedding, $qe) AS sim
    WITH dst.name AS dst_name,
         sum(src_res * $decay * CASE WHEN sim > 0 THEN sim ELSE 0 END) AS incoming
    WHERE incoming > $threshold
    RETURN collect({name: dst_name, incoming: incoming}) AS deltas
}
WITH seed_names,
     apoc.map.merge(
       resource_map,
       apoc.map.fromPairs(
         [d IN deltas | [d.name, coalesce(resource_map[d.name], 0) + d.incoming]]
       )
     ) AS resource_map
"""


_PROPAGATION_UNIFORM = """
CALL (resource_map) {
    UNWIND keys(resource_map) AS src_name
    WITH src_name, resource_map[src_name] AS src_res, resource_map
    WHERE src_res > $threshold
    MATCH (src:Entity {name: src_name})-[:RELATES_TO]-(dst:Entity)
    WHERE resource_map[dst.name] IS NULL OR resource_map[dst.name] < src_res
    WITH dst.name AS dst_name,
         sum(src_res * $decay) AS incoming
    WHERE incoming > $threshold
    RETURN collect({name: dst_name, incoming: incoming}) AS deltas
}
WITH seed_names,
     apoc.map.merge(
       resource_map,
       apoc.map.fromPairs(
         [d IN deltas | [d.name, coalesce(resource_map[d.name], 0) + d.incoming]]
       )
     ) AS resource_map
"""


_COLLECT_SUFFIX = """
CALL (seed_names, resource_map) {
    MATCH path = (seed:Entity)-[:RELATES_TO*1..4]-(target:Entity)
    WHERE seed.name IN seed_names
      AND resource_map[target.name] IS NOT NULL
      AND seed <> target
      AND ALL(n IN nodes(path) WHERE resource_map[n.name] IS NOT NULL)
      AND ALL(i IN range(0, size(nodes(path))-2)
          WHERE NOT nodes(path)[i] IN nodes(path)[(i+1)..])
    WITH reduce(w = 0.0, n IN nodes(path) | w + coalesce(resource_map[n.name], 0))
             / size(nodes(path)) AS path_weight,
         [n IN nodes(path) | n.name] AS names,
         [i IN range(0, size(relationships(path))-1) |
             {relation: relationships(path)[i].relation,
              forward: startNode(relationships(path)[i]) = nodes(path)[i]}
         ] AS edges
    ORDER BY path_weight DESC
    LIMIT $top_k_paths
    RETURN collect({names: names, edges: edges, path_weight: path_weight}) AS paths
}

CALL (resource_map) {
    UNWIND keys(resource_map) AS name
    WITH name, resource_map[name] AS resource
    ORDER BY resource DESC LIMIT $top_k_entities
    MATCH (e:Entity {name: name})
    RETURN collect({
        name: e.name, resource: resource,
        type: e.type, description: e.description
    }) AS entities
}

RETURN seed_names, paths, entities
"""


class QASARetriever(QueryAwareRetriever):
    """Query-Aware Spreading Activation retriever — the production implementation.

    Runs the entire retrieval procedure (seed mapping, ``max_steps`` propagation
    steps, top-K path and entity collection) as a *single* Cypher round-trip to
    Neo4j, without ever mutating the graph. Per-entity activation is carried
    inside the query as a Cypher ``MAP`` variable (``resource_map``) that flows
    through ``WITH`` clauses; the propagation block is concatenated ``max_steps``
    times at query-build time.

    Compared to the stateful reference :class:`~qasa_rag.retrieval.QueryAwareRetriever`,
    this implementation is:

    * **Parallel-safe** — concurrent retrievals operate on independent local maps
      and cannot corrupt each other's state, so evaluation can be threaded.
    * **Crash-safe** — an aborted query leaves zero residual state on the graph.
    * **Cleanup-free** — no ``_resource`` property is ever written, so no
      ``REMOVE`` pass is required.

    The algorithm is otherwise identical to the reference variant; end-to-end
    metrics (EM / F1 / LLM-accuracy) match within LLM stochasticity. Requires the
    APOC plugin (``apoc.map.fromPairs`` and ``apoc.map.merge``) and the GDS plugin
    (``gds.similarity.cosine``).
    """

    def retrieve(self, question: str) -> RetrievalResult:
        query_entities = self._extract_query_entities(question)
        if not query_entities:
            return RetrievalResult()

        query_embedding = self._embedder.embed(question)

        entity_data = [
            {"name": entity.name, "emb": self._embedder.embed(entity.search_hint)}
            for entity in query_entities
        ]

        k_per_entity = max(1, self._seed_fallback_k // len(query_entities))

        propagation = _PROPAGATION_QA if self._query_aware else _PROPAGATION_UNIFORM
        full_query = _SEED_PREFIX + (propagation * self._max_steps) + _COLLECT_SUFFIX

        with self._driver.session() as session:
            record = session.run(
                full_query,
                entity_data=entity_data,
                k_per_entity=k_per_entity,
                qe=query_embedding,
                decay=self._decay,
                threshold=self._threshold,
                top_k_entities=self._top_k_entities,
                top_k_paths=self._top_k_paths,
            ).single()

        if not record:
            return RetrievalResult(extracted_entities=query_entities)

        return RetrievalResult(
            paths=self._deduplicate_paths(record["paths"]),
            entities=record["entities"],
            seed_entities=record["seed_names"],
            extracted_entities=query_entities,
            steps_taken=self._max_steps,
        )
