from neo4j import Driver


class RetrieverDebugger:
    def __init__(self, driver: Driver, decay: float, threshold: float) -> None:
        self._driver = driver
        self._decay = decay
        self._threshold = threshold

    def log_seeds(self, seeds: list[dict]) -> None:
        print(f"[Debug] {len(seeds)} seeds:")
        for s in seeds:
            print(f"  {s['name']} (resource={s['resource']:.4f})")

    def log_flows(self, step: int, query_embedding: list[float]) -> None:
        with self._driver.session() as session:
            rows = session.run(
                """
                MATCH (src:Entity) WHERE src._resource > $threshold
                MATCH (src)-[r:RELATES_TO]-(dst:Entity)
                WHERE dst.description_embedding IS NOT NULL
                  AND (dst._resource IS NULL OR dst._resource < src._resource)
                WITH src.name AS src_name, dst.name AS dst_name, r.relation AS rel,
                     startNode(r) = src AS forward,
                     gds.similarity.cosine(dst.description_embedding, $qe) AS sim,
                     src._resource AS parent_res
                WITH src_name, dst_name, rel, forward, sim,
                     parent_res * $decay * CASE WHEN sim > 0 THEN sim ELSE 0 END AS contrib
                WHERE contrib > 0
                RETURN src_name, rel, dst_name, forward,
                       round(sim * 10000) / 10000.0 AS sim,
                       round(contrib * 10000) / 10000.0 AS contrib
                ORDER BY contrib DESC
                LIMIT 15
                """,
                qe=query_embedding,
                decay=self._decay,
                threshold=self._threshold,
            ).data()

        print(f"\n[Debug] Step {step} — top flows about to propagate:")
        if not rows:
            print("  (none)")
            return

        for r in rows:
            arrow = f"--[{r['rel']}]-->" if r["forward"] else f"<--[{r['rel']}]--"
            print(f"  {r['src_name']} {arrow} {r['dst_name']}  sim={r['sim']:.3f} contrib={r['contrib']:.4f}")

    def log_state(self, step: int, active: int) -> None:
        with self._driver.session() as session:
            rows = session.run("""
                MATCH (e:Entity) WHERE e._resource IS NOT NULL
                RETURN e.name AS name,
                       round(e._resource * 10000) / 10000.0 AS res
                ORDER BY e._resource DESC LIMIT 20
            """).data()

        print(f"\n[Debug] Step {step} result: {active} entities updated")
        for r in rows:
            print(f"  {r['res']:.4f}  {r['name']}")
