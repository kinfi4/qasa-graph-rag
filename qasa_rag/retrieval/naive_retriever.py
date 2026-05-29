from neo4j import Driver

from qasa_rag.embedder import Embedder

from .retriever import RetrievalResult


class NaiveVectorRetriever:
    """Baseline: pure vector similarity search, no graph traversal."""

    def __init__(
        self,
        driver: Driver,
        embedder: Embedder,
        top_k: int = 30,
    ) -> None:
        self._driver = driver
        self._embedder = embedder
        self._top_k = top_k

    def retrieve(self, question: str) -> RetrievalResult:
        query_embedding = self._embedder.embed(question)

        with self._driver.session() as session:
            rows = session.run(
                """
                CALL db.index.vector.queryNodes(
                    'entity_description_embeddings', $k, $emb
                )
                YIELD node, score
                RETURN node.name AS name, score,
                       node.type AS type, node.description AS description
                """,
                k=self._top_k,
                emb=query_embedding,
            ).data()

        return RetrievalResult(
            entities=[
                {
                    "name": r["name"],
                    "resource": r["score"],
                    "type": r["type"],
                    "description": r["description"],
                }
                for r in rows
            ],
        )
