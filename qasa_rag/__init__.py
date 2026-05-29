"""QASA-RAG: Query-Aware Spreading Activation for multi-hop retrieval over knowledge graphs.

Public API:

* :class:`~qasa_rag.loader.KnowledgeGraphLoader` builds the Neo4j knowledge graph
  from a document corpus (entity/relation extraction, resolution, embedding).
* :class:`~qasa_rag.retrieval.QASARetriever` is the production retriever: a single,
  stateless, parallel-safe Cypher round-trip implementing query-aware spreading
  activation.
* :class:`~qasa_rag.retrieval.QueryAwareRetriever` is the stateful reference
  implementation (supports verbose step-by-step tracing).
* :class:`~qasa_rag.retrieval.NaiveVectorRetriever` is the vector-search baseline.
* :class:`~qasa_rag.retrieval.AnswerAgent` generates and judges answers.
* :class:`~qasa_rag.retrieval.AblationEvaluator` reproduces the ablation results.
"""

from qasa_rag.loader import KnowledgeGraphLoader
from qasa_rag.retrieval import (
    AblationConfig,
    AblationEvaluator,
    AnswerAgent,
    NaiveVectorRetriever,
    QASARetriever,
    QueryAwareRetriever,
)

__all__ = [
    "KnowledgeGraphLoader",
    "QASARetriever",
    "QueryAwareRetriever",
    "NaiveVectorRetriever",
    "AnswerAgent",
    "AblationConfig",
    "AblationEvaluator",
]
