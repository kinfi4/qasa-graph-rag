# QASA-RAG — Query-Aware Spreading Activation for Multi-Hop Retrieval over Knowledge Graphs

Reference implementation for the paper *Query-Aware Knowledge-Graph Traversal for
Multi-Hop Questions with a Cypher-Native Implementation*.

Most Graph-RAG systems are **query-blind** during traversal: the question only
selects seed nodes, and the walk from there depends solely on graph structure.
**QASA** makes the traversal query-aware. Activation spreads from question-derived
seed entities for a fixed number of steps, and each step passes through a
**per-step semantic gate** whose intensity is the cosine similarity between a
candidate entity's description and the question. The entire retrieval procedure —
seed mapping, propagation, and top-K path/entity selection — is expressed as a
**single, stateless Cypher query** that runs in one round-trip to Neo4j, without
loading the graph into Python and without mutating the graph.

## Why the implementation is stateless

`QASARetriever` carries per-entity activation inside the query as a Cypher `MAP`
variable rather than writing a node property. This makes it:

- **Parallel-safe** — concurrent retrievals use independent state, so evaluation
  can be threaded (see `notebooks/4-parallel-safety-test.ipynb`);
- **Crash-safe** — an aborted query leaves no residual state on the graph;
- **Cleanup-free** — no property is ever written, so no cleanup pass is needed.

The package also ships `QueryAwareRetriever`, a stateful *reference*
implementation of the same algorithm with verbose step-by-step tracing, useful
for understanding the propagation dynamics.

## Repository layout

```
qasa_rag/                     Python package
├── client.py                 Google GenAI client (Developer API or Vertex AI)
├── embedder.py               Text/entity embedding with on-disk cache
├── extractor.py              LLM entity/relation extraction
├── entity_resolver.py        Cross-document entity resolution
├── neo4j_client.py           Neo4j write helpers + vector index
├── loader.py                 KnowledgeGraphLoader (end-to-end ingestion)
└── retrieval/
    ├── retriever.py          QueryAwareRetriever (base + stateful reference)
    ├── qasa_retriever.py     QASARetriever (stateless, single-Cypher, production)
    ├── naive_retriever.py    NaiveVectorRetriever (vector-search baseline)
    ├── agent.py              AnswerAgent (generation + LLM-judge + EM/F1)
    ├── debug.py              Step-by-step propagation tracer
    └── evaluator.py          AblationEvaluator (batch + parallel + latency)

notebooks/
├── 1-load-musique-dataset.ipynb   Build the KG for MuSiQue
├── 1-load-2wiki-dataset.ipynb     Build the KG for 2WikiMultiHopQA
├── 2-retrieval-demo.ipynb         Single-question demo + batch evaluation
├── 3-ablation-study.ipynb         Reproduce the ablation tables
├── 4-parallel-safety-test.ipynb   Verify concurrent correctness + speedup
└── 5-retrieval-latency-benchmark.ipynb   Retrieval-only latency (serial)
    ablation-results-musique/      Per-config result CSVs (MuSiQue)
    ablation-results-2wiki/        Per-config result CSVs (2WikiMultiHopQA)

docker-compose.yml            Neo4j 5 Community + GDS + APOC
```

## Setup

### 1. Start Neo4j (with GDS + APOC plugins)

```bash
docker compose up -d
```

This launches Neo4j 5 on `bolt://localhost:7687` (browser at `http://localhost:7474`,
credentials `neo4j` / `password123`) with the Graph Data Science and APOC plugins,
which the retrieval Cypher requires (`gds.similarity.cosine`, `apoc.map.*`).

### 2. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure credentials

```bash
cp .env.example .env
# edit .env and set GOOGLE_API_KEY
```

The pipeline uses **Gemini 2.5 Flash** for entity extraction, answer generation,
and LLM-as-a-judge, and **text-embedding-004** (768-d) for embeddings.

## Reproducing the results

Run the notebooks in order (they expect to be launched from the `notebooks/`
directory, which is where the dataset/cache artifacts are written):

1. **Build the knowledge graph.** Run `1-load-musique-dataset.ipynb` and/or
   `1-load-2wiki-dataset.ipynb`. Each downloads the dataset from HuggingFace,
   extracts entities/relations with the LLM, resolves entities, embeds
   descriptions, and writes everything to Neo4j. It also caches a
   `ground_truth-<dataset>.pkl` used by the later notebooks.
   *This step is LLM-heavy and is the bulk of the wall-clock cost.*
2. **Demo a single question and batch-evaluate.** `2-retrieval-demo.ipynb`.
3. **Run the ablation study.** `3-ablation-study.ipynb` writes one CSV per config
   into `ablation-results-<dataset>/`. Committed CSVs from our runs are included.
4. **Verify parallel-safety.** `4-parallel-safety-test.ipynb`.
5. **Measure retrieval latency.** `5-retrieval-latency-benchmark.ipynb` (serial,
   `MAX_WORKERS=1`, for accurate per-query latency).

Generated artifacts (`notebooks/cache/`, `*.pkl`) are git-ignored and recreated
on first run.

## Headline results

On the standard n = 1000 split with Gemini 2.5 Flash as generator:

| Dataset | EM | F1 |
|---------|------|------|
| MuSiQue | 0.328 | 0.417 |
| 2WikiMultiHopQA | 0.478 | 0.550 |

See [`RESULTS.md`](RESULTS.md) for the full ablation tables, latency figures, and
comparison with prior work.

## Quick API example

```python
from pathlib import Path
from neo4j import GraphDatabase
from qasa_rag import QASARetriever, AnswerAgent
from qasa_rag.embedder import Embedder

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password123"))
embedder = Embedder(cache_path=Path("cache/embeddings_cache.pkl"))

retriever = QASARetriever(driver=driver, embedder=embedder, max_steps=3, decay=0.7)
agent = AnswerAgent(retriever=retriever)

print(agent.answer("Who is the child of Caroline LeRoy's spouse?")["prediction"])
```

## License

Released under the [MIT License](LICENSE).
