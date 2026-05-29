# Results

All numbers use `QASARetriever` (single-Cypher, stateless, parallel-safe) with
Gemini 2.5 Flash (extraction + generation + judge) and text-embedding-004 (768-d)
on Neo4j 5 Community + GDS + APOC.

Default config: `max_steps=3, decay=0.7, query_aware=True, top_k=30, threshold=0.01`.
95% CI at n=500 (F1 ≈ 0.5): ±0.044.

## Knowledge-graph statistics

| Dataset | Documents | Paragraphs | Entities | Mentions | Relations |
|---------|-----------|------------|----------|----------|-----------|
| 2WikiMultiHopQA | 6 322 | 6 367 | 31 143 | 44 535 | 40 784 |
| MuSiQue | 10 785 | 12 493 | 55 823 | 97 283 | 76 107 |

## Headline (n=1000, default config)

| Dataset | EM | F1 | LLM-acc | Prompt tok |
|---------|------|------|---------|------------|
| MuSiQue | 0.328 | 0.417 | 0.380 | 11 838 |
| 2WikiMultiHopQA | 0.478 | 0.550 | 0.540 | 4 532 |

## MuSiQue ablation (n=500)

`steps-3` and `decay-0.7` are the same default config (stochasticity check).
Retrieval latency (serial) is reliable only for the ✓ rows; decay-sweep latencies
come from parallel runs and are not quoted in the paper.

| Config | EM | F1 | Retrieval (s) | Latency reliable | Prompt tok | Paths | Entities |
|--------|------|------|---------------|:----------------:|------------|-------|----------|
| **steps-3 / decay-0.7 (default)** | **0.366** | **0.467** | **1.453** | ✓ | 14 130 | 16.1 | 27.9 |
| steps-2 | 0.344 | 0.448 | 0.587 | ✓ | 9 494 | 14.0 | 23.5 |
| decay-0.5 | 0.360 | 0.465 | — | | 12 796 | 15.3 | 26.8 |
| decay-1.0 | 0.364 | 0.455 | — | | 15 020 | 16.6 | 28.0 |
| decay-0.9 | 0.346 | 0.443 | — | | 14 939 | 16.6 | 28.0 |
| steps-4 (n=499) | 0.343 | 0.446 | 3.506 | ✓ | 16 727 | 15.9 | 28.5 |
| decay-0.3 | 0.314 | 0.427 | — | | 9 886 | 14.3 | 23.6 |
| no-query-aware | 0.282 | 0.382 | 2.156 | ✓ | 17 976 | 16.5 | 28.0 |
| naive-vector | 0.280 | 0.362 | 0.374 | ✓ | 1 931 | 0.0 | 30.0 |
| steps-1 | 0.224 | 0.322 | 0.032 | ✓ | 3 801 | 9.2 | 12.5 |

## 2WikiMultiHopQA ablation (n=500)

| Config | EM | F1 | Retrieval (s) | Latency reliable | Prompt tok | Paths | Entities |
|--------|------|------|---------------|:----------------:|------------|-------|----------|
| steps-4 | 0.506 | 0.575 | 0.304 | ✓ | 4 749 | 18.3 | 28.8 |
| decay-1.0 | 0.504 | 0.574 | — | | 4 621 | 18.7 | 28.4 |
| steps-2 | 0.506 | 0.572 | 0.021 | ✓ | 3 617 | 16.2 | 24.3 |
| decay-0.3 | 0.502 | 0.572 | — | | 3 684 | 16.6 | 24.7 |
| **steps-3 / decay-0.7 (default)** | **0.490** | **0.559** | **0.090** | ✓ | 4 505 | 18.0 | 28.3 |
| decay-0.5 | 0.484 | 0.551 | — | | 4 242 | 17.4 | 27.3 |
| decay-0.9 | 0.476 | 0.546 | — | | 4 587 | 18.5 | 28.4 |
| no-query-aware | 0.450 | 0.518 | 0.443 | ✓ | 5 137 | 19.1 | 28.6 |
| naive-vector | 0.350 | 0.399 | 0.325 | ✓ | 1 782 | 0.0 | 30.0 |
| steps-1 | 0.330 | 0.384 | 0.008 | ✓ | 1 605 | 10.4 | 11.7 |

## Query-aware vs uniform propagation (serial latency)

Both measured serially (`5-retrieval-latency-benchmark.ipynb`, `MAX_WORKERS=1`).
Query-aware adds a per-neighbor cosine gate that prunes the activation frontier;
uniform propagation has no gate.

| Dataset | QA default (s) | no-query-aware (s) | Speedup | ΔF1 (QA − no-QA) |
|---------|----------------|--------------------|---------|-------------------|
| MuSiQue | 1.453 | 2.156 | 1.5× | +8.5 |
| 2WikiMultiHopQA | 0.090 | 0.443 | 4.9× | +4.2 |

The query-aware gate improves F1 on both datasets. The latency advantage is large
on 2wiki and modest on MuSiQue; absolute timings are sensitive to Neo4j cache
warmth and machine load.

## Parallel-safety

`QASARetriever` produces results identical to serial execution across worker
counts (fingerprint match on seeds/entities/paths) with a wall-clock speedup as
workers increase — see `4-parallel-safety-test.ipynb`.

## Comparison with prior work

Competitor numbers are from QAFD-RAG (evaluated with GPT-4o-mini); ours uses
Gemini 2.5 Flash, so the comparison is indicative rather than head-to-head.

| Method | MuSiQue F1 | MuSiQue EM | 2wiki F1 | 2wiki EM |
|--------|------------|------------|----------|----------|
| GraphRAG | 39.40 | 17.60 | 15.20 | 7.00 |
| LightRAG | 1.40 | 0.10 | 8.20 | 1.00 |
| HippoRAG | 38.33 | 27.55 | **70.33** | **61.16** |
| QAFD-RAG | **47.99** | 33.50 | 69.41 | 59.50 |
| **Ours (n=500, default)** | 46.71 | **36.60** | 55.91 | 49.00 |
| Ours (n=1000, default) | 41.68 | 32.80 | 55.00 | 47.80 |
