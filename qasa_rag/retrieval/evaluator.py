from __future__ import annotations

import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from neo4j import Driver
from tqdm import tqdm

from qasa_rag.embedder import Embedder

from .agent import AnswerAgent
from .naive_retriever import NaiveVectorRetriever
from .retriever import QueryAwareRetriever, QueryEntity
from .qasa_retriever import QASARetriever


@dataclass
class AblationConfig:
    name: str
    max_steps: int = 3
    decay: float = 0.7
    query_aware: bool = True
    naive_vector: bool = False
    top_k_paths: int = 30
    top_k_entities: int = 30
    seed_vector_fallback_k: int = 5
    resource_threshold: float = 0.01

    def describe(self) -> str:
        if self.naive_vector:
            return f"[{self.name}] naive vector search, top_k={self.top_k_entities}"
        qa = "query-aware" if self.query_aware else "uniform"
        return (
            f"[{self.name}] steps={self.max_steps}, decay={self.decay}, "
            f"{qa}, threshold={self.resource_threshold}"
        )


@dataclass
class RunResult:
    config: AblationConfig
    df: pd.DataFrame
    elapsed_s: float

    @property
    def em(self) -> float:
        return self.df["em"].mean()

    @property
    def f1(self) -> float:
        return self.df["f1"].mean()

    @property
    def llm_acc(self) -> float | None:
        if "llm_accuracy" in self.df.columns and self.df["llm_accuracy"].notna().any():
            return self.df["llm_accuracy"].mean()
        return None


class AblationEvaluator:
    def __init__(
        self,
        driver: Driver,
        embedder: Embedder,
        ground_truth: list[dict],
        output_dir: str = "ablation-results",
        llm_model: str = "gemini-2.5-flash",
        llm_judge: bool = True,
        retriever_cls: type = QueryAwareRetriever,
        query_entity_cache_path: Path | str | None = None,
    ) -> None:
        self._driver = driver
        self._embedder = embedder
        self._ground_truth = ground_truth
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._llm_model = llm_model
        self._llm_judge = llm_judge
        self._retriever_cls = retriever_cls
        self._results: list[RunResult] = []

        self._cache_path = Path(query_entity_cache_path) if query_entity_cache_path else None
        self._query_entity_cache: dict[str, list[QueryEntity]] = {}
        if self._cache_path and self._cache_path.exists():
            with open(self._cache_path, "rb") as f:
                self._query_entity_cache = pickle.load(f)
            print(f"[Evaluator] Loaded {len(self._query_entity_cache)} cached query entities")

    def _save_query_entity_cache(self) -> None:
        if not self._cache_path:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._cache_path, "wb") as f:
            pickle.dump(self._query_entity_cache, f)

    def _build_retriever(self, config: AblationConfig):
        if config.naive_vector:
            return NaiveVectorRetriever(
                driver=self._driver,
                embedder=self._embedder,
                top_k=config.top_k_entities,
            )
        return self._retriever_cls(
            driver=self._driver,
            embedder=self._embedder,
            llm_model=self._llm_model,
            max_steps=config.max_steps,
            decay=config.decay,
            query_aware=config.query_aware,
            resource_threshold=config.resource_threshold,
            top_k_paths=config.top_k_paths,
            top_k_entities=config.top_k_entities,
            seed_vector_fallback_k=config.seed_vector_fallback_k,
            query_entity_cache=self._query_entity_cache,
        )

    def run_config(
        self,
        config: AblationConfig,
        n_eval: int | None = None,
        report_every: int = 25,
        max_workers: int = 1,
    ) -> RunResult:
        """Run a single ablation configuration.

        ``max_workers > 1`` enables parallel evaluation via a thread pool. This
        requires either the naive-vector baseline or an evaluator constructed
        with ``retriever_cls=QASARetriever`` — the stateful reference retriever
        (``QueryAwareRetriever``) mutates shared graph properties and will
        corrupt results under concurrency.
        """
        if max_workers > 1 and not config.naive_vector:
            if not issubclass(self._retriever_cls, QASARetriever):
                raise ValueError(
                    f"Parallel evaluation (max_workers={max_workers}) requires "
                    f"retriever_cls=QASARetriever, got "
                    f"{self._retriever_cls.__name__}. The stateful reference "
                    "retriever mutates shared graph properties (_resource) and "
                    "produces corrupt results when called concurrently."
                )

        print(f"\n{'='*60}")
        print(f"  {config.describe()}")
        if max_workers > 1:
            print(f"  workers={max_workers}")
        print(f"{'='*60}")

        retriever = self._build_retriever(config)
        agent = AnswerAgent(
            retriever=retriever,
            llm_model=self._llm_model,
            llm_judge=self._llm_judge,
        )

        questions = self._ground_truth[:n_eval] if n_eval else self._ground_truth
        t0 = time.time()

        if max_workers > 1:
            rows = self._run_parallel(agent, questions, config.name, max_workers, report_every)
        else:
            rows = self._run_serial(agent, questions, config.name, report_every)

        elapsed = time.time() - t0
        df = pd.DataFrame(rows)

        csv_path = self._output_dir / f"{config.name}.csv"
        df.to_csv(csv_path, index=False)

        self._save_query_entity_cache()

        run = RunResult(config=config, df=df, elapsed_s=elapsed)
        self._results.append(run)

        print(f"\n  [{config.name}] {len(df)} questions in {elapsed:.0f}s")
        print(f"  EM:  {run.em:.4f}")
        print(f"  F1:  {run.f1:.4f}")
        if run.llm_acc is not None:
            print(f"  LLM Acc: {run.llm_acc:.4f}")
        print(f"  Saved: {csv_path}")

        return run

    def _run_serial(
        self,
        agent: AnswerAgent,
        questions: list[dict],
        name: str,
        report_every: int,
    ) -> list[dict]:
        rows: list[dict] = []
        progress = tqdm(questions, desc=name)
        for idx, gt in enumerate(progress, start=1):
            try:
                result = agent.evaluate(gt["question"], gt["answer"])
                rows.append(AnswerAgent.flatten_result(result, gt["question_id"]))

                if idx % report_every == 0:
                    df_partial = pd.DataFrame(rows)
                    progress.set_postfix(
                        EM=f"{df_partial['em'].mean():.3f}",
                        F1=f"{df_partial['f1'].mean():.3f}",
                    )
                    self._save_query_entity_cache()
            except Exception as e:
                print(f"  Error on {gt['question_id']}: {e}")
        return rows

    def _run_parallel(
        self,
        agent: AnswerAgent,
        questions: list[dict],
        name: str,
        max_workers: int,
        report_every: int,
    ) -> list[dict]:
        rows_by_idx: list[dict | None] = [None] * len(questions)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx = {
                pool.submit(agent.evaluate, gt["question"], gt["answer"]): i
                for i, gt in enumerate(questions)
            }
            progress = tqdm(
                as_completed(future_to_idx),
                total=len(future_to_idx),
                desc=name,
            )
            for done_count, fut in enumerate(progress, start=1):
                idx = future_to_idx[fut]
                gt = questions[idx]
                try:
                    result = fut.result()
                    rows_by_idx[idx] = AnswerAgent.flatten_result(result, gt["question_id"])
                except Exception as e:
                    print(f"  Error on {gt['question_id']}: {e}")

                if done_count % report_every == 0:
                    valid_rows = [r for r in rows_by_idx if r is not None]
                    if valid_rows:
                        df_partial = pd.DataFrame(valid_rows)
                        progress.set_postfix(
                            EM=f"{df_partial['em'].mean():.3f}",
                            F1=f"{df_partial['f1'].mean():.3f}",
                            ok=len(valid_rows),
                        )
                    self._save_query_entity_cache()

        return [r for r in rows_by_idx if r is not None]

    def benchmark_retrieval(
        self,
        config: AblationConfig,
        n_eval: int | None = None,
        max_workers: int = 1,
        patch_csv: bool = True,
    ) -> pd.DataFrame:
        """Measure retrieval latency without running the LLM pipeline.

        Calls ``retriever.retrieve(question)`` for each question, records
        ``retrieval_s``, and returns a summary DataFrame. If ``patch_csv=True``
        and a CSV for the config already exists in ``output_dir``, the
        ``retrieval_s`` column is overwritten with the fresh measurements
        (all other columns are left untouched).

        Uses the shared ``query_entity_cache`` so LLM entity extraction is
        skipped for already-seen questions, giving a clean Neo4j-only latency.
        Supports ``max_workers > 1`` with ``QASARetriever``.
        """
        if max_workers > 1 and not config.naive_vector:
            if not issubclass(self._retriever_cls, QASARetriever):
                raise ValueError(
                    f"Parallel benchmark (max_workers={max_workers}) requires "
                    f"retriever_cls=QASARetriever, got "
                    f"{self._retriever_cls.__name__}."
                )

        print(f"\n{'='*60}")
        print(f"  [retrieval-only] {config.describe()}")
        if max_workers > 1:
            print(f"  workers={max_workers}")
        print(f"{'='*60}")

        retriever = self._build_retriever(config)
        questions = self._ground_truth[:n_eval] if n_eval else self._ground_truth

        def _time_one(gt: dict) -> dict:
            t0 = time.perf_counter()
            retriever.retrieve(gt["question"])
            return {"question_id": gt["question_id"], "retrieval_s": time.perf_counter() - t0}

        rows: list[dict] = []
        if max_workers > 1:
            rows_by_idx: list[dict | None] = [None] * len(questions)
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                future_to_idx = {pool.submit(_time_one, gt): i for i, gt in enumerate(questions)}
                for fut in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc=config.name):
                    i = future_to_idx[fut]
                    try:
                        rows_by_idx[i] = fut.result()
                    except Exception as e:
                        print(f"  Error on {questions[i]['question_id']}: {e}")
            rows = [r for r in rows_by_idx if r is not None]
        else:
            for gt in tqdm(questions, desc=config.name):
                try:
                    rows.append(_time_one(gt))
                except Exception as e:
                    print(f"  Error on {gt['question_id']}: {e}")

        timing_df = pd.DataFrame(rows)
        avg = timing_df["retrieval_s"].mean()
        p50 = timing_df["retrieval_s"].median()
        p95 = timing_df["retrieval_s"].quantile(0.95)
        print(f"\n  [{config.name}] n={len(timing_df)}  avg={avg:.3f}s  p50={p50:.3f}s  p95={p95:.3f}s")

        if patch_csv:
            csv_path = self._output_dir / f"{config.name}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                merged = df.merge(
                    timing_df.rename(columns={"retrieval_s": "retrieval_s_new"}),
                    on="question_id",
                    how="left",
                )
                df["retrieval_s"] = merged["retrieval_s_new"].fillna(df.get("retrieval_s", float("nan")))
                df.to_csv(csv_path, index=False)
                print(f"  Patched: {csv_path}")

        return timing_df

    def summary(self) -> pd.DataFrame:
        """Build summary from all CSVs in output_dir (including previous runs)."""
        rows = []
        for csv_path in sorted(self._output_dir.glob("*.csv")):
            df = pd.read_csv(csv_path)
            row = {
                "config": csv_path.stem,
                "n": len(df),
                "em": df["em"].mean(),
                "f1": df["f1"].mean(),
                "avg_prompt_tok": df["prompt_tokens"].mean(),
                "avg_total_tok": df["total_tokens"].mean(),
                "avg_paths": df["num_paths"].mean(),
                "avg_entities": df["num_entities"].mean(),
            }
            if "llm_accuracy" in df.columns and df["llm_accuracy"].notna().any():
                row["llm_acc"] = df["llm_accuracy"].mean()
            if "retrieval_s" in df.columns:
                row["avg_retrieval_s"] = df["retrieval_s"].mean()
            if "generation_s" in df.columns:
                row["avg_generation_s"] = df["generation_s"].mean()
            rows.append(row)

        return pd.DataFrame(rows).sort_values("f1", ascending=False)
