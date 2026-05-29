import re
import string
import time
from collections import Counter

from qasa_rag.client import create_genai_client

from .retriever import QueryAwareRetriever, RetrievalResult


JUDGE_PROMPT = """Given the question and the expected answer, is the predicted answer correct?
Consider semantic equivalence: different phrasing, abbreviations, or minor variations of the same answer should count as correct.

Question: {question}
Expected answer: {expected}
Predicted answer: {predicted}

Answer only YES or NO."""


ANSWER_PROMPT = """Answer the question using ONLY the knowledge graph below.

Path notation:
- A --[relation]--> B means "A relation B" (e.g. einstein --[born in]--> ulm means Einstein was born in Ulm)
- A <--[relation]-- B means "B relation A" (e.g. ulm <--[born in]-- einstein means Einstein was born in Ulm)

Knowledge graph paths:
{paths}

Entity descriptions:
{descriptions}

Trace the connections step by step to find the answer.
Output ONLY the answer — a single entity name or short phrase, nothing else.

Question: {question}
Answer:"""


class AnswerAgent:
    def __init__(
        self,
        retriever: QueryAwareRetriever,
        llm_model: str = "gemini-2.5-flash",
        llm_judge: bool = False,
    ) -> None:
        self._retriever = retriever
        self._genai = create_genai_client()
        self._llm_model = llm_model
        self._llm_judge = llm_judge

    def answer(self, question: str) -> dict:
        t_retrieval = time.perf_counter()
        retrieval = self._retriever.retrieve(question)
        retrieval_s = time.perf_counter() - t_retrieval

        paths_text, desc_text = self._build_context(retrieval)

        t_generation = time.perf_counter()
        response = self._genai.models.generate_content(
            model=self._llm_model,
            contents=ANSWER_PROMPT.format(
                paths=paths_text, descriptions=desc_text, question=question,
            ),
            config={"temperature": 0.0, "max_output_tokens": 100, "thinking_config": {"thinking_budget": 50}},
        )
        generation_s = time.perf_counter() - t_generation

        prediction = response.text.strip() if response.text else ""

        usage = response.usage_metadata
        return {
            "question": question,
            "prediction": prediction,
            "paths_context": paths_text,
            "descriptions_context": desc_text,
            "retrieval": retrieval,
            "prompt_tokens": getattr(usage, "prompt_token_count", 0) or 0,
            "completion_tokens": getattr(usage, "candidates_token_count", 0) or 0,
            "total_tokens": getattr(usage, "total_token_count", 0) or 0,
            "retrieval_s": retrieval_s,
            "generation_s": generation_s,
        }

    def evaluate(self, question: str, ground_truth_answer: str) -> dict:
        result = self.answer(question)
        pred = result["prediction"]
        em = compute_exact_match(pred, ground_truth_answer)
        f1 = compute_f1(pred, ground_truth_answer)

        metrics: dict = {"ground_truth": ground_truth_answer, "em": em, "f1": f1}

        if self._llm_judge:
            metrics["llm_accuracy"] = self._judge_accuracy(
                question=question,
                expected=ground_truth_answer,
                predicted=pred,
            )

        return {**result, **metrics}

    @staticmethod
    def flatten_result(result: dict, question_id: str = "") -> dict:
        """Convert an evaluate() result into a flat dict suitable for CSV export."""
        retrieval: RetrievalResult = result.get("retrieval", RetrievalResult())
        return {
            "question_id": question_id,
            "question": result.get("question", ""),
            "ground_truth": result.get("ground_truth", ""),
            "prediction": result.get("prediction", ""),
            "em": result.get("em", 0.0),
            "f1": result.get("f1", 0.0),
            "llm_accuracy": result.get("llm_accuracy"),
            "prompt_tokens": result.get("prompt_tokens", 0),
            "completion_tokens": result.get("completion_tokens", 0),
            "total_tokens": result.get("total_tokens", 0),
            "retrieval_s": result.get("retrieval_s", 0.0),
            "generation_s": result.get("generation_s", 0.0),
            "num_paths": len(retrieval.paths),
            "num_entities": len(retrieval.entities),
            "num_seeds": len(retrieval.seed_entities),
            "steps_taken": retrieval.steps_taken,
            "seed_entities": ", ".join(retrieval.seed_entities),
            "extracted_entities": ", ".join(
                e.name for e in retrieval.extracted_entities
            ),
        }

    def _judge_accuracy(
        self, question: str, expected: str, predicted: str,
    ) -> float:
        if not predicted.strip():
            return 0.0

        response = self._genai.models.generate_content(
            model=self._llm_model,
            contents=JUDGE_PROMPT.format(
                question=question,
                expected=expected,
                predicted=predicted,
            ),
            config={"temperature": 0.0, "max_output_tokens": 25, "thinking_config": {"thinking_budget": 20}},
        )

        verdict = (response.text or "").strip().upper()
        return 1.0 if verdict.startswith("YES") else 0.0

    def _build_context(self, retrieval: RetrievalResult) -> tuple[str, str]:
        if not retrieval.paths and not retrieval.entities:
            return "No paths found.", "No entities found."

        path_lines = []
        for p in retrieval.paths:
            names = p["names"]
            edges = p["edges"]
            chain = names[0]
            for i, edge in enumerate(edges):
                rel = edge["relation"]
                if edge["forward"]:
                    chain += f" --[{rel}]--> '{names[i + 1]}'"
                else:
                    chain += f" <--[{rel}]-- '{names[i + 1]}'"
            path_lines.append(chain)

        desc_lines = []
        for e in retrieval.entities:
            if e.get("description"):
                desc_lines.append(f"- {e['name']}: {e['description']}")

        paths_text = "\n".join(path_lines) if path_lines else "No paths found."
        desc_text = "\n".join(desc_lines) if desc_lines else "No descriptions available."

        return paths_text, desc_text


def _normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    return 1.0 if _normalize_answer(prediction) == _normalize_answer(ground_truth) else 0.0


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gt_tokens = _normalize_answer(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)
