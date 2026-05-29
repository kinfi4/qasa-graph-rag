import pickle
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path

from tqdm import tqdm
from google.api_core import exceptions as google_exceptions

from qasa_rag.client import create_genai_client


class Embedder:
    def __init__(
        self,
        model_id: str = "text-embedding-004",
        cache_path: Path | None = None,
        max_retries: int = 5,
        base_delay: float = 2.0,
    ) -> None:
        self._client = create_genai_client()
        self._model_id = model_id
        self._cache_path = cache_path
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._cache: dict[str, list[float]] = {}

        if cache_path and cache_path.exists():
            with open(cache_path, "rb") as f:
                self._cache = pickle.load(f)
            self._migrate_legacy_cache_keys()
            print(f"[Embedder] Loaded {len(self._cache)} cached embeddings")

    def embed(self, text: str) -> list[float]:
        cache_key = self._cache_key(text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        embedding = self._embed_with_retry(text)
        self._cache[cache_key] = embedding
        return embedding

    def embed_batch(
        self,
        texts: list[str],
        max_workers: int = 5,
        save_every: int = 500,
    ) -> dict[str, list[float]]:
        text_to_cache_key = {text: self._cache_key(text) for text in texts}
        remaining = [text for text in texts if text_to_cache_key[text] not in self._cache]
        print(f"[Embedder] Need {len(remaining)} embeddings ({len(self._cache)} cached)")

        if not remaining:
            return {
                text: self._cache[text_to_cache_key[text]]
                for text in texts
            }

        processed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._embed_with_retry, t): t for t in remaining}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Embeddings"):
                text = futures[future]
                cache_key = text_to_cache_key[text]
                self._cache[cache_key] = future.result()
                processed += 1

                if save_every and processed % save_every == 0:
                    self._save_cache()

        self._save_cache()
        return {
            text: self._cache[text_to_cache_key[text]]
            for text in texts
        }

    def _embed_with_retry(self, text: str) -> list[float]:
        for attempt in range(self._max_retries):
            try:
                result = self._client.models.embed_content(
                    model=self._model_id,
                    contents=text,
                )
                return result.embeddings[0].values

            except (google_exceptions.ResourceExhausted, google_exceptions.TooManyRequests):
                delay = self._base_delay * (2**attempt)

                # if random.random() < 0.2:
                #     print(f"[Throttled] Waiting {delay:.1f}s (attempt {attempt + 1})")

                time.sleep(delay)
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    delay = self._base_delay * (2**attempt)
                    print(f"[Throttled] Waiting {delay:.1f}s (attempt {attempt + 1})")
                    time.sleep(delay)
                else:
                    raise

        raise RuntimeError(f"Embedding failed after {self._max_retries} retries")

    def _save_cache(self) -> None:
        if self._cache_path:
            with open(self._cache_path, "wb") as f:
                pickle.dump(self._cache, f)
            print(f"[Embedder] Saved {len(self._cache)} embeddings")

    def _cache_key(self, text: str) -> str:
        payload = f"{self._model_id}\n{text}".encode("utf-8")
        return sha256(payload).hexdigest()

    def _migrate_legacy_cache_keys(self) -> None:
        updated_cache: dict[str, list[float]] = {}

        for key, value in self._cache.items():
            if len(key) == 64 and all(char in "0123456789abcdef" for char in key):
                updated_cache[key] = value
                continue

            migrated_key = self._cache_key(key)
            if migrated_key not in updated_cache:
                updated_cache[migrated_key] = value

        self._cache = updated_cache
