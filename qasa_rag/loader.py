import pickle
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path
from typing import Any

from tqdm import tqdm

from qasa_rag.embedder import Embedder
from qasa_rag.entity_resolver import EntityResolver
from qasa_rag.extractor import ExtractionResult, TripleExtractor
from qasa_rag.neo4j_client import Neo4jClient


class KnowledgeGraphLoader:
    """Loads MuSiQue-style examples into a Neo4j knowledge graph.

    Expected example format::

        {
            "question": str,
            "answer": str,
            "paragraphs": [
                {"title": str, "text": str, "is_supporting": bool},
                ...
            ],
        }
    """

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        cache_dir: Path = Path("cache"),
        llm_model: str = "gemini-2.5-flash",
        embedding_model: str = "text-embedding-004",
    ) -> None:
        self._neo4j = Neo4jClient(neo4j_uri, neo4j_user, neo4j_password)
        self._llm_model = llm_model
        self._extractor = TripleExtractor(model_id=llm_model)
        self._embedder = Embedder(
            model_id=embedding_model,
            cache_path=cache_dir / "embeddings_cache.pkl",
        )
        self._entity_embedder = Embedder(
            model_id=embedding_model,
            cache_path=cache_dir / "entity_embeddings_cache.pkl",
        )
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(exist_ok=True)
        self._extractions_cache: dict[str, ExtractionResult] = {}

        extractions_path = cache_dir / "extractions_cache.pkl"
        if extractions_path.exists():
            with open(extractions_path, "rb") as f:
                self._extractions_cache = pickle.load(f)
            print(f"[Loader] Loaded {len(self._extractions_cache)} cached extractions")

        # (title, text) -> paragraph_id, built during load_examples
        self._paragraph_id_map: dict[tuple[str, str], str] = {}
        # title -> hash key for extraction cache
        self._document_cache_key_map: dict[str, str] = {}

    def clear_and_init(self) -> None:
        self._neo4j.clear_database()
        self._neo4j.create_indexes()

    def finalize(self) -> None:
        self._neo4j.create_vector_index()
        self._neo4j.create_entity_vector_index()

    def close(self) -> None:
        self._neo4j.close()

    def load_examples(
        self,
        examples: list[dict[str, Any]],
        max_extraction_workers: int = 3,
        max_embedding_workers: int = 5,
        save_every: int = 100,
    ) -> list[dict[str, Any]]:
        self._collect_paragraphs(examples)
        documents = self._collect_documents(examples)
        print(f"[Loader] Total unique paragraphs: {len(self._paragraph_id_map)}")
        print(f"[Loader] Total unique documents (titles): {len(documents)}")

        print("\n[Loader] Phase 1: Extracting entities and relations...")
        self._extract_all(documents, max_extraction_workers, save_every)

        # print("\n[Loader] Phase 1.5: Entity resolution...")
        # self._resolve_entities()

        print("\n[Loader] Phase 2: Loading into Neo4j...")
        ground_truth = self._load_into_neo4j(examples)

        print("\n[Loader] Phase 3: Post-processing entities...")
        self._neo4j.update_entity_doc_count()
        self._embed_entity_descriptions(max_embedding_workers, save_every)

        # print("\n[Loader] Phase 4: Embedding paragraphs...")
        # self._embed_paragraphs(max_embedding_workers, save_every)

        return ground_truth

    def embed_paragraphs(
        self,
        max_workers: int = 5,
        save_every: int = 500,
    ) -> None:
        """Standalone paragraph embedding phase. Idempotent via text_hash.

        Safe to call without running load_examples(), as long as paragraphs
        already exist in Neo4j.
        """
        self._embed_paragraphs(max_workers, save_every)

    def _collect_paragraphs(self, examples: list[dict[str, Any]]) -> list[str]:
        """Build (title, text) -> paragraph_id mapping and return unique texts."""
        unique_texts: set[str] = set()

        for example in examples:
            for paragraph in example["paragraphs"]:
                title = paragraph["title"]
                text = paragraph["text"]
                key = (title, text)

                if key not in self._paragraph_id_map:
                    paragraph_hash = self._hash_text(f"{title}\n{text}")[:10]
                    self._paragraph_id_map[key] = f"{title}__{paragraph_hash}"
                    unique_texts.add(text)

        return list(unique_texts)

    def _collect_documents(self, examples: list[dict[str, Any]]) -> dict[str, str]:
        """Merge all unique paragraphs per title into one text for extraction."""
        title_paragraphs: dict[str, list[str]] = defaultdict(list)
        title_seen: dict[str, set[str]] = defaultdict(set)

        for example in examples:
            for paragraph in example["paragraphs"]:
                title = paragraph["title"]
                text = paragraph["text"]
                if text not in title_seen[title]:
                    title_seen[title].add(text)
                    title_paragraphs[title].append(text)

        documents: dict[str, str] = {}
        for title, texts in title_paragraphs.items():
            documents[title] = f"**{title}**\n" + "\n".join(texts)

        return documents

    def _extract_all(
        self,
        documents: dict[str, str],
        max_workers: int,
        save_every: int,
    ) -> None:
        for title, doc_text in documents.items():
            self._document_cache_key_map[title] = self._hash_text(
                f"{self._llm_model}\n{title}\n{doc_text}"
            )

        remaining = {
            title: doc_text
            for title, doc_text in documents.items()
            if self._document_cache_key_map[title] not in self._extractions_cache
        }
        print(f"[Loader] Need to extract {len(remaining)} ({len(self._extractions_cache)} cached)")

        if not remaining:
            return

        cache_path = self._cache_dir / "extractions_cache.pkl"
        processed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._extractor.extract, doc_text): (
                    title,
                    self._document_cache_key_map[title],
                )
                for title, doc_text in remaining.items()
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Extraction"):
                title, cache_key = futures[future]
                try:
                    self._extractions_cache[cache_key] = future.result()
                except Exception as e:
                    print(f"\n[Error] Extraction failed for: {title} - {e}")
                    self._extractions_cache[cache_key] = ExtractionResult(entities=[], triples=[])

                processed += 1
                if processed % save_every == 0:
                    with open(cache_path, "wb") as f:
                        pickle.dump(self._extractions_cache, f)
                    print(f"\n[Saved] {len(self._extractions_cache)} extractions")

        with open(cache_path, "wb") as f:
            pickle.dump(self._extractions_cache, f)
        print(f"[Loader] Saved {len(self._extractions_cache)} extractions")

    def _resolve_entities(self) -> None:
        resolver = EntityResolver(llm_model=self._llm_model)

        if (mapping := resolver.resolve(self._extractions_cache)):
            self._extractions_cache = resolver.apply(self._extractions_cache, mapping)

            with open(self._cache_dir / "extractions_cache.pkl", "wb") as f:
                pickle.dump(self._extractions_cache, f)

            print(f"[Loader] Saved resolved extractions ({len(mapping)} aliases merged)")

    def _load_into_neo4j(
        self,
        examples: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        ground_truth = []

        for i, example in enumerate(tqdm(examples, desc="Loading Neo4j")):
            self._load_example(example)

            supporting_ids = [
                self._paragraph_id_map[(paragraph["title"], paragraph["text"])]
                for paragraph in example["paragraphs"]
                if paragraph["is_supporting"]
            ]

            ground_truth.append({
                "question_id": f"q_{i}",
                "question": example["question"],
                "answer": example["answer"],
                "supporting_paragraphs": supporting_ids,
            })

        return ground_truth

    def _load_example(
        self,
        example: dict[str, Any],
    ) -> None:
        matched_entities_by_title: dict[str, set[str]] = defaultdict(set)

        for paragraph in example["paragraphs"]:
            title = paragraph["title"]
            text = paragraph["text"]
            paragraph_id = self._paragraph_id_map[(title, text)]

            self._neo4j.create_document(title)
            self._neo4j.create_paragraph(paragraph_id, text, title)

            extraction_cache_key = self._document_cache_key_map.get(title)
            extraction = self._extractions_cache.get(
                extraction_cache_key,
                ExtractionResult(entities=[], triples=[]),
            )

            if extraction is None:
                continue

            for entity in extraction.entities:
                self._neo4j.create_entity(entity.name, entity.type, entity.description)

            for triple in extraction.triples:
                self._neo4j.create_entity(triple.subject)
                self._neo4j.create_entity(triple.object)
                self._neo4j.create_relation(triple.subject, triple.relation, triple.object)

            text_lower = text.lower()
            for entity in extraction.entities:
                if entity.name.lower() in text_lower:
                    self._neo4j.create_mention(paragraph_id, entity.name)
                    matched_entities_by_title[title].add(entity.name)

            for triple in extraction.triples:
                if triple.subject.lower() in text_lower:
                    self._neo4j.create_mention(paragraph_id, triple.subject)
                if triple.object.lower() in text_lower:
                    self._neo4j.create_mention(paragraph_id, triple.object)

        seen_titles: set[str] = set()
        for paragraph in example["paragraphs"]:
            title = paragraph["title"]
            if title in seen_titles:
                continue
            seen_titles.add(title)

            extraction_cache_key = self._document_cache_key_map.get(title)
            extraction = self._extractions_cache.get(
                extraction_cache_key,
                ExtractionResult(entities=[], triples=[]),
            )
            if extraction is None:
                continue

            for entity in extraction.entities:
                if entity.name not in matched_entities_by_title[title]:
                    self._neo4j.create_mention_to_document(title, entity.name)

    def get_stats(self) -> dict[str, int]:
        return self._neo4j.get_stats()

    def _embed_entity_descriptions(
        self,
        max_workers: int,
        save_every: int,
    ) -> None:
        entities = self._neo4j.get_entities_for_embedding()
        pending_entities: list[dict[str, str]] = []

        for entity in entities:
            name = entity["name"]
            description = entity["description"]
            stored_hash = entity["description_hash"]
            if not name or not description:
                continue

            canonical_description = self._canonicalize_description(description)
            canonical_description = f"**{name.title()}**: {canonical_description}"

            description_hash = self._hash_text(canonical_description)
            if stored_hash == description_hash:
                continue

            pending_entities.append({
                "name": name,
                "description": canonical_description,
                "description_hash": description_hash,
            })

        if not pending_entities:
            print("[Loader] Entity description embeddings are up to date")
            return

        print(f"[Loader] Need to embed {len(pending_entities)} entity descriptions")
        embeddings = self._entity_embedder.embed_batch(
            [entity["description"] for entity in pending_entities],
            max_workers=max_workers,
            save_every=save_every,
        )

        for entity in pending_entities:
            self._neo4j.set_entity_embedding(
                name=entity["name"],
                description_hash=entity["description_hash"],
                embedding=embeddings[entity["description"]],
            )

    def _embed_paragraphs(
        self,
        max_workers: int,
        save_every: int,
    ) -> None:
        paragraphs = self._neo4j.get_paragraphs_for_embedding()
        pending: list[dict[str, str]] = []

        for paragraph in paragraphs:
            paragraph_id = paragraph["id"]
            text = paragraph["text"]
            doc_title = paragraph["doc_title"]
            stored_hash = paragraph["text_hash"]
            if not paragraph_id or not text:
                continue

            canonical_text = self._canonicalize_paragraph(doc_title, text)
            text_hash = self._hash_text(canonical_text)
            if stored_hash == text_hash:
                continue

            pending.append({
                "id": paragraph_id,
                "text": canonical_text,
                "text_hash": text_hash,
            })

        if not pending:
            print("[Loader] Paragraph embeddings are up to date")
            return

        print(f"[Loader] Need to embed {len(pending)} paragraphs")
        embeddings = self._embedder.embed_batch(
            [p["text"] for p in pending],
            max_workers=max_workers,
            save_every=save_every,
        )

        for paragraph in pending:
            self._neo4j.set_paragraph_embedding(
                paragraph_id=paragraph["id"],
                text_hash=paragraph["text_hash"],
                embedding=embeddings[paragraph["text"]],
            )

    def _canonicalize_paragraph(self, doc_title: str | None, text: str) -> str:
        if doc_title:
            return f"**{doc_title}**: {text.strip()}"

        return text.strip()

    def _canonicalize_description(self, description: str) -> str:
        chunks = [
            chunk.strip()
            for chunk in description.split(" | ")
            if chunk.strip()
        ]
        unique_chunks = sorted(set(chunks))
        return " | ".join(unique_chunks)

    def _hash_text(self, text: str) -> str:
        return sha256(text.encode("utf-8")).hexdigest()
