from typing import Any

from neo4j import GraphDatabase


class Neo4jClient:
    def __init__(self, uri: str, user: str, password: str) -> None:
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self._driver.close()

    def clear_database(self) -> None:
        with self._driver.session() as session:
            session.run(
                """
                MATCH (n)
                CALL {
                    WITH n
                    DETACH DELETE n
                } IN TRANSACTIONS OF 10000 ROWS
                """
            )
        print("[Neo4j] Database cleared")

    def create_indexes(self) -> None:
        with self._driver.session() as session:
            session.run("CREATE INDEX doc_title IF NOT EXISTS FOR (d:Document) ON (d.title)")
            session.run("CREATE INDEX para_id IF NOT EXISTS FOR (p:Paragraph) ON (p.id)")
            session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)")
        print("[Neo4j] Indexes created")

    def create_vector_index(self, dimensions: int = 768) -> None:
        with self._driver.session() as session:
            session.run(
                f"""
                CREATE VECTOR INDEX paragraph_embeddings IF NOT EXISTS
                FOR (p:Paragraph) ON (p.embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {dimensions},
                    `vector.similarity_function`: 'cosine'
                }}}}
                """
            )
        print("[Neo4j] Paragraph vector index created")

    def create_entity_vector_index(self, dimensions: int = 768) -> None:
        with self._driver.session() as session:
            session.run(
                f"""
                CREATE VECTOR INDEX entity_description_embeddings IF NOT EXISTS
                FOR (e:Entity) ON (e.description_embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {dimensions},
                    `vector.similarity_function`: 'cosine'
                }}}}
                """
            )
        print("[Neo4j] Entity vector index created")

    def create_document(self, title: str) -> None:
        with self._driver.session() as session:
            session.run("MERGE (d:Document {title: $title})", title=title)

    def create_paragraph(
        self,
        paragraph_id: str,
        text: str,
        doc_title: str,
    ) -> None:
        with self._driver.session() as session:
            session.run(
                """
                MERGE (p:Paragraph {id: $id})
                ON CREATE SET p.text = $text, p.doc_title = $doc_title
                WITH p
                MATCH (d:Document {title: $doc_title})
                MERGE (d)-[:HAS_PARAGRAPH]->(p)
                """,
                id=paragraph_id,
                text=text,
                doc_title=doc_title,
            )

    def create_entity(
        self, name: str, entity_type: str | None = None, description: str | None = None
    ) -> None:
        with self._driver.session() as session:
            session.run(
                """
                MERGE (e:Entity {name: $name})
                ON CREATE SET e.type = $type, e.description = $description
                ON MATCH SET e.description = CASE 
                    WHEN e.description IS NULL THEN $description
                    WHEN $description IS NULL THEN e.description
                    WHEN e.description CONTAINS $description THEN e.description
                    ELSE e.description + ' | ' + $description
                END
                """,
                name=name,
                type=entity_type,
                description=description,
            )

    def create_mention(self, paragraph_id: str, entity_name: str) -> None:
        with self._driver.session() as session:
            session.run(
                """
                MATCH (p:Paragraph {id: $pid})
                MATCH (e:Entity {name: $ename})
                MERGE (p)-[:MENTIONS]->(e)
                """,
                pid=paragraph_id,
                ename=entity_name,
            )

    def create_mention_to_document(self, document_title: str, entity_name: str) -> None:
        with self._driver.session() as session:
            session.run(
                """
                MATCH (d:Document {title: $title})
                MATCH (e:Entity {name: $ename})
                MERGE (d)-[:MENTIONS]->(e)
                """,
                title=document_title,
                ename=entity_name,
            )

    def create_relation(self, subject: str, relation: str, obj: str) -> None:
        with self._driver.session() as session:
            session.run(
                """
                MATCH (s:Entity {name: $subject})
                MATCH (o:Entity {name: $object})
                MERGE (s)-[r:RELATES_TO]->(o)
                ON CREATE SET r.relation = $relation
                ON MATCH SET r.relation = CASE
                    WHEN r.relation IS NULL THEN $relation
                    WHEN $relation IS NULL THEN r.relation
                    WHEN r.relation CONTAINS $relation THEN r.relation
                    ELSE r.relation + ' | ' + $relation
                END
                """,
                subject=subject,
                object=obj,
                relation=relation,
            )

    def update_entity_doc_count(self) -> None:
        with self._driver.session() as session:
            session.run(
                """
                MATCH (e:Entity)
                OPTIONAL MATCH (p:Paragraph)-[:MENTIONS]->(e)
                WITH e, collect(DISTINCT p.doc_title) AS paragraph_doc_titles
                OPTIONAL MATCH (d:Document)-[:MENTIONS]->(e)
                WITH
                    e,
                    paragraph_doc_titles,
                    collect(DISTINCT d.title) AS document_titles
                WITH
                    e,
                    size([title IN paragraph_doc_titles WHERE title IS NOT NULL]) AS paragraph_count,
                    size(
                        [
                            title IN document_titles
                            WHERE title IS NOT NULL
                            AND NOT title IN paragraph_doc_titles
                        ]
                    ) AS extra_document_count
                WITH e, paragraph_count + extra_document_count AS doc_count
                SET e.doc_count = doc_count
                """
            )

    def get_entities_for_embedding(self) -> list[dict[str, str | None]]:
        with self._driver.session() as session:
            return session.run(
                """
                MATCH (e:Entity)
                RETURN e.name AS name, e.description AS description, e.description_hash AS description_hash
                """
            ).data()

    def set_entity_embedding(
        self,
        name: str,
        description_hash: str,
        embedding: list[float],
    ) -> None:
        with self._driver.session() as session:
            session.run(
                """
                MATCH (e:Entity {name: $name})
                SET e.description_hash = $description_hash, e.description_embedding = $embedding
                """,
                name=name,
                description_hash=description_hash,
                embedding=embedding,
            )

    def get_paragraphs_for_embedding(self) -> list[dict[str, str | None]]:
        with self._driver.session() as session:
            return session.run(
                """
                MATCH (p:Paragraph)
                RETURN p.id AS id, p.text AS text, p.doc_title AS doc_title,
                       p.text_hash AS text_hash
                """
            ).data()

    def set_paragraph_embedding(
        self,
        paragraph_id: str,
        text_hash: str,
        embedding: list[float],
    ) -> None:
        with self._driver.session() as session:
            session.run(
                """
                MATCH (p:Paragraph {id: $id})
                SET p.text_hash = $text_hash, p.embedding = $embedding
                """,
                id=paragraph_id,
                text_hash=text_hash,
                embedding=embedding,
            )

    def get_stats(self) -> dict[str, int]:
        with self._driver.session() as session:
            return {
                "documents": session.run(
                    "MATCH (d:Document) RETURN count(d) as c"
                ).single()["c"],
                "paragraphs": session.run(
                    "MATCH (p:Paragraph) RETURN count(p) as c"
                ).single()["c"],
                "entities": session.run(
                    "MATCH (e:Entity) RETURN count(e) as c"
                ).single()["c"],
                "has_paragraph": session.run(
                    "MATCH ()-[r:HAS_PARAGRAPH]->() RETURN count(r) as c"
                ).single()["c"],
                "mentions": session.run(
                    "MATCH ()-[r:MENTIONS]->() RETURN count(r) as c"
                ).single()["c"],
                "relations": session.run(
                    "MATCH ()-[r:RELATES_TO]->() RETURN count(r) as c"
                ).single()["c"],
            }

    def run_query(self, query: str, **params: Any) -> list[dict[str, Any]]:
        with self._driver.session() as session:
            return session.run(query, **params).data()
