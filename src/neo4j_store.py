import hashlib

from neo4j import GraphDatabase


class Neo4jStore:
    """Persist extracted chunks, entities and relationships in Neo4j."""

    def __init__(self, uri, user, password, database="neo4j"):
        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def verify_connection(self):
        self.driver.verify_connectivity()

    def close(self):
        self.driver.close()

    def create_constraints(self):
        statements = (
            "CREATE CONSTRAINT entity_name IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS "
            "FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT document_name IF NOT EXISTS "
            "FOR (d:Document) REQUIRE d.name IS UNIQUE",
        )
        with self.driver.session(database=self.database) as session:
            for statement in statements:
                session.run(statement).consume()

    @staticmethod
    def chunk_id(source, chunk_index, text):
        value = f"{source}\0{chunk_index}\0{text}".encode("utf-8")
        return hashlib.sha256(value).hexdigest()

    @staticmethod
    def _relationship_type(value):
        """Escape a business relation name for use as a Cypher type."""
        value = (value or "相关").strip() or "相关"
        return value.replace("`", "``")

    def begin_build(self, expected_chunks, embedding_model):
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MERGE (m:IndexMetadata {name: 'default'})
                SET m.status = 'building',
                    m.expected_chunks = $expected_chunks,
                    m.embedding_model = $embedding_model,
                    m.updated_at = datetime()
                """,
                expected_chunks=expected_chunks,
                embedding_model=embedding_model,
            ).consume()

    def complete_build(self):
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MATCH (m:IndexMetadata {name: 'default'})
                SET m.status = 'complete', m.updated_at = datetime()
                """
            ).consume()

    def has_complete_index(self):
        with self.driver.session(database=self.database) as session:
            record = session.run(
                """
                MATCH (m:IndexMetadata {name: 'default'})
                OPTIONAL MATCH (c:Chunk)
                WITH m, count(c) AS chunk_count,
                     count(c.embedding) AS embedding_count
                RETURN m.status = 'complete'
                    AND m.expected_chunks > 0
                    AND chunk_count = m.expected_chunks
                    AND embedding_count = m.expected_chunks AS complete
                """
            ).single()
        return bool(record and record["complete"])

    def save_extraction(self, chunk_index, text, source, embedding, data):
        chunk_id = self.chunk_id(source, chunk_index, text)
        embedding = [float(value) for value in embedding]
        entities = [
            {
                "name": entity.get("name"),
                "type": entity.get("type", "未知"),
                "description": entity.get("description", ""),
            }
            for entity in data.get("entities", [])
            if entity.get("name")
        ]
        relationships = [
            {
                "source": rel.get("source"),
                "target": rel.get("target"),
                "type": rel.get("relation", "相关"),
                "description": rel.get("description", ""),
            }
            for rel in data.get("relationships", [])
            if rel.get("source") and rel.get("target")
        ]

        query = """
        MERGE (d:Document {name: $source})
        MERGE (c:Chunk {id: $chunk_id})
        SET c.text = $text,
            c.index = $chunk_index,
            c.source = $source,
            c.embedding = $embedding
        MERGE (c)-[:PART_OF]->(d)
        WITH c
        CALL (c) {
          UNWIND $entities AS item
          MERGE (e:Entity {name: item.name})
          SET e.type = item.type,
              e.description = CASE
                WHEN item.description = '' THEN coalesce(e.description, '')
                ELSE item.description
              END
          MERGE (c)-[:MENTIONS]->(e)
        }
        RETURN c.id AS chunk_id
        """
        with self.driver.session(database=self.database) as session:
            def persist(tx):
                tx.run(
                    query,
                    chunk_id=chunk_id,
                    chunk_index=chunk_index,
                    text=text,
                    source=source,
                    embedding=embedding,
                    entities=entities,
                ).consume()
                for item in relationships:
                    rel_type = self._relationship_type(item["type"])
                    tx.run(
                        f"""
                        MATCH (c:Chunk {{id: $chunk_id}})
                        MERGE (source:Entity {{name: $source}})
                        MERGE (target:Entity {{name: $target}})
                        MERGE (source)-[r:`{rel_type}` {{
                          description: $description
                        }}]->(target)
                        SET r.business_type = $business_type,
                            r.updated_at = datetime(),
                            r.chunk_ids = CASE
                              WHEN c.id IN coalesce(r.chunk_ids, [])
                                THEN r.chunk_ids
                              ELSE coalesce(r.chunk_ids, []) + c.id
                            END
                        """,
                        chunk_id=chunk_id,
                        source=item["source"],
                        target=item["target"],
                        description=item["description"],
                        business_type=item["type"],
                    ).consume()

            session.execute_write(persist)
        return chunk_id

    def migrate_related_relationships(self):
        """Replace legacy :RELATED edges with their concrete business types."""
        with self.driver.session(database=self.database) as session:
            rows = [
                dict(record)
                for record in session.run(
                    """
                    MATCH (source:Entity)-[r:RELATED]->(target:Entity)
                    RETURN elementId(r) AS relationship_id,
                           source.name AS source,
                           target.name AS target,
                           r.type AS business_type,
                           r.description AS description,
                           r.chunk_ids AS chunk_ids,
                           r.updated_at AS updated_at
                    """
                )
            ]

            def migrate(tx, item):
                rel_type = self._relationship_type(item["business_type"])
                tx.run(
                    f"""
                    MATCH ()-[old]->()
                    WHERE elementId(old) = $relationship_id
                    MATCH (source:Entity {{name: $source}})
                    MATCH (target:Entity {{name: $target}})
                    MERGE (source)-[r:`{rel_type}` {{
                      description: $description
                    }}]->(target)
                    SET r.business_type = $business_type,
                        r.chunk_ids = $chunk_ids,
                        r.updated_at = coalesce($updated_at, datetime())
                    DELETE old
                    """,
                    **item,
                ).consume()

            for row in rows:
                session.execute_write(migrate, row)
        return len(rows)

    def load_index(self):
        """Load the persisted retrieval index and graph from Neo4j."""
        with self.driver.session(database=self.database) as session:
            chunks = [
                dict(record)
                for record in session.run(
                    """
                    MATCH (c:Chunk)
                    RETURN c.id AS id, c.index AS index, c.text AS text,
                           c.source AS source, c.embedding AS embedding
                    ORDER BY c.index
                    """
                )
            ]
            entities = [
                dict(record)
                for record in session.run(
                    """
                    MATCH (e:Entity)
                    RETURN e.name AS name, e.type AS type,
                           e.description AS description
                    """
                )
            ]
            mentions = [
                dict(record)
                for record in session.run(
                    """
                    MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                    RETURN c.index AS chunk_index, e.name AS entity
                    """
                )
            ]
            relationships = [
                dict(record)
                for record in session.run(
                    """
                    MATCH (source:Entity)-[r]->(target:Entity)
                    RETURN source.name AS source, target.name AS target,
                           coalesce(r.business_type, r.type, type(r)) AS relation,
                           r.description AS description
                    """
                )
            ]
        return {
            "chunks": chunks,
            "entities": entities,
            "mentions": mentions,
            "relationships": relationships,
        }
