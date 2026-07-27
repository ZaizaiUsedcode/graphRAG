from config.settings import config
from src.neo4j_store import Neo4jStore


def main():
    store = Neo4jStore(
        config.NEO4J_URI,
        config.NEO4J_USER,
        config.NEO4J_PASSWORD,
        config.NEO4J_DATABASE,
    )
    try:
        store.verify_connection()
        store.create_constraints()
        chunk_id = store.save_extraction(
            0,
            "机构应当报告可疑交易。",
            "__neo4j_smoke_test__",
            [0.1, 0.2, 0.3],
            {
                "entities": [
                    {"name": "__机构__", "type": "部门/角色", "description": ""},
                    {"name": "__可疑交易__", "type": "业务概念", "description": ""},
                ],
                "relationships": [
                    {
                        "source": "__机构__",
                        "target": "__可疑交易__",
                        "relation": "执行",
                        "description": "测试关系",
                    }
                ],
            },
        )
        with store.driver.session(database=config.NEO4J_DATABASE) as session:
            record = session.run(
                """
                MATCH (c:Chunk {id: $chunk_id})-[:MENTIONS]->(:Entity)
                RETURN count(*) AS count, c.embedding AS embedding
                """,
                chunk_id=chunk_id,
            ).single()
            assert record["count"] == 2
            assert record["embedding"] == [0.1, 0.2, 0.3]

            relationship = session.run(
                """
                MATCH (:Entity {name: '__机构__'})
                      -[r:执行]->
                      (:Entity {name: '__可疑交易__'})
                RETURN r.business_type AS business_type,
                       r.description AS description
                """
            ).single()
            assert relationship["business_type"] == "执行"
            assert relationship["description"] == "测试关系"

            session.run(
                """
                MATCH (c:Chunk {id: $chunk_id}) DETACH DELETE c
                WITH 1 AS ignored
                MATCH (d:Document {name: '__neo4j_smoke_test__'}) DETACH DELETE d
                WITH ignored
                MATCH (e:Entity)
                WHERE e.name IN ['__机构__', '__可疑交易__']
                DETACH DELETE e
                """,
                chunk_id=chunk_id,
            ).consume()
        print("✓ Neo4j 连接、约束及图数据写入均正常")
    finally:
        store.close()


if __name__ == "__main__":
    main()
