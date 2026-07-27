from config.settings import config
from src.llm_client import OnlineLLM, OnlineEmbedding
from src.loader import load_document, chunk_text
from src.index import GraphRAGIndex
from src.neo4j_store import Neo4jStore
import os

def main():
    llm = OnlineLLM(config.API_KEY, config.BASE_URL, config.LLM_MODEL)
    emb = OnlineEmbedding(config.API_KEY, config.BASE_URL, config.EMBED_MODEL)

    store = Neo4jStore(
        config.NEO4J_URI,
        config.NEO4J_USER,
        config.NEO4J_PASSWORD,
        config.NEO4J_DATABASE,
    )
    try:
        store.verify_connection()
        store.create_constraints()
        index = GraphRAGIndex(llm, emb, graph_store=store)

        if store.has_complete_index():
            print("检测到完整的 Neo4j 索引，跳过文档解析和 LLM 抽取。")
            index.load()
        else:
            all_chunks = []
            chunk_sources = []
            for fname in os.listdir(config.DATA_DIR):
                path = os.path.join(config.DATA_DIR, fname)
                if fname.lower().endswith(('.pdf', '.txt')):
                    print(f"加载：{fname}")
                    text = load_document(path)
                    chunks = chunk_text(
                        text, config.CHUNK_SIZE, config.CHUNK_OVERLAP
                    )
                    all_chunks.extend(chunks)
                    chunk_sources.extend([fname] * len(chunks))

            print(f"共 {len(all_chunks)} 个文本块，开始建索引...\n")
            index.build(all_chunks, chunk_sources)

        index.inspect_graph()
        index.find_cross_doc_entities()

        print("\n===== 开始问答 =====")
        while True:
            q = input("\n请输入问题（输入 q 退出）：")
            if q.strip().lower() == 'q':
                break
            print("\n答：", index.search(q, config.TOP_K, config.HOPS))
    finally:
        store.close()

if __name__ == "__main__":
    main()
