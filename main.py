from config.settings import config
from src.llm_client import OnlineLLM, OnlineEmbedding
from src.loader import load_document, chunk_text
from src.index import GraphRAGIndex
from src.neo4j_store import Neo4jStore
import os


def print_help():
    """Print available commands."""
    print("\nAvailable commands:")
    print("  q or quit    - Exit the program")
    print("  mode local   - Switch to Local Search mode")
    print("  mode global  - Switch to Global Search mode")
    print("  mode auto    - Switch to Auto mode (automatic routing)")
    print("  build        - Build communities (for Global Search)")
    print("  communities  - View community overview")
    print("  graph        - View knowledge graph overview")
    print("  help         - Show this help message")
    print("  Other input  - Search as a question")


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
            print("Complete Neo4j index detected, skipping document parsing and LLM extraction.")
            index.load()
        else:
            all_chunks = []
            chunk_sources = []
            for fname in os.listdir(config.DATA_DIR):
                path = os.path.join(config.DATA_DIR, fname)
                if fname.lower().endswith(('.pdf', '.txt')):
                    print(f"Loading: {fname}")
                    text = load_document(path)
                    chunks = chunk_text(
                        text, config.CHUNK_SIZE, config.CHUNK_OVERLAP
                    )
                    all_chunks.extend(chunks)
                    chunk_sources.extend([fname] * len(chunks))

            print(f"Total {len(all_chunks)} chunks, starting index build...\n")
            index.build(all_chunks, chunk_sources)

        # Try to load existing communities
        if store.has_communities():
            print("Community data detected, loading...")
            index.load_communities()
            index.inspect_communities()
        else:
            print("No community data detected. Use 'build' command to build communities for Global Search.")

        index.inspect_graph()
        index.find_cross_doc_entities()

        # Default search mode
        search_mode = "local"

        print("\n===== GraphRAG Q&A System =====")
        print(f"Current search mode: {search_mode}")
        print_help()

        while True:
            q = input(f"\n[{search_mode}] Enter question or command: ").strip()

            if not q:
                continue

            if q.lower() in ('q', 'quit'):
                break

            if q.lower() == 'help':
                print_help()
                continue

            if q.lower().startswith('mode '):
                new_mode = q.lower().split(' ', 1)[1].strip()
                if new_mode in ('local', 'global', 'auto'):
                    search_mode = new_mode
                    print(f"Search mode switched to: {search_mode}")
                else:
                    print("Invalid mode. Options: local, global, auto")
                continue

            if q.lower() == 'build':
                print("\nBuilding communities...")
                index.build_communities(
                    resolution=config.COMMUNITY_RESOLUTION,
                    max_levels=config.COMMUNITY_MAX_LEVELS,
                    min_size=config.COMMUNITY_MIN_SIZE,
                    generate_summaries=True,
                    save=True,
                )
                index.inspect_communities()
                continue

            if q.lower() == 'communities':
                index.inspect_communities()
                continue

            if q.lower() == 'graph':
                index.inspect_graph()
                index.find_cross_doc_entities()
                continue

            # Perform search
            print(f"\n[Searching with {search_mode} mode...]")
            answer = index.search(
                q,
                mode=search_mode,
                top_k=config.TOP_K,
                hops=config.HOPS,
                verbose=True,
            )
            print("\nAnswer:", answer)

    finally:
        store.close()


if __name__ == "__main__":
    main()
