import json
import networkx as nx
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.extractor import extract_graph
from src.prompts import (
    ANSWER_PROMPT,
    GLOBAL_MAP_PROMPT,
    GLOBAL_REDUCE_PROMPT,
    QUERY_ROUTER_PROMPT,
)
from src.community import CommunityManager


class GraphRAGIndex:
    def __init__(self, llm, embedding, graph_store=None):
        self.llm = llm
        self.embedding = embedding
        self.graph_store = graph_store
        self.graph = nx.Graph()
        self.chunks = []
        self.chunk_embeddings = []
        self.entity_to_chunks = {}
        self.community_manager = None

    def build(self, chunks, chunk_sources=None):
        self.chunks = chunks
        self.chunk_embeddings = self.embedding.embed(chunks)
        chunk_sources = chunk_sources or ["unknown"] * len(chunks)
        if self.graph_store:
            self.graph_store.begin_build(
                len(chunks), self.embedding.model_name
            )

        for idx, chunk in enumerate(chunks):
            print(f"Extracting {idx+1}/{len(chunks)}...")
            data = extract_graph(chunk, self.llm)
            if self.graph_store:
                self.graph_store.save_extraction(
                    idx,
                    chunk,
                    chunk_sources[idx],
                    self.chunk_embeddings[idx],
                    data,
                )

            for ent in data.get("entities", []):
                name = ent.get("name")
                if not name:
                    continue
                desc = ent.get("description", "")
                etype = ent.get("type", "Unknown")

                if self.graph.has_node(name):
                    old = self.graph.nodes[name].get("description", "")
                    self.graph.nodes[name]["description"] = old + " | " + desc
                else:
                    self.graph.add_node(name, type=etype, description=desc)
                self.entity_to_chunks.setdefault(name, set()).add(idx)

            for rel in data.get("relationships", []):
                source = rel.get("source")
                target = rel.get("target")
                if not source or not target:
                    continue
                self.graph.add_edge(source, target,
                                    relation=rel.get("relation", "related_to"),
                                    description=rel.get("description", ""))

        if self.graph_store:
            self.graph_store.complete_build()
        print(f"Graph built: {self.graph.number_of_nodes()} entities, "
              f"{self.graph.number_of_edges()} relationships")

    def load(self):
        """Restore chunks, embeddings and the NetworkX graph from Neo4j."""
        if not self.graph_store:
            raise RuntimeError("Graph store not configured, cannot load persisted index")

        data = self.graph_store.load_index()
        chunk_rows = data["chunks"]
        expected_indices = list(range(len(chunk_rows)))
        actual_indices = [row["index"] for row in chunk_rows]
        if actual_indices != expected_indices:
            raise RuntimeError("Chunk indices in Neo4j are not contiguous, please rebuild the index")

        self.chunks = [row["text"] for row in chunk_rows]
        self.chunk_embeddings = [row["embedding"] for row in chunk_rows]
        self.graph.clear()
        self.entity_to_chunks.clear()

        for entity in data["entities"]:
            self.graph.add_node(
                entity["name"],
                type=entity["type"] or "Unknown",
                description=entity["description"] or "",
            )
        for mention in data["mentions"]:
            self.entity_to_chunks.setdefault(
                mention["entity"], set()
            ).add(mention["chunk_index"])
        for relationship in data["relationships"]:
            self.graph.add_edge(
                relationship["source"],
                relationship["target"],
                relation=relationship["relation"] or "related_to",
                description=relationship["description"] or "",
            )

        print(
            f"Index restored from Neo4j: {len(self.chunks)} chunks, "
            f"{self.graph.number_of_nodes()} entities, "
            f"{self.graph.number_of_edges()} relationships"
        )

    # ========== Unified Search Interface ==========

    def search(self, query, mode="local", top_k=3, hops=2, community_level=0, verbose=True):
        """
        Unified search interface supporting multiple modes.

        Args:
            query: User query string
            mode: 'local', 'global', or 'auto'
            top_k: Number of top chunks for local search
            hops: Graph expansion hops for local search
            community_level: Community level for global search
            verbose: Print detailed retrieval info

        Returns:
            Generated answer string
        """
        if mode == "auto":
            mode = self._route_query(query)
            if verbose:
                print(f"[Auto] Routed to {mode} mode")

        if mode == "global":
            return self._global_search(query, level=community_level, verbose=verbose)
        else:
            return self._local_search(query, top_k=top_k, hops=hops, verbose=verbose)

    # ========== Local Search ==========

    def _local_search(self, query, top_k=3, hops=2, verbose=True):
        """
        Local search: vector retrieval -> seed entities -> graph expansion -> answer.
        """
        # 1. Vector retrieval for entry chunks
        top_chunk_ids = self._vector_retrieve(query, top_k)

        # 2. Get seed entities from entry chunks
        seed_entities = self._get_seed_entities(top_chunk_ids)

        # 3. Graph expansion
        expanded_entities, expanded_chunks, related_info = self._expand_graph_context(
            seed_entities, hops, top_chunk_ids
        )

        if verbose:
            print("\n" + "-"*50)
            print("[Local Search Retrieval Details]")
            print(f"  Entry chunks: {sorted(int(c) for c in top_chunk_ids)}")
            print(f"  Seed entities ({len(seed_entities)}): {seed_entities}")
            print(f"  Expanded entities ({len(expanded_entities)}): {expanded_entities}")
            print(f"  Retrieved relationships ({len(set(related_info))}):")
            for r in sorted(set(related_info)):
                print(f"    {r}")
            print("-"*50 + "\n")

        # 4. Assemble context and generate answer
        context_chunks = "\n---\n".join(self.chunks[cid] for cid in expanded_chunks)
        context_relations = "\n".join(set(related_info))
        prompt = ANSWER_PROMPT.format(
            context_chunks=context_chunks,
            context_relations=context_relations,
            query=query)
        return self.llm.chat(prompt, temperature=0.3)

    def _vector_retrieve(self, query, top_k):
        """Retrieve top-k chunks by vector similarity."""
        query_emb = self.embedding.embed([query])[0]
        sims = [np.dot(query_emb, ce) for ce in self.chunk_embeddings]
        return np.argsort(sims)[-top_k:][::-1]

    def _get_seed_entities(self, chunk_ids):
        """Get entities mentioned in the given chunks."""
        seed_entities = set()
        for cid in chunk_ids:
            for ent, entity_chunks in self.entity_to_chunks.items():
                if cid in entity_chunks:
                    seed_entities.add(ent)
        return seed_entities

    def _expand_graph_context(self, seed_entities, hops, initial_chunks):
        """Expand from seed entities via graph traversal."""
        related_info = []
        expanded_entities = set(seed_entities)
        expanded_chunks = set(int(c) for c in initial_chunks)

        for entity in seed_entities:
            if not self.graph.has_node(entity):
                continue
            neighbors = nx.single_source_shortest_path_length(
                self.graph, entity, cutoff=hops)
            for neighbor in neighbors:
                expanded_entities.add(neighbor)
                if self.graph.has_edge(entity, neighbor):
                    edge = self.graph.edges[entity, neighbor]
                    related_info.append(
                        f"{entity} --[{edge.get('relation','related_to')}]--> {neighbor}: "
                        f"{edge.get('description','')}")
                expanded_chunks.update(self.entity_to_chunks.get(neighbor, set()))

        return expanded_entities, expanded_chunks, related_info

    # ========== Global Search ==========

    def _global_search(self, query, level=0, verbose=True):
        """
        Global search: community map -> reduce -> answer.
        """
        if not self.community_manager or not self.community_manager.communities:
            if verbose:
                print("No communities detected, attempting to load...")
            self.load_communities()

        if not self.community_manager or not self.community_manager.communities:
            if verbose:
                print("No community data available, falling back to local search")
            return self._local_search(query, verbose=verbose)

        communities = self.community_manager.get_communities_by_level(level)
        if not communities:
            if verbose:
                print(f"No communities at level {level}, using all communities")
            communities = list(self.community_manager.communities.values())

        if verbose:
            print(f"\n[Global Search] Using {len(communities)} communities")

        # Map phase: evaluate each community
        partial_answers = self._map_communities(query, communities, verbose)

        if not partial_answers:
            if verbose:
                print("No relevant communities found, falling back to local search")
            return self._local_search(query, verbose=verbose)

        # Reduce phase: synthesize final answer
        return self._reduce_answers(query, partial_answers, verbose)

    def _map_communities(self, query, communities, verbose=True, min_relevance=3, max_communities=10):
        """
        Map phase: Process each community and generate partial answers.
        """
        results = []

        for idx, community in enumerate(communities):
            if verbose:
                print(f"  Map [{idx + 1}/{len(communities)}] Community {community.id}...")

            entities_str, relationships_str = self.community_manager._get_community_context(community)
            prompt = GLOBAL_MAP_PROMPT.format(
                query=query,
                community_summary=community.summary or "(No summary)",
                entities=entities_str or "(No entities)",
                relationships=relationships_str or "(No relationships)",
            )

            try:
                response = self.llm.chat(prompt, temperature=0.2)
                # Parse JSON response
                response = response.strip()
                if response.startswith("```"):
                    response = response.split("```")[1]
                    if response.startswith("json"):
                        response = response[4:]
                data = json.loads(response)
                relevance = int(data.get("relevance", 0))
                answer = data.get("answer", "")

                if relevance >= min_relevance and answer and answer.lower() != "no relevant information":
                    results.append({
                        "community_id": community.id,
                        "relevance": relevance,
                        "answer": answer,
                    })
                    if verbose:
                        print(f"    Relevance: {relevance}, kept")
                elif verbose:
                    print(f"    Relevance: {relevance}, filtered")
            except Exception as e:
                if verbose:
                    print(f"    Processing failed: {e}")

        # Sort by relevance and take top N
        results.sort(key=lambda x: x["relevance"], reverse=True)
        results = results[:max_communities]

        if verbose:
            print(f"  Map phase complete: {len(results)} relevant communities")

        return results

    def _reduce_answers(self, query, partial_answers, verbose=True):
        """
        Reduce phase: Synthesize partial answers into final answer.
        """
        if verbose:
            print("  Reduce phase: Synthesizing final answer...")

        answers_text = "\n\n".join([
            f"[Community {pa['community_id']} (Relevance: {pa['relevance']})]\n{pa['answer']}"
            for pa in partial_answers
        ])

        prompt = GLOBAL_REDUCE_PROMPT.format(
            query=query,
            partial_answers=answers_text,
        )

        return self.llm.chat(prompt, temperature=0.3)

    # ========== Query Router ==========

    def _route_query(self, query):
        """
        Auto-route query to local or global mode.
        """
        prompt = QUERY_ROUTER_PROMPT.format(query=query)
        try:
            response = self.llm.chat(prompt, temperature=0.1).strip().lower()
            if "global" in response:
                return "global"
        except Exception:
            pass
        return "local"

    # ========== Community Management ==========

    def build_communities(self, resolution=1.0, max_levels=3, min_size=2,
                          generate_summaries=True, save=True):
        """
        Detect communities and optionally generate summaries.

        Args:
            resolution: Louvain resolution parameter
            max_levels: Maximum hierarchy levels
            min_size: Minimum community size
            generate_summaries: Whether to generate LLM summaries
            save: Whether to save to Neo4j
        """
        self.community_manager = CommunityManager(
            graph=self.graph,
            llm=self.llm,
            graph_store=self.graph_store,
        )

        self.community_manager.detect_communities(
            resolution=resolution,
            max_levels=max_levels,
            min_size=min_size,
        )

        if generate_summaries:
            self.community_manager.generate_all_summaries()

        if save and self.graph_store:
            self.community_manager.save_to_neo4j()

        return self.community_manager.communities

    def load_communities(self):
        """Load communities from Neo4j."""
        if not self.graph_store:
            return {}

        if not self.graph_store.has_communities():
            print("No community data in Neo4j")
            return {}

        self.community_manager = CommunityManager(
            graph=self.graph,
            llm=self.llm,
            graph_store=self.graph_store,
        )
        return self.community_manager.load_from_neo4j()

    # ========== Inspection Methods ==========

    def inspect_graph(self):
        """Print graph overview: all entities and relationships."""
        print("\n" + "="*50)
        print(f"[Knowledge Graph Overview] {self.graph.number_of_nodes()} entities, "
              f"{self.graph.number_of_edges()} relationships")
        print("="*50)

        print("\n--- Entity List ---")
        for name, attr in self.graph.nodes(data=True):
            chunks = sorted(self.entity_to_chunks.get(name, set()))
            print(f"  [{attr.get('type','Unknown')}] {name}  (in chunks {chunks})")

        print("\n--- Relationship List ---")
        for src, tgt, attr in self.graph.edges(data=True):
            print(f"  {src} --[{attr.get('relation','related_to')}]--> {tgt}")

    def find_cross_doc_entities(self, file_chunk_ranges=None):
        """Find cross-document entities - entities appearing in multiple chunks."""
        print("\n--- Cross-chunk Entities (Cross-document Reasoning Bridges) ---")
        for name in self.graph.nodes():
            chunks = self.entity_to_chunks.get(name, set())
            if len(chunks) >= 2:
                print(f"  {name}  ->  appears in {len(chunks)} chunks: {sorted(chunks)}")

    def inspect_communities(self):
        """Print community overview."""
        if not self.community_manager or not self.community_manager.communities:
            print("No community data")
            return

        print("\n" + "="*50)
        print(f"[Community Overview] Total {len(self.community_manager.communities)} communities")
        print("="*50)

        by_level = {}
        for c in self.community_manager.communities.values():
            by_level.setdefault(c.level, []).append(c)

        for level in sorted(by_level.keys()):
            communities = by_level[level]
            print(f"\n--- Level {level} ({len(communities)} communities) ---")
            for c in communities:
                summary_preview = (c.summary[:50] + "...") if c.summary and len(c.summary) > 50 else (c.summary or "No summary")
                print(f"  [{c.id}] {c.entity_count} entities, {c.relationship_count} relationships")
                print(f"       Entities: {', '.join(list(c.entities)[:5])}{'...' if len(c.entities) > 5 else ''}")
                print(f"       Summary: {summary_preview}")
