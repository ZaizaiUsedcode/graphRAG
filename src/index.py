import networkx as nx
import numpy as np
from src.extractor import extract_graph
from src.prompts import ANSWER_PROMPT

class GraphRAGIndex:
    def __init__(self, llm, embedding, graph_store=None):
        self.llm = llm
        self.embedding = embedding
        self.graph_store = graph_store
        self.graph = nx.Graph()
        self.chunks = []
        self.chunk_embeddings = []
        self.entity_to_chunks = {}

    def build(self, chunks, chunk_sources=None):
        self.chunks = chunks
        self.chunk_embeddings = self.embedding.embed(chunks)
        chunk_sources = chunk_sources or ["unknown"] * len(chunks)
        if self.graph_store:
            self.graph_store.begin_build(
                len(chunks), self.embedding.model_name
            )

        for idx, chunk in enumerate(chunks):
            print(f"抽取中 {idx+1}/{len(chunks)}...")
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
                if not name:          # 连名字都没有的实体，跳过
                    continue
                desc = ent.get("description", "")   # 缺 description 就用空串
                etype = ent.get("type", "未知")

                if self.graph.has_node(name):
                    old = self.graph.nodes[name].get("description", "")
                    self.graph.nodes[name]["description"] = old + " | " + desc
                else:
                    self.graph.add_node(name, type=etype, description=desc)
                self.entity_to_chunks.setdefault(name, set()).add(idx)

            for rel in data.get("relationships", []):
                source = rel.get("source")
                target = rel.get("target")
                if not source or not target:   # 缺端点的关系，跳过
                    continue
                self.graph.add_edge(source, target,
                                    relation=rel.get("relation", "相关"),
                                    description=rel.get("description", ""))

        if self.graph_store:
            self.graph_store.complete_build()
        print(f"建图完成：{self.graph.number_of_nodes()} 实体，"
              f"{self.graph.number_of_edges()} 关系")

    def load(self):
        """Restore chunks, embeddings and the NetworkX graph from Neo4j."""
        if not self.graph_store:
            raise RuntimeError("未配置图存储，无法加载持久化索引")

        data = self.graph_store.load_index()
        chunk_rows = data["chunks"]
        expected_indices = list(range(len(chunk_rows)))
        actual_indices = [row["index"] for row in chunk_rows]
        if actual_indices != expected_indices:
            raise RuntimeError("Neo4j 中的 chunk 索引不连续，请重新构建索引")

        self.chunks = [row["text"] for row in chunk_rows]
        self.chunk_embeddings = [row["embedding"] for row in chunk_rows]
        self.graph.clear()
        self.entity_to_chunks.clear()

        for entity in data["entities"]:
            self.graph.add_node(
                entity["name"],
                type=entity["type"] or "未知",
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
                relation=relationship["relation"] or "相关",
                description=relationship["description"] or "",
            )

        print(
            f"已从 Neo4j 恢复索引：{len(self.chunks)} 个文本块，"
            f"{self.graph.number_of_nodes()} 个实体，"
            f"{self.graph.number_of_edges()} 条关系"
        )

    def search(self, query, top_k=3, hops=2, verbose=True):
        # 1. 向量检索入口块
        query_emb = self.embedding.embed([query])[0]
        sims = [np.dot(query_emb, ce) for ce in self.chunk_embeddings]
        top_chunk_ids = np.argsort(sims)[-top_k:][::-1]

        # 2. 入口块涉及的种子实体
        seed_entities = set()
        for cid in top_chunk_ids:
            for ent, chunk_ids in self.entity_to_chunks.items():
                if cid in chunk_ids:
                    seed_entities.add(ent)

        # 3. 图上多跳扩展
        related_info = []
        expanded_entities = set(seed_entities)
        expanded_chunks = set(int(c) for c in top_chunk_ids)
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
                        f"{entity} --[{edge.get('relation','相关')}]--> {neighbor}: "
                        f"{edge.get('description','')}")
                expanded_chunks.update(self.entity_to_chunks.get(neighbor, set()))

        # ===== 关键：打印召回过程 =====
        if verbose:
            print("\n" + "-"*50)
            print("【本次召回详情】")
            print(f"  命中入口块: {sorted(int(c) for c in top_chunk_ids)}")
            print(f"  种子实体（{len(seed_entities)}个）: {seed_entities}")
            print(f"  扩展后实体（{len(expanded_entities)}个）: {expanded_entities}")
            print(f"  召回关系（{len(set(related_info))}条）:")
            for r in sorted(set(related_info)):
                print(f"    {r}")
            print("-"*50 + "\n")

        # 4. 组装上下文 + 生成答案
        context_chunks = "\n---\n".join(self.chunks[cid] for cid in expanded_chunks)
        context_relations = "\n".join(set(related_info))
        prompt = ANSWER_PROMPT.format(
            context_chunks=context_chunks,
            context_relations=context_relations,
            query=query)
        return self.llm.chat(prompt, temperature=0.3)

    def inspect_graph(self):
        """打印图谱概览：所有实体、所有关系"""
        print("\n" + "="*50)
        print(f"【知识图谱概览】{self.graph.number_of_nodes()} 实体，"
              f"{self.graph.number_of_edges()} 关系")
        print("="*50)

        print("\n--- 实体列表 ---")
        for name, attr in self.graph.nodes(data=True):
            # 显示实体名、类型、出现在哪几个文档块
            chunks = sorted(self.entity_to_chunks.get(name, set()))
            print(f"  [{attr.get('type','未知')}] {name}  (出现于块 {chunks})")

        print("\n--- 关系列表 ---")
        for src, tgt, attr in self.graph.edges(data=True):
            print(f"  {src} --[{attr.get('relation','相关')}]--> {tgt}")

    def find_cross_doc_entities(self, file_chunk_ranges=None):
        """找出跨文档实体——出现在多个块、尤其跨文件的实体，这是跨文档推理的关键"""
        print("\n--- 跨块出现的实体（跨文档推理的桥梁）---")
        for name in self.graph.nodes():
            chunks = self.entity_to_chunks.get(name, set())
            if len(chunks) >= 2:   # 出现在2个及以上的块
                print(f"  {name}  →  出现于 {len(chunks)} 个块: {sorted(chunks)}")
