"""Community detection and management for GraphRAG Global Search."""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import networkx as nx

from src.prompts import COMMUNITY_SUMMARY_PROMPT


@dataclass
class Community:
    """A detected community in the knowledge graph."""

    id: str
    level: int
    entities: Set[str] = field(default_factory=set)
    summary: Optional[str] = None
    entity_count: int = 0
    relationship_count: int = 0

    def __post_init__(self):
        if not self.entity_count:
            self.entity_count = len(self.entities)


class CommunityManager:
    """Manages community detection and summarization for GraphRAG."""

    def __init__(self, graph: nx.Graph, llm=None, graph_store=None):
        """
        Args:
            graph: NetworkX graph containing entities and relationships
            llm: LLM client for generating summaries
            graph_store: Neo4jStore for persistence
        """
        self.graph = graph
        self.llm = llm
        self.graph_store = graph_store
        self.communities: Dict[str, Community] = {}

    def detect_communities(self, resolution=1.0, max_levels=3, min_size=2):
        """
        Detect communities using the Louvain algorithm.

        Args:
            resolution: Louvain resolution parameter (higher = smaller communities)
            max_levels: Maximum hierarchy levels to detect
            min_size: Minimum community size

        Returns:
            Dict mapping community_id to Community objects
        """
        if self.graph.number_of_nodes() == 0:
            print("No nodes in graph, cannot detect communities")
            return {}

        self.communities.clear()

        for level in range(max_levels):
            current_resolution = resolution * (2 ** level)
            try:
                partition = nx.community.louvain_communities(
                    self.graph,
                    resolution=current_resolution,
                    seed=42,
                )
            except Exception as e:
                print(f"Community detection failed (level={level}): {e}")
                break

            level_communities = 0
            for idx, members in enumerate(partition):
                if len(members) < min_size:
                    continue
                community_id = f"L{level}_C{idx}"
                community = Community(
                    id=community_id,
                    level=level,
                    entities=set(members),
                    entity_count=len(members),
                )
                subgraph = self.graph.subgraph(members)
                community.relationship_count = subgraph.number_of_edges()
                self.communities[community_id] = community
                level_communities += 1

            print(f"Level {level} (resolution={current_resolution:.2f}): "
                  f"detected {level_communities} communities")

            if level_communities <= 1:
                break

        total = len(self.communities)
        print(f"Community detection complete: {total} communities total")
        return self.communities

    def _get_community_context(self, community: Community):
        """Get entities and relationships info for a community."""
        entity_infos = []
        for entity in community.entities:
            if self.graph.has_node(entity):
                node_data = self.graph.nodes[entity]
                etype = node_data.get("type", "Unknown")
                desc = node_data.get("description", "")
                entity_infos.append(f"- {entity} ({etype}): {desc}")

        relationship_infos = []
        subgraph = self.graph.subgraph(community.entities)
        for src, tgt, data in subgraph.edges(data=True):
            rel = data.get("relation", "related_to")
            desc = data.get("description", "")
            relationship_infos.append(f"- {src} --[{rel}]--> {tgt}: {desc}")

        return "\n".join(entity_infos), "\n".join(relationship_infos)

    def generate_community_summary(self, community: Community):
        """Generate a summary for a single community using LLM."""
        if not self.llm:
            raise RuntimeError("LLM not configured, cannot generate community summary")

        entities_str, relationships_str = self._get_community_context(community)
        prompt = COMMUNITY_SUMMARY_PROMPT.format(
            community_id=community.id,
            level=community.level,
            entities=entities_str or "(No entity information)",
            relationships=relationships_str or "(No relationship information)",
        )

        try:
            summary = self.llm.chat(prompt, temperature=0.3)
            community.summary = summary.strip()
        except Exception as e:
            print(f"Failed to generate summary for community {community.id}: {e}")
            community.summary = f"Community contains {community.entity_count} entities"

        return community.summary

    def generate_all_summaries(self, level=None, verbose=True):
        """
        Generate summaries for all communities at a given level.

        Args:
            level: Only generate for this level, or all levels if None
            verbose: Print progress
        """
        target_communities = [
            c for c in self.communities.values()
            if level is None or c.level == level
        ]

        total = len(target_communities)
        if verbose:
            print(f"Generating summaries for {total} communities...")

        for idx, community in enumerate(target_communities):
            if verbose:
                print(f"  [{idx + 1}/{total}] Generating summary for community {community.id}...")
            self.generate_community_summary(community)

        if verbose:
            print("Community summary generation complete")

    def save_to_neo4j(self):
        """Persist all communities to Neo4j."""
        if not self.graph_store:
            raise RuntimeError("Graph store not configured, cannot save communities")

        communities_data = [
            {
                "id": c.id,
                "level": c.level,
                "entities": list(c.entities),
                "summary": c.summary,
                "entity_count": c.entity_count,
                "relationship_count": c.relationship_count,
            }
            for c in self.communities.values()
        ]
        self.graph_store.save_communities_batch(communities_data)
        print(f"Saved {len(communities_data)} communities to Neo4j")

    def load_from_neo4j(self):
        """Load communities from Neo4j."""
        if not self.graph_store:
            raise RuntimeError("Graph store not configured, cannot load communities")

        communities_data = self.graph_store.load_communities()
        self.communities.clear()

        for data in communities_data:
            community = Community(
                id=data["id"],
                level=data["level"],
                entities=data["entities"],
                summary=data["summary"],
                entity_count=data["entity_count"] or len(data["entities"]),
                relationship_count=data["relationship_count"] or 0,
            )
            self.communities[community.id] = community

        print(f"Loaded {len(self.communities)} communities from Neo4j")
        return self.communities

    def get_communities_by_level(self, level: int) -> List[Community]:
        """Get all communities at a specific level."""
        return [c for c in self.communities.values() if c.level == level]

    def get_entity_community(self, entity_name: str, level: int = 0) -> Optional[Community]:
        """Find which community an entity belongs to at a given level."""
        for community in self.communities.values():
            if community.level == level and entity_name in community.entities:
                return community
        return None
