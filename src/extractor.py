"""Entity and relationship extraction from text chunks."""

import re
from src.prompts import EXTRACTION_PROMPT


def parse_extraction_response(response: str) -> dict:
    """
    Parse the GraphRAG-style extraction response.

    Expected format:
    ("entity"<|><name><|><type><|><description>)
    ##
    ("relationship"<|><source><|><target><|><description><|><type><|><strength>)
    ...
    <|COMPLETE|>
    """
    entities = []
    relationships = []

    # Remove the completion marker
    response = response.replace("<|COMPLETE|>", "")

    # Split by ## delimiter
    items = re.split(r'\s*##\s*', response.strip())

    for item in items:
        item = item.strip()
        if not item:
            continue

        # Parse entity: ("entity"<|><name><|><type><|><description>)
        entity_match = re.match(
            r'\("entity"<\|>(.+?)<\|>(.+?)<\|>(.+?)\)',
            item,
            re.DOTALL
        )
        if entity_match:
            name, etype, description = entity_match.groups()
            entities.append({
                "name": name.strip(),
                "type": etype.strip(),
                "description": description.strip(),
            })
            continue

        # Parse relationship: ("relationship"<|><source><|><target><|><description><|><type><|><strength>)
        rel_match = re.match(
            r'\("relationship"<\|>(.+?)<\|>(.+?)<\|>(.+?)<\|>(.+?)<\|>(\d+)\)',
            item,
            re.DOTALL
        )
        if rel_match:
            source, target, description, rel_type, strength = rel_match.groups()
            relationships.append({
                "source": source.strip(),
                "target": target.strip(),
                "relation": rel_type.strip(),
                "description": description.strip(),
                "strength": int(strength),
            })
            continue

        # Try alternative format without strength (some models might omit it)
        rel_match_alt = re.match(
            r'\("relationship"<\|>(.+?)<\|>(.+?)<\|>(.+?)<\|>(.+?)\)',
            item,
            re.DOTALL
        )
        if rel_match_alt:
            source, target, description, rel_type = rel_match_alt.groups()
            relationships.append({
                "source": source.strip(),
                "target": target.strip(),
                "relation": rel_type.strip(),
                "description": description.strip(),
                "strength": 5,  # default strength
            })

    return {"entities": entities, "relationships": relationships}


def extract_graph(chunk: str, llm) -> dict:
    """
    Extract entities and relationships from a text chunk.

    Args:
        chunk: Text chunk to extract from
        llm: LLM client with chat() method

    Returns:
        Dict with 'entities' and 'relationships' lists
    """
    prompt = EXTRACTION_PROMPT.format(chunk=chunk)
    response = llm.chat(prompt)

    try:
        result = parse_extraction_response(response)

        # Filter out invalid entries
        result["entities"] = [
            e for e in result["entities"]
            if e.get("name") and len(e["name"]) > 0
        ]
        result["relationships"] = [
            r for r in result["relationships"]
            if r.get("source") and r.get("target")
            and r["source"] != r["target"]  # no self-loops
        ]

        return result

    except Exception as e:
        print(f"Extraction parsing failed: {e}")
        return {"entities": [], "relationships": []}
