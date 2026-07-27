import re
import json
from src.prompts import EXTRACTION_PROMPT

def extract_graph(chunk, llm):
    prompt = EXTRACTION_PROMPT.format(chunk=chunk)
    response = llm.chat(prompt)
    try:
        json_str = re.search(r'\{.*\}', response, re.DOTALL).group()
        return json.loads(json_str)
    except (json.JSONDecodeError, AttributeError):
        return {"entities": [], "relationships": []}