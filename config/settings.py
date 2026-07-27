import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    API_KEY = os.getenv("DASHSCOPE_API_KEY")
    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    LLM_MODEL = "qwen-plus"
    EMBED_MODEL = "text-embedding-v4"

    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    TOP_K = 3
    HOPS = 2

    DATA_DIR = "./data/raw"
    OUTPUT_DIR = "./data/output"

    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j-local-password")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

config = Config()
