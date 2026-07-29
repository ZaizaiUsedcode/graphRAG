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

    # 社区检测配置
    COMMUNITY_RESOLUTION = 1.0
    COMMUNITY_MAX_LEVELS = 3
    COMMUNITY_MIN_SIZE = 2

    # Global Search配置
    GLOBAL_MIN_RELEVANCE = 3
    GLOBAL_MAX_COMMUNITIES = 10

config = Config()
