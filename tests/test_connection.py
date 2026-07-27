# tests/test_connection.py
import os, sys
sys.path.append('..')
from config.settings import config
from src.llm_client import OnlineLLM, OnlineEmbedding

def main():
    if not config.API_KEY:
        print("✗ 没读到 API Key，先设置环境变量 DASHSCOPE_API_KEY")
        return

    llm = OnlineLLM(config.API_KEY, config.BASE_URL, config.LLM_MODEL)
    emb = OnlineEmbedding(config.API_KEY, config.BASE_URL, config.EMBED_MODEL)

    print("测试 LLM...")
    print("  ✓", llm.chat("回复'连接成功'"))

    print("测试 Embedding...")
    vec = emb.embed(["测试文本"])
    print(f"  ✓ 向量维度：{len(vec[0])}")

    print("\n✅ 在线 API 就绪，可以开始跑 GraphRAG")

if __name__ == "__main__":
    main()