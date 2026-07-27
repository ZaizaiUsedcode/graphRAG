config/settings.py --配置集中管理
src/llm_client.py -- 模型接口
../extractor.py -- 实体和关系抽取
../index.py -- 创建图像，搜索，成像功能
../loader.py --文本转换和切块
../prompts.py -- 提示词集中管理
main.py -- 主入口

## Neo4j

启动数据库：

```bash
docker compose up -d
uv sync
uv run python -m tests.test_neo4j_connection
```

Neo4j Browser: http://localhost:7474

- 用户名：`neo4j`
- 本地默认密码：`neo4j-local-password`

程序运行 `uv run python main.py` 后会写入：

- `(:Document)`：源文件
- `(:Chunk)-[:PART_OF]->(:Document)`：文本块及所属文档
- `(:Chunk).embedding`：文本块的向量
- `(:Chunk)-[:MENTIONS]->(:Entity)`：文本块提及的实体
- `(:Entity)-[:执行|约束|包含|引用...]->(:Entity)`：具体业务关系
- 业务关系的 `description`：关系说明
- 业务关系的 `chunk_ids`：支撑该关系的证据文本块 ID
- `(:IndexMetadata)`：索引构建状态、chunk 数量和 embedding 模型

首次运行会读取文档、切块、计算 embedding 并调用 LLM 抽取。索引完整写入
Neo4j 后，再次运行会直接恢复 chunk、embedding、实体关系和 NetworkX 内存图，
不会重复执行上述构建流程。

可在 Browser 中执行：

```cypher
MATCH (a:Entity)-[r]->(b:Entity)
RETURN a, r, b
LIMIT 100
```
