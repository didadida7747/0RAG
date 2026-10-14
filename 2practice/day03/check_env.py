from pymilvus import MilvusClient
from langchain_ollama import OllamaEmbeddings

# 1. 检查 Milvus
client = MilvusClient(uri="http://localhost:19530")
print("Milvus 连接成功，现有集合：", client.list_collections())

# 2. 检查 Embedding 模型
embeddings = OllamaEmbeddings(model="bge-m3")
vector = embeddings.embed_query("你好")
print("Embedding 成功，向量维度：", len(vector))   # bge-m3 应为 1024
