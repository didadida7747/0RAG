from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# 从根目录的 .env 文件读取 API 密钥
load_dotenv()

# 格式："供应商:模型名"
# model = init_chat_model("gpt-6.1-sol")
model = init_chat_model(
    "gpt-6.1-sol",
    temperature=1,    # 0 = 最稳定、最一致；1 = 更有创意
    max_tokens=500,   # 限制最多输出多少 token，用来控制费用和长度
    timeout=30,       # 超时时间（秒）
)

# 1. invoke：一次性拿到完整回答（最常用）
response = model.invoke("什么是人工智能？请用三句话回答")
print(response.content)

# 2. stream：逐字流式输出（像打字机效果，体验更好）
for chunk in model.stream("讲一个关于程序员的冷笑话"):
    print(chunk.content, end="", flush=True)
print()

# 3. batch：同时处理多个问题
answers = model.batch(["1+1等于几？", "首都是北京的国家是哪个？"])
for a in answers:
    print(a.content)