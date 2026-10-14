from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# 从根目录的 .env 文件读取 API 密钥
load_dotenv()

# "供应商:模型名" 的格式，这里使用 OpenAI 的一个小模型
# 模型名会更新，如果报错请到官方文档查看当前可用的模型名
model = init_chat_model("gpt-6.1-sol")

response = model.invoke("用一句话介绍你自己")
print(response.content)

