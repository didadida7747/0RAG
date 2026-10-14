from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# 从根目录的 .env 文件读取 API 密钥
load_dotenv()


# 方式一：from_template，适合单条提示词
prompt = ChatPromptTemplate.from_template("请把下面这句话翻译成{language}：{text}")

# 填入具体内容，得到可以发送给模型的消息
filled = prompt.invoke({"language": "英文", "text": "今天天气真好"})
print(filled.to_messages())

# 方式二：from_messages，适合包含系统设定和多轮对话的复杂提示词
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位专业的{role}，回答要专业且简洁。"),
    ("human", "{question}"),
])
filled = prompt.invoke({"role": "营养师", "question": "早餐吃什么比较健康？"})
