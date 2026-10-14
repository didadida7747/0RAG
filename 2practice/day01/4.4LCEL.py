from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# 从根目录的 .env 文件读取 API 密钥
load_dotenv()

model = init_chat_model("gpt-6.1-sol")

prompt = ChatPromptTemplate.from_template("请用一句有趣的话介绍{product}")
parser = StrOutputParser()   # 把模型的输出对象，转成纯字符串

# 流水线：提示词 → 模型 → 解析成字符串
chain = prompt | model | parser

# 调用流水线
result = chain.invoke({"product": "咖啡"})
print(result)   # 直接得到字符串，不用再写 .content
