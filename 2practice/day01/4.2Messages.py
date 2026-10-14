from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

# 从根目录的 .env 文件读取 API 密钥
load_dotenv()

model = init_chat_model(
    "gpt-6.1-sol",
    temperature=0,    # 0 = 最稳定、最一致；1 = 更有创意
    max_tokens=500,   # 限制最多输出多少 token，用来控制费用和长度
    timeout=30,       # 超时时间（秒）
)


messages = [
    SystemMessage("你是一位耐心的数学老师，回答要简短易懂。"),
    HumanMessage("什么是质数？"),
    AIMessage("质数是只能被 1 和它自己整除的大于 1 的整数，例如 2、3、5、7。"),
    HumanMessage("那 9 是质数吗？"),   # 模型能看到前面的问答，知道"那"指的是质数
]

response = model.invoke(messages)
print(response.content)