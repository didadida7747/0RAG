from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

load_dotenv()

agent = create_agent(
    model = "openai:gpt-6.1-sol",
    tools = [],
    system_prompt = "你是一个乐于助人的助手，回答要简洁。",
    checkpointer = InMemorySaver(),
)

config = {"configurable": {"thread_id": "小明的对话"}}
agent.invoke({"messages": [{"role": "user", "content": "我叫小明，我喜欢打篮球"}]}, config = config)
result = agent.invoke({"messages": [{"role": "user", "content": "我叫什么？喜欢什么运动？"}]}, config = config)
print(result["messages"][-1].content)

other = {"configurable": {"thread_id": "小红的对话"}}
result = agent.invoke({"messages": [{"role": "user", "content": "我叫什么？喜欢什么运动？"}]}, config = other)
print(result["messages"][-1].content)

