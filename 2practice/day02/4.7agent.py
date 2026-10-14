from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气。

    Args:
        city: 城市名称，例如 北京、上海
    """
    fake_data = {"北京": "晴, 25℃", "上海": "多云, 22℃"}
    return fake_data.get(city, "暂无该城市的天气数据")

@tool
def add_numbers(a: int, b: int) -> int:
    """计算两个整数的和。"""
    return a + b

agent = create_agent(
    model = "openai:gpt-6.1-sol",
    tools = [get_weather, add_numbers],
    system_prompt = "你是一个乐于助人的助手，回答要简洁。",
)

result = agent.invoke({"messages": [{"role": "user", "content": "北京天气怎么样？另外，12+30等于多少"}]})

print(result["messages"][-1].content)
