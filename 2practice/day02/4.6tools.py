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

print(get_weather.invoke({"city": "北京"}))
print(add_numbers.invoke({"a": 3,"b": 5}))
