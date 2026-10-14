from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import before_model, after_model, ModelCallLimitMiddleware
from dataclasses import dataclass
from typing import Any
from dotenv import load_dotenv 

load_dotenv()

@dataclass
class UserContext:
    user_id: str

FAKE_ORDERS = {
    "u001": ["A1001 机械键盘 已发货", "A1003 鼠标垫 已签收"],
    "u002": ["A1002 显示器 待付款"],
}

@tool
def list_my_orders(runtime: ToolRuntime[UserContext]) -> str:
    """查询当前登录用户的全部订单。"""
    user_id = runtime.context.user_id      # 从房卡上读，而不是让模型猜
    orders = FAKE_ORDERS.get(user_id, [])
    return "\n".join(orders) if orders else "该用户没有订单"

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

# 检查站 1：每次调用模型之前，打印一行日志
@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"[日志] 准备调用模型，当前共有 {len(state['messages'])} 条消息")
    return None   # 返回 None 表示"什么都不改，放行"

# 检查站 2：模型返回之后，看看它是要调工具还是直接回答
@after_model
def log_after_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    last = state["messages"][-1]
    if last.tool_calls:
        print(f"[日志] 模型要调用工具：{[c['name'] for c in last.tool_calls]}")
    else:
        print("[日志] 模型给出了最终回答")
    return None

agent = create_agent(
    model="openai:gpt-5-nano",
    tools=[list_my_orders, get_weather, add_numbers],
    context_schema=UserContext,      # 第三步：告诉智能体房卡的格式
    system_prompt="你是客服，回答简洁。",
    checkpointer = InMemorySaver(),
    middleware=[
        log_before_model,
        log_after_model,
        ModelCallLimitMiddleware(run_limit=5),   # 自带中间件：一次任务最多调用模型 5 次
    ],
)

config = {"configurable": {"thread_id": "day02-demo"}}
context = UserContext(user_id = "u001")

while True:
    question = input("你：")
    if question.strip() in ("quit", "exit", "退出"):
        break
    result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config = config,
            context = context,
        )
    print(result["messages"][-1].content)
        