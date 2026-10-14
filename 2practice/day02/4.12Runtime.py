from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime

# 第一步：定义"房卡"上写什么（运行时上下文的格式）
@dataclass
class UserContext:
    user_id: str

FAKE_ORDERS = {
    "u001": ["A1001 机械键盘 已发货", "A1003 鼠标垫 已签收"],
    "u002": ["A1002 显示器 待付款"],
}

# 第二步：工具加一个 runtime 参数，类型写 ToolRuntime[UserContext]
#        这个参数对模型是"隐形"的：模型看不到它，也没法乱填
@tool
def list_my_orders(runtime: ToolRuntime[UserContext]) -> str:
    """查询当前登录用户的全部订单。"""
    user_id = runtime.context.user_id      # 从房卡上读，而不是让模型猜
    orders = FAKE_ORDERS.get(user_id, [])
    return "\n".join(orders) if orders else "该用户没有订单"

print(list_my_orders.tool_call_schema.model_json_schema())  # 模型看到的参数：没有 runtime

agent = create_agent(
    model="openai:gpt-5-nano",
    tools=[list_my_orders],
    context_schema=UserContext,      # 第三步：告诉智能体房卡的格式
    system_prompt="你是电商客服，回答简洁。",
)

# 第四步：调用时用 context= 发房卡（真实项目中来自登录信息）
for uid in ["u001", "u002"]:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "我买了哪些东西？"}]},
        context=UserContext(user_id=uid),
    )
    print(uid, "→", result["messages"][-1].content)
