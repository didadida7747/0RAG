from typing import Any
from langchain.agents import create_agent, AgentState
from langchain.tools import tool
from langgraph.runtime import Runtime
from langchain.agents.middleware import before_model, after_model, ModelCallLimitMiddleware

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
    tools=[add_numbers],
    middleware=[
        log_before_model,
        log_after_model,
        ModelCallLimitMiddleware(run_limit=5),   # 自带中间件：一次任务最多调用模型 5 次
    ],
)

result = agent.invoke({"messages": [{"role": "user", "content": "12 加 30 等于多少？再加 8 呢？"}]})
print(result["messages"][-1].content)
