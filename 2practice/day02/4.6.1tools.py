from typing import Literal
from pydantic import BaseModel, Field
from langchain.tools import tool

# 第一步：写"参数说明书"
class OrderQuery(BaseModel):
    """查询订单所需的参数"""
    order_id: str = Field(description="订单号，格式如 A1001")
    detail: Literal["简要", "详细"] = Field(default="简要", description="返回信息的详细程度")

# 第二步：把说明书交给 @tool
@tool(args_schema=OrderQuery)
def query_order(order_id: str, detail: str = "简要") -> str:
    """根据订单号查询订单状态。用户问到订单、物流、发货时使用。"""
    fake_orders = {
        "A1001": {"状态": "已发货", "快递": "顺丰 SF123"},
        "A1002": {"状态": "待付款", "快递": "无"},
    }
    order = fake_orders.get(order_id)
    if order is None:
        return f"没有找到订单 {order_id}"
    if detail == "详细":
        return f"订单 {order_id}：{order['状态']}，快递：{order['快递']}"
    return f"订单 {order_id}：{order['状态']}"

# 看看模型会"看到"什么：工具名、工具描述、参数说明
print(query_order.name)          # query_order
print(query_order.description)   # 来自函数的 docstring
print(query_order.args)          # 来自 args_schema：每个参数的类型、描述、可选值

# 像普通函数一样测试
print(query_order.invoke({"order_id": "A1001", "detail": "详细"}))
