
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


# ============================================================
# 一、三种模型：同一个接口，只改配置
# ============================================================
def build_models():
    """返回 {显示名: 模型对象}。三个模型的调用方式完全一样。"""
    names = [
        "gpt-5.5",
        "gpt-6-sol",
        "gpt-6.1-sol",
    ]
    # "openai:" 前缀让 LangChain 用 OpenAI 兼容协议去调；
    # 实际地址和密钥由 .env 里的 OPENAI_BASE_URL / OPENAI_API_KEY 决定
    return {name: init_chat_model(f"openai:{name}") for name in names}


models = build_models()

question_prompt = ChatPromptTemplate.from_template("{question}")
question = "用一句话介绍 Python 语言。"

print("=" * 60)
print("一、同一个问题，三种模型各回答一次")
print("=" * 60)

for name, model in models.items():
    try:
        chain = question_prompt | model | StrOutputParser()
        print(f"[{name}] {chain.invoke({'question': question})}")
    except Exception as e:
        # 某个模型不可用时不影响其他模型
        print(f"[{name}] 调用失败：{type(e).__name__}: {e}")
    print("-" * 60)


# ============================================================
# 二、提示词模板 + 结构化输出
# ============================================================
class Movie(BaseModel):
    title: str = Field(description="电影名称")
    year: int | None = Field(description="上映年份；文字中没有提到则为 null")
    genre: str | None = Field(description="电影类型，如科幻、喜剧；文字中没有提到则为 null")


# 指令放 system（固定不变），数据放 human（用 {text} 占位）
extract_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是信息提取助手。只根据用户给出的文字提取信息，"
               "文字里没有提到的内容填 null，不要编造。"),
    ("human", "请从下面的文字中提取电影信息：\n\n{text}"),
])

texts = [
    "《盗梦空间》是 2010 年上映的科幻电影，由克里斯托弗·诺兰执导。",
    "《大话西游》是一部喜剧电影。",   # 故意不给年份，看模型会不会编造
]

print("=" * 60)
print("二、模板 + 结构化输出")
print("=" * 60)

# 格式转换由 with_structured_output 完成，链里不接 parser
extract_model = models["gpt-6.1-sol"]
extract_chain = extract_prompt | extract_model.with_structured_output(Movie)

for text in texts:
    try:
        movie = extract_chain.invoke({"text": text})
        print(f"输入：{text}")
        print(f"结果：{movie.title} | {movie.year} | {movie.genre}")
        print(f"字典：{movie.model_dump()}")
    except Exception as e:
        print(f"抽取失败：{type(e).__name__}: {e}")
    print("-" * 60)


# ============================================================
# 附：接真正的第三方厂商（配好 key 后取消注释，并 import ChatOpenAI）
# ============================================================
# from langchain_openai import ChatOpenAI
#
# models["DeepSeek"] = ChatOpenAI(
#     model="deepseek-chat",
#     base_url="https://api.deepseek.com",
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
# )
# models["智谱"] = ChatOpenAI(
#     model="glm-4-flash",
#     base_url="https://open.bigmodel.cn/api/paas/v4",
#     api_key=os.getenv("ZHIPU_API_KEY"),
# )
# models["Ollama本地"] = ChatOpenAI(        # 装好 Ollama 后可用，无需 key
#     model="qwen2.5:7b",
#     base_url="http://localhost:11434/v1",
#     api_key="ollama",
# )
