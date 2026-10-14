from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class Movie(BaseModel):
    title: str = Field(description="电影名称")
    year: int | None = Field(description="上映年份；文字中没有提到则为 null")
    genre: str | None = Field(description="电影类型，如科幻、喜剧；文字中没有提到则为 null")

# def build_models():
#     """返回 {显示名: 模型对象}。三个模型的调用方式完全一样。"""
#     names = [
#         "gpt-5.5",
#         "gpt-6-sol",
#         "gpt-6.1-sol",
#     ]
#     # "openai:" 前缀让 LangChain 用 OpenAI 兼容协议去调；
#     # 实际地址和密钥由 .env 里的 OPENAI_BASE_URL / OPENAI_API_KEY 决定
#     return {name: init_chat_model(f"openai:{name}") for name in names}

# # models = build_models()

models = {
      "中转站-A": init_chat_model("openai:gpt-6.1-sol"),
      "中转站-B": init_chat_model("openai:gpt-5.5"),
      "Ollama本地": ChatOllama(model="qwen2.5:7b"),
  }

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是信息提取助手。只根据用户给出的文字提取信息，"
               "文字里没有提到的内容填 null，不要编造。"),
    ("human", "请从下面的文字中提取电影信息：\n\n{text}"),
])

texts = [
    "《盗梦空间》是 2010 年上映的科幻电影，由克里斯托弗·诺兰执导。",
    "《大话西游》是一部喜剧电影。",   # 故意不给年份，看模型会不会编造
]



# for name, model in models.items():
#     structured_model = model.with_structured_output(Movie)

#     chain = prompt | structured_model

#     for text in texts:
#         try:
#             movie = chain.invoke({"text": text})
#             print(f"输入：{text}")
#             print(f"结果：{movie.title} | {movie.year} | {movie.genre}")
#             print(f"字典：{movie.model_dump()}")   
#         except Exception as e:
#             # 某个模型不可用时不影响其他模型
#             print(f"[{name}] 调用失败：{type(e).__name__}: {e}")
#         print("-" * 60)

for name, model in models.items():
      # 链跟着模型走，所以放在循环里
    try:
        chain = prompt | model.with_structured_output(Movie)
    except Exception as e:
        print(f"[{name}] 构建链失败：{type(e).__name__}: {e}")
        continue

    for text in texts:
        try:
            movie = chain.invoke({"text": text})
            print(f"[{name}] 输入：{text}")
            print(f"[{name}] 结果：{movie.title} | {movie.year} | {movie.genre}")
            print(f"[{name}] 字典：{movie.model_dump()}")
        except Exception as e:
            print(f"[{name}] 调用失败：{type(e).__name__}: {e}")
        print("-" * 60)