from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class Movie(BaseModel):
    title: str = Field(description="电影名称")
    year: int = Field(description = "上映年份")
    genre: str = Field(description = "电影类型，如科幻，喜剧")

model = init_chat_model(
    "gpt-6.1-sol",
    temperature=0,    # 0 = 最稳定、最一致；1 = 更有创意
    max_tokens=500,   # 限制最多输出多少 token，用来控制费用和长度
    timeout=30,       # 超时时间（秒）
)

prompt = ChatPromptTemplate.from_messages([
      ("system", "你是信息提取助手。只根据用户给出的文字提取信息，文字里没有提到的内容不要编造。"),
      ("human", "请从下面的文字中提取电影信息：\n\n{text}"),
  ])

structured_model = model.with_structured_output(Movie)

chain = prompt | structured_model

movie = chain.invoke({"text": "盗梦空间"})

print(movie.title, movie.year, movie.genre)
print(movie.model_dump())   # 转成字典形式
