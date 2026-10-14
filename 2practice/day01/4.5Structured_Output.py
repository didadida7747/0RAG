from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

# 第一步：定义你想要的数据结构（像一张空白表格）
class Movie(BaseModel):
    title: str = Field(description="电影名称")
    year: int = Field(description="上映年份")
    genre: str = Field(description="电影类型，如科幻、喜剧")

model = init_chat_model("openai:gpt-5-nano")

# 第二步：让模型按这张表格返回
structured_model = model.with_structured_output(Movie)

# 第三步：调用，拿到的是 Movie 对象，而不是乱七八糟的文字
movie = structured_model.invoke("介绍一下电影《盗梦空间》")
print(movie.title, movie.year, movie.genre)
print(movie.model_dump())   # 转成字典形式
