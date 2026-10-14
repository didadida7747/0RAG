# LangChain 入门完全指南（面向初学者）

> **适用版本**：LangChain 1.x（Python 版）
> **更新时间**：2026 年
> **说明**：LangChain 更新很快，代码写法以官方文档为准。官方文档地址：https://docs.langchain.com/oss/python/langchain/overview
> **阅读建议**：先通读第 0～2 章建立整体印象，再按第 4 章逐节动手敲代码，最后做第 7 章的练习。
> **配合逐日规划使用**：按《05-阶段二RAG就业班10天逐日规划》学习时，先看第 4 章开头的「逐日对照表」，找到当天对应的小节。RAG（Day 3 下午～Day 5）请看配套的《RAG入门指南.md》。

---

## 目录

- [第 0 章　先搞懂：LangChain 到底是什么？](#第-0-章先搞懂langchain-到底是什么)
- [第 1 章　学习地图（总览）](#第-1-章学习地图总览)
- [第 2 章　环境准备（Windows）](#第-2-章环境准备windows)
- [第 3 章　核心概念速查表](#第-3-章核心概念速查表)
- [第 4 章　知识点详解（含代码）](#第-4-章知识点详解含代码)
- [第 5 章　常见坑与排错](#第-5-章常见坑与排错)
- [第 6 章　学习路线（四周计划）](#第-6-章学习路线四周计划)
- [第 7 章　练习项目](#第-7-章练习项目)
- [第 8 章　学习资源](#第-8-章学习资源)
- [第 9 章　小结与自测清单](#第-9-章小结与自测清单)

---

## 第 0 章　先搞懂：LangChain 到底是什么？

### 0.1 一个比喻

把大模型（如 ChatGPT、Claude、DeepSeek 背后的模型）想象成一个**非常聪明、但手脚被绑住的大脑**：

- 它能聊天、能写文章，但**不能直接查天气、不能读你的文件、不能记住上一次聊了什么**。
- **LangChain 就是给这个大脑装上"手、眼睛和笔记本"的工具箱**。

### 0.2 它解决了什么问题？

| 痛点 | LangChain 的做法 |
|---|---|
| 不同厂商的模型 API 格式各不相同，换一家就要重写代码 | 提供**统一接口**，换模型通常只改一行 |
| 提示词（给模型的指令）写死在代码里，难维护 | 提供**提示词模板**，可以带"空位"，运行时填入内容 |
| 模型不知道实时信息 | 让模型可以调用**工具**（查天气、查数据库、算数…） |
| 模型记不住之前的对话 | 提供**记忆**功能 |
| 模型不了解你自己的文档 | 提供 **RAG**（检索增强生成）：先从你的资料里找相关内容，再让模型回答 |
| 多个步骤要串起来很麻烦 | 提供**链（Chain）**和**智能体（Agent）**，把步骤组合起来 |

### 0.3 LangChain 家族的三兄弟

| 名称 | 角色 | 类比 | 本指南的位置 |
|---|---|---|---|
| **LangChain** | 核心框架，写应用的"积木" | 乐高基础积木 | 本指南重点 |
| **LangGraph** | 用"流程图"描述复杂流程，支持分支、循环、暂停等待人工确认 | 施工流程图 | 第 4.10 节简介 |
| **LangSmith** | 观测与调试平台，能看到每一步发生了什么 | 行车记录仪 | 第 4.11 节简介 |

> **一句话记住**：LangChain 负责"拼积木"，LangGraph 负责"画流程图"，LangSmith 负责"查录像"。

---

## 第 1 章　学习地图（总览）

下面是整个 LangChain 的知识地图。**从上往下学**，每一层都建立在上一层的基础上。

```
┌─────────────────────────────────────────────────────────────────┐
│  【第 6 层】工程化     LangSmith 调试/评测 · 部署上线 · 成本控制      │
├─────────────────────────────────────────────────────────────────┤
│  【第 5 层】进阶编排     LangGraph 状态图 · 多智能体 · 人工审批        │
├─────────────────────────────────────────────────────────────────┤
│  【第 4 层】知识库(RAG)  文档加载 → 切分 → 向量化 → 向量库 → 检索器    │
├─────────────────────────────────────────────────────────────────┤
│  【第 3 层】智能体       工具 Tools · 智能体 Agent · 记忆 Memory      │
├─────────────────────────────────────────────────────────────────┤
│  【第 2 层】组合编排     LCEL 管道 (|) · Runnable 接口 · 结构化输出    │
├─────────────────────────────────────────────────────────────────┤
│  【第 1 层】基础积木     模型 Model · 消息 Message · 提示词模板 Prompt  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.1 每层学会后能做什么？

| 学完这一层 | 你就能做出 |
|---|---|
| 第 1 层 | 一个会翻译、会润色文字的命令行小工具 |
| 第 2 层 | 一条"输入 → 提示词 → 模型 → 整理输出"的完整流水线 |
| 第 3 层 | 一个能查天气、会计算、记得上下文的智能助手 |
| 第 4 层 | 一个能回答你自己 PDF / 笔记内容的问答机器人 |
| 第 5 层 | 一个有分支判断、需要人工审批的复杂工作流 |
| 第 6 层 | 能上线、能监控、能控制费用的正式应用 |

---

## 第 2 章　环境准备（Windows）

> 本章所有命令都在 **PowerShell** 中执行。打开方法：点击开始菜单，搜索 `PowerShell` 并打开。

### 2.1 安装 Python

需要 **Python 3.12 或更高版本**（和项目 `pyproject.toml` 的要求一致）。

**安装步骤**：
1. 打开 https://www.python.org/downloads/ ，下载最新版的 Windows 安装包。
2. 运行安装包时，**一定要勾选窗口底部的 "Add python.exe to PATH"**。
   > **类比**：PATH 就像电脑的"通讯录"，勾选后，任何地方输入 `python` 电脑都能找到它。不勾选，后面所有命令都会报错。
3. 安装完成后，在 PowerShell 中检查：

```powershell
python --version
```

如果显示 `Python 3.12.x` 或更高，就可以继续。

> **如果输入 `python` 后弹出 Microsoft Store 或提示"不是内部或外部命令"**：说明没有勾选 PATH。重新运行安装包，选择"修改"（Modify）并勾选 PATH；或者卸载后重装。也可以先试试 `py --version`。

### 2.2 认识 uv：自动管理环境与依赖

**类比**：`uv` 是一个管家。你说"我要用 langchain"，它会自动给项目准备一个独立的"工具抽屉"（`.venv` 文件夹），并把需要的包都装进去。项目之间的依赖互不干扰。

本指南统一使用 **uv**（比 pip 快很多，并且会自动记录你装了哪些包）。你不需要手动创建或激活环境。

**先确认 uv 可以使用**：

```powershell
uv --version
```

如果显示版本号（如 `uv 0.12.10`），就可以继续。如果提示"无法识别"，说明 uv 所在的文件夹没有加入 Windows 的 PATH，见第 5 章排错表。

**进入项目根目录**（把路径换成你电脑上 `0RAG` 文件夹的实际位置）：

```powershell
cd E:/0RAG
```

之后所有 `uv` 命令都在这个根目录下执行。

### 2.3 安装 LangChain 与模型插件

```powershell
uv add langchain langchain-openai
```

**说明**：
- `langchain`：核心框架（本指南的主角）。
- `langchain-openai`：让 LangChain 能连接 OpenAI 模型的"插件"。LangChain 采用**模块化**设计，用哪家模型就装哪家的插件。
- 想用 Anthropic（Claude）则装 `langchain-anthropic`；想用 Google 则装 `langchain-google-genai`。
- `uv add` 会把包记录在 `pyproject.toml` 中，换电脑时运行 `uv sync` 就能装回全部依赖。

> **pip 与 uv 的对应关系**：`pip install 包名` 对应 `uv add 包名`；`pip install -U 包名` 对应 `uv lock --upgrade-package 包名`；`python xxx.py` 对应 `uv run python xxx.py`。

### 2.4 设置 API Key

API Key 相当于**访问模型的门票**，必须自己去对应平台申请。

**推荐方法：`.env` 文件（只对本项目生效）**

1. 安装读取 `.env` 的小工具（在根目录执行）：
   ```powershell
   uv add python-dotenv
   ```
2. 在 `0RAG` 根目录新建一个名为 `.env` 的**文件**（不是文件夹，也不是 `.txt`）。用 VS Code 新建最不容易出错，见第 5 章排错表。
3. 文件里只写一行，把 `sk-你的密钥` 替换成你自己的密钥：
   ```
   OPENAI_API_KEY=sk-你的密钥
   ```
4. 运行示例前，在代码**最上面**加两行（第 2.5 节和第 4 章的示例都要这样做）：
   ```python
   from dotenv import load_dotenv
   load_dotenv()
   ```

> **类比**：`.env` 像一张写着密钥的便签，贴在项目根目录。`load_dotenv()` 就是让程序去读这张便签。
>
> `.env` 已经写在 `.gitignore` 里，git 不会上传它。

**备选方法：Windows 环境变量（所有项目都能用）**

```powershell
# 临时设置（只对当前 PowerShell 窗口有效，关掉窗口就失效）
$env:OPENAI_API_KEY = "sk-你的密钥"

# 永久设置（写入 Windows 用户环境变量）
setx OPENAI_API_KEY "sk-你的密钥"
```

> **注意**：`setx` 设置的是**永久**变量，但**只对之后新打开的窗口生效**，当前窗口仍然读不到。设置完请关闭并重新打开 PowerShell。
>
> 用这种方法时，示例代码**不需要**加 `load_dotenv()` 两行。如果你用的是 cmd（命令提示符），临时设置写法是 `set OPENAI_API_KEY=sk-你的密钥`。

> **安全警告**：
> - **不要**把 API Key 直接写进代码并提交到 git 或发给别人。
> - **不要**把 Key 截图分享。
> - 如果不小心泄露，立即去平台后台作废它并重新生成。

> **国内用户补充**：很多国内模型（如 DeepSeek、通义千问等）提供了与 OpenAI 兼容的接口，可以用 `ChatOpenAI(base_url=..., api_key=...)` 的方式接入。具体地址和模型名请以对应平台文档为准。

### 2.5 验证环境是否正常

新建文件 `hello.py`，放在 `0RAG` 根目录下（用 VS Code 新建，保存时编码选 UTF-8），写入下面内容：

```python
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# 读取根目录 .env 中的密钥（第 2.4 节）
load_dotenv()

# "供应商:模型名" 的格式，这里使用 OpenAI 的一个小模型
# 模型名会更新，如果报错请到官方文档查看当前可用的模型名
model = init_chat_model("openai:gpt-5-nano")

response = model.invoke("用一句话介绍你自己")
print(response.content)
```

```powershell
uv run python hello.py
```

能看到模型的回答，说明环境已经配置成功。

> **为什么用 `uv run python`？** `uv run` 会自动使用项目的 `.venv`，不需要手动激活。

---

## 第 3 章　核心概念速查表

学习前先混个脸熟，后面遇到不懂的术语可以回来查。

| 英文 | 中文 | 通俗理解 |
|---|---|---|
| **Model** | 模型 | 那个"聪明的大脑"，负责理解和生成文字 |
| **Message** | 消息 | 对话里的一条话。分为 `System`（系统设定）、`Human`（用户说的）、`AI`（模型回的）、`Tool`（工具返回的） |
| **Prompt** | 提示词 | 你给模型的指令或问题 |
| **Prompt Template** | 提示词模板 | 带"空位"的提示词，比如 `"把{text}翻译成英文"`，运行时填入具体内容 |
| **Runnable** | 可运行对象 | LangChain 里所有"积木"的统一接口，都有 `invoke`、`stream`、`batch` 三个方法 |
| **Chain** | 链 | 把多个步骤像流水线一样串起来 |
| **LCEL** | 表达式语言 | 用 `\|` 符号串联积木的写法，例如 `prompt \| model` |
| **Output Parser** | 输出解析器 | 把模型返回的内容整理成你想要的格式（比如纯文本、JSON） |
| **Structured Output** | 结构化输出 | 让模型按你定义的"表格格式"返回数据 |
| **Tool** | 工具 | 一个你写的函数，模型可以决定何时调用它（如查天气） |
| **Agent** | 智能体 | 会自己思考、自己决定调用哪个工具、直到完成任务的"执行者" |
| **Memory** | 记忆 | 让模型记住之前的对话内容 |
| **Checkpointer** | 检查点存储 | 把对话状态保存下来的"存档仓库" |
| **Thread ID** | 会话编号 | 区分不同对话的编号，类似"聊天窗口的名字" |
| **Document** | 文档 | 一段文字加上它的来源信息（如来自哪个文件） |
| **Document Loader** | 文档加载器 | 把 PDF、txt、网页等读进 LangChain 的工具 |
| **Text Splitter** | 文本切分器 | 把长文档切成小段，方便检索 |
| **Embedding** | 向量化 / 嵌入 | 把文字变成一串数字，**意思相近的文字，数字也相近** |
| **Vector Store** | 向量库 | 存放这些数字，并能快速找出"最相似"内容的数据库 |
| **Retriever** | 检索器 | 输入一个问题，返回最相关的文档片段 |
| **RAG** | 检索增强生成 | 先查资料，再让模型基于资料回答。类比"开卷考试" |
| **Token** | 令牌 | 模型计算和计费的最小单位，大致相当于一个汉字或一个英文单词片段 |
| **Temperature** | 温度 | 控制回答的随机性。越低越稳定、越高越有创意 |
| **LangGraph** | 图框架 | 用节点和连线描述复杂流程的框架 |
| **LangSmith** | 观测平台 | 记录和分析每次模型调用的工具 |
| **Tool Call** | 工具调用请求 | 模型说"我想调用某某工具，参数是……"。注意它只是**请求**，真正执行函数的是程序 |
| **args_schema** | 参数结构 | 用 Pydantic 写的"参数说明书"，告诉模型每个参数叫什么、什么类型、什么意思 |
| **tool_choice** | 工具选择策略 | 控制模型"可以不调用 / 必须调用 / 必须调用某一个"工具 |
| **Runtime** | 运行时 | 智能体运行时随身带的"工具包"，里面有上下文、状态、存储等 |
| **Context** | 运行时上下文 | 调用时由程序传入的固定信息（如当前用户 ID），**模型看不到、也改不了** |
| **Middleware** | 中间件 | 插在智能体循环固定位置的"检查站"，可以记日志、限流、人工审批 |
| **Hook** | 钩子 | 中间件挂载的位置，如 `before_model`（调用模型前）、`after_model`（调用模型后） |
| **MCP** | 模型上下文协议 | 给 AI 应用接入外部能力的"统一插座标准" |
| **MCP Server / Client** | MCP 服务端 / 客户端 | Server 提供工具（电器），Client 接入并使用工具（插座） |

---

## 第 4 章　知识点详解（含代码）

> 下面每个小节都可以独立运行。建议在 `2practice/day01` ～ `day10` 中，每节新建一个 `.py` 文件跟着敲，运行时用 `uv run python 文件名.py`。
>
> **重要**：如果你用的是第 2.4 节的 `.env` 方法，每个示例文件**最上面**都要先加上 `from dotenv import load_dotenv` 和 `load_dotenv()` 两行，否则会报 `Missing credentials`。为了让示例简洁，下面的代码块不再重复这两行。
>
> **关于模型名**：示例统一写 `"openai:gpt-5-nano"`。你在练习文件里用的是自己的模型名（如 `4.7agent.py` 里那个），照旧替换成你的即可。

### 逐日对照表（配合逐日规划使用）

| 逐日规划 | 学习要点 | 看本指南哪一节 |
|---|---|---|
| **Day 1** | Models、Messages、Prompts | 4.1 模型 · 4.2 消息 · 4.3 提示词模板 · 4.4 LCEL · 4.5 结构化输出 |
| **Day 2** | Tools（`@tool`、`args_schema`、`tool_choice`） | 4.6 工具（含 4.6.1、4.6.2 补充）· 4.7 智能体 |
| **Day 2** | Memory（checkpointer 短期记忆） | 4.8 记忆 |
| **Day 2** | Runtime（把用户 ID 等传给工具） | **4.12 运行时 Runtime** |
| **Day 2** | Middleware（只讲概念） | **4.13 中间件 Middleware** |
| **Day 2** | 动手任务：带工具、有记忆的对话脚本 | **4.15 Day 2 综合任务思路** |
| **Day 3 上午** | MCP 入门 | **4.14 MCP** |
| **Day 3 下午～Day 5** | RAG | 4.9 是简介，详细内容见《RAG入门指南.md》 |
| **Day 6～8** | LangGraph | 4.10 是简介，到时另发《LangGraph入门指南》 |

> 4.12～4.15 是后来补充的小节，编号排在 LangGraph、LangSmith 简介之后，是为了不打乱你已经按编号命名的练习文件。**学习顺序以上表为准**。

### 4.1 模型（Chat Model）

**核心思想**：LangChain 用 `init_chat_model` 作为**统一入口**，不管底层是哪家模型，调用方式都一样。

```python
from langchain.chat_models import init_chat_model

# 格式："供应商:模型名"
model = init_chat_model("openai:gpt-5-nano")

# 1. invoke：一次性拿到完整回答（最常用）
response = model.invoke("什么是人工智能？请用三句话回答")
print(response.content)

# 2. stream：逐字流式输出（像打字机效果，体验更好）
for chunk in model.stream("讲一个关于程序员的冷笑话"):
    print(chunk.content, end="", flush=True)
print()

# 3. batch：同时处理多个问题
answers = model.batch(["1+1等于几？", "首都是北京的国家是哪个？"])
for a in answers:
    print(a.content)
```

**常用参数**：

```python
model = init_chat_model(
    "openai:gpt-5-nano",
    temperature=0,    # 0 = 最稳定、最一致；1 = 更有创意
    max_tokens=500,   # 限制最多输出多少 token，用来控制费用和长度
    timeout=30,       # 超时时间（秒）
)
```

**要点**：
- `invoke` 返回的是一个 **消息对象**（`AIMessage`），真正的文字在 `.content` 里。
- 写代码时，如果需要稳定的结果（如数据提取），把 `temperature` 设为 `0`。

---

### 4.2 消息（Messages）

**核心思想**：模型并不是"一句话一句话地听"，而是**整段对话历史一起看**。所以对话被表示为一个"消息列表"。

| 消息类型 | 谁说的 | 作用 |
|---|---|---|
| `SystemMessage` | 开发者 | 设定角色和规则，例如"你是一位耐心的数学老师" |
| `HumanMessage` | 用户 | 用户的问题或指令 |
| `AIMessage` | 模型 | 模型的回答 |
| `ToolMessage` | 工具 | 工具执行后返回的结果 |

```python
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage

model = init_chat_model("openai:gpt-5-nano")

messages = [
    SystemMessage("你是一位耐心的数学老师，回答要简短易懂。"),
    HumanMessage("什么是质数？"),
    AIMessage("质数是只能被 1 和它自己整除的大于 1 的整数，例如 2、3、5、7。"),
    HumanMessage("那 9 是质数吗？"),   # 模型能看到前面的问答，知道"那"指的是质数
]

response = model.invoke(messages)
print(response.content)
```

**要点**：
- 想让模型"记得"前文，就要把前文的消息都放进列表里再发送。这也是后面"记忆"功能的基本原理。

---

### 4.3 提示词模板（Prompt Template）

**核心思想**：不要把提示词写死，而是做成**带空位的模板**，运行时填入不同内容。好比**Word 的邮件合并功能**。

```python
from langchain_core.prompts import ChatPromptTemplate

# 方式一：from_template，适合单条提示词
prompt = ChatPromptTemplate.from_template("请把下面这句话翻译成{language}：{text}")

# 填入具体内容，得到可以发送给模型的消息
filled = prompt.invoke({"language": "英文", "text": "今天天气真好"})
print(filled.to_messages())

# 方式二：from_messages，适合包含系统设定和多轮对话的复杂提示词
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位专业的{role}，回答要专业且简洁。"),
    ("human", "{question}"),
])
filled = prompt.invoke({"role": "营养师", "question": "早餐吃什么比较健康？"})
```

**要点**：
- `{变量名}` 就是"空位"，调用 `invoke({...})` 时传入对应的值。
- 空位名字要和字典的键**完全一致**，否则会报错。

---

### 4.4 LCEL 管道：用 `|` 把积木串起来

**核心思想**：LCEL（LangChain Expression Language）用 `|` 符号把多个步骤连成流水线，数据从左流到右。**类比工厂流水线**：原料 → 加工 → 包装 → 成品。

```python
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = init_chat_model("openai:gpt-5-nano")

prompt = ChatPromptTemplate.from_template("请用一句有趣的话介绍{product}")
parser = StrOutputParser()   # 把模型的输出对象，转成纯字符串

# 流水线：提示词 → 模型 → 解析成字符串
chain = prompt | model | parser

# 调用流水线
result = chain.invoke({"product": "咖啡"})
print(result)   # 直接得到字符串，不用再写 .content
```

**要点**：
- 链也是 **Runnable**，同样支持 `invoke`、`stream`、`batch`。
- `StrOutputParser` 是最常用的"出口"，把复杂的消息对象变成普通字符串。
- **阅读方法**：`A | B | C` 读作"先做 A，把结果交给 B，再把结果交给 C"。

---

### 4.5 结构化输出（Structured Output）

**核心思想**：有时你需要模型返回**固定格式的数据**（如姓名、年龄、评分），而不是一段自由文字。用 **Pydantic**（一个数据校验库）定义"表格的格式"，模型就会按格式返回。

```python
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
```

**要点**：
- `Field(description=...)` 的描述很重要，相当于告诉模型"这一栏填什么"。
- 这是做**信息提取**（从合同、简历、订单中抽取字段）最常用的方法。

---

### 4.6 工具（Tools）

**核心思想**：工具就是你自己写的普通 Python 函数，用 `@tool` 装饰一下，模型就能"知道"并决定何时使用它。

```python
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气。

    Args:
        city: 城市名称，例如 北京、上海
    """
    # 这里是示例数据；真实项目中可以调用天气 API
    fake_data = {"北京": "晴，25°C", "上海": "多云，22°C"}
    return fake_data.get(city, "暂无该城市的天气数据")

@tool
def add_numbers(a: int, b: int) -> int:
    """计算两个整数的和。"""
    return a + b

# 工具也可以像普通函数一样直接测试
print(get_weather.invoke({"city": "北京"}))
print(add_numbers.invoke({"a": 3, "b": 5}))
```

**要点**（非常重要）：
- **函数的文档字符串（三引号里的说明）会被发给模型**，模型据此判断"什么时候该用这个工具"。所以文档要写清楚！
- **类型标注**（如 `city: str`）告诉模型参数应该是什么类型。
- 工具就像给模型配的"计算器、搜索引擎、数据库查询"等外挂。

#### 4.6.1 补充：用 `args_schema` 把参数说清楚（Day 2）

**核心思想**：参数一多，光靠类型标注和 docstring 不够清楚。`args_schema` 用 Pydantic 写一张**"参数说明书"**：每个参数叫什么、什么类型、什么意思、默认值是多少、只能取哪几个值。**类比**：网购时的下单表单，"收货地址"、"配送方式（快递/自提）"每栏都写清楚，顾客就不容易填错。

```python
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
```

**要点**：
- `Literal["简要", "详细"]` 限定参数**只能取这几个值**，能有效减少模型乱填。
- 另一种轻量写法：`@tool(parse_docstring=True)`，它会读取 docstring 里 `Args:` 部分的参数说明。参数少时用它，参数多、有限定值时用 `args_schema`。
- `print(工具.args)` 是检查"模型看到了什么"的好办法。模型不调用或乱调用工具时，先打印出来看看描述清不清楚。

#### 4.6.2 补充：模型只"点菜"，不"做菜"——`bind_tools` 与 `tool_choice`（Day 2）

**核心思想**（Day 2 最关键的理解）：**模型并不执行函数，它只决定调用哪个函数、传什么参数。** **类比**：模型是餐厅里的顾客，只负责看菜单（工具描述）点菜（tool call），真正做菜的是后厨（你的程序）。智能体（4.7 节）只是帮你把"点菜 → 做菜 → 上菜 → 再点菜"这个循环自动化了。

下面的代码不用智能体，手动走一遍这个过程，看清每一步：

```python
from langchain.chat_models import init_chat_model
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气。"""
    print(f"  >>> 后厨开始做菜：get_weather({city})")   # 用来观察函数是否真的被执行
    return {"北京": "晴，25°C", "上海": "多云，22°C"}.get(city, "暂无数据")

model = init_chat_model("openai:gpt-5-nano")

# bind_tools：把"菜单"交给模型
model_with_tools = model.bind_tools([get_weather])

messages = [{"role": "user", "content": "北京天气怎么样？"}]

# 第 1 步：模型点菜
ai_msg = model_with_tools.invoke(messages)
print("模型的文字回答：", repr(ai_msg.content))   # 通常是空的
print("模型点的菜：", ai_msg.tool_calls)
# 类似 [{'name': 'get_weather', 'args': {'city': '北京'}, 'id': 'call_xxx', 'type': 'tool_call'}]
# 注意：到这里为止，">>> 后厨开始做菜" 一次都没有打印！
messages.append(ai_msg)

# 第 2 步：程序做菜（执行函数）
for call in ai_msg.tool_calls:
    tool_msg = get_weather.invoke(call)   # 传入整个 tool_call，返回的是 ToolMessage
    print("工具结果：", tool_msg.content)
    messages.append(tool_msg)

# 第 3 步：把结果交回模型，让它组织成最终回答
final = model_with_tools.invoke(messages)
print("最终回答：", final.content)
```

**`tool_choice`：控制模型"要不要点菜"**

```python
from langchain.chat_models import init_chat_model
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气。"""
    return "晴，25°C"

model = init_chat_model("openai:gpt-5-nano")

auto_model = model.bind_tools([get_weather])                             # 默认：模型自己决定
must_model = model.bind_tools([get_weather], tool_choice="any")          # 必须调用某个工具
pick_model = model.bind_tools([get_weather], tool_choice="get_weather")  # 必须调用指定的工具

print(auto_model.invoke("你好").tool_calls)   # 多半是 []：打个招呼不需要查天气
print(must_model.invoke("你好").tool_calls)   # 就算只是打招呼，也会硬调一次工具
```

| `tool_choice` 的值 | 含义 | 适合场景 |
|---|---|---|
| 不写（默认） | 模型自己决定调不调用 | 绝大多数情况，包括智能体 |
| `"any"` | 必须调用工具，调哪个模型自己选 | 确定这一步一定要用工具，如"必须先查数据库" |
| `"工具名"` | 必须调用这一个工具 | 固定流程中的某一步，如"必须先做信息提取" |

**要点**：
- `tool_calls` 是一个**列表**，模型一次可以点多道菜（同时调用多个工具）。
- **不要在智能体里强制 `tool_choice="any"`**：智能体每一轮都要先问模型，强制调用会让它每轮都调工具，永远停不下来。
- 少数模型或第三方接口不支持 `tool_choice`，报错时先去掉这个参数，确认是不是它引起的。

---

### 4.7 智能体（Agent）

**核心思想**：智能体 = **模型 + 工具箱 + 一个"想→做→看结果→再想"的循环**。你只给一个目标，它自己决定要用哪些工具、用几次，直到完成任务。

```
用户提问 → 模型思考 → 需要工具？
              ├── 是 → 调用工具 → 拿到结果 → 回到"模型思考"
              └── 否 → 给出最终答案
```

```python
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气。"""
    fake_data = {"北京": "晴，25°C", "上海": "多云，22°C"}
    return fake_data.get(city, "暂无数据")

@tool
def add_numbers(a: int, b: int) -> int:
    """计算两个整数的和。"""
    return a + b

agent = create_agent(
    model="openai:gpt-5-nano",          # 直接用字符串指定模型
    tools=[get_weather, add_numbers],   # 给智能体配的工具箱
    system_prompt="你是一个乐于助人的助手，回答要简洁。",
)

# 注意：输入是"消息列表"
result = agent.invoke({
    "messages": [{"role": "user", "content": "北京天气怎么样？另外 12 加 30 等于多少？"}]
})

# 最后一条消息就是智能体的最终回答
print(result["messages"][-1].content)
```

**要点**：
- `create_agent` 是 LangChain 1.x 中**创建智能体的标准方式**。网上很多旧教程用的 `initialize_agent`、`AgentExecutor` 已经过时。
- 你不需要告诉智能体"先查天气再做加法"，它会根据问题自己规划。
- 智能体的行为质量很大程度取决于：**工具的文档写得好不好**、**system_prompt 写得清不清楚**。

---

### 4.8 记忆（Memory）

**核心思想**：记忆分两种。

| 类型 | 含义 | 类比 | LangChain 的实现 |
|---|---|---|---|
| **短期记忆** | 记住同一次对话里的内容 | 聊天窗口的聊天记录 | `checkpointer` + `thread_id` |
| **长期记忆** | 跨越多次对话，长期保存的信息（如用户偏好） | 客户的档案卡 | `store`（进阶，可在 LangGraph 中了解） |

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver   # 内存中的"存档仓库"

agent = create_agent(
    model="openai:gpt-5-nano",
    tools=[],
    checkpointer=InMemorySaver(),   # 有了它，智能体才会保存对话状态
)

# thread_id 是"聊天窗口编号"：同一个编号 = 同一段对话
config = {"configurable": {"thread_id": "小明的对话"}}

agent.invoke({"messages": [{"role": "user", "content": "我叫小明，我喜欢打篮球"}]}, config=config)
result = agent.invoke({"messages": [{"role": "user", "content": "我叫什么？喜欢什么运动？"}]}, config=config)
print(result["messages"][-1].content)   # 能回答出"小明"和"篮球"

# 换一个 thread_id，就是全新的对话，它不记得小明
other = {"configurable": {"thread_id": "小红的对话"}}
result = agent.invoke({"messages": [{"role": "user", "content": "我叫什么？"}]}, config=other)
print(result["messages"][-1].content)   # 不知道
```

**要点**：
- `InMemorySaver` 存在内存里，**程序一关就丢失**。正式项目应换成持久化存储（如 SQLite、PostgreSQL 的 checkpointer），具体见官方文档。
- 不同用户、不同对话，一定要用**不同的 thread_id**，否则会串台。
- **过时写法提醒**：网上老教程里的 `ConversationBufferMemory`、`ConversationChain`、`RunnableWithMessageHistory` 都是旧版的记忆写法。LangChain 1.x 中智能体的短期记忆统一用 **checkpointer + thread_id**。

---

### 4.9 RAG：让模型读懂你的资料

> 本节是 RAG 的**简介**，帮你先跑通一个最小例子。逐日规划 Day 3 下午～Day 5 的完整内容（多格式加载、切分对比、Milvus 向量库、MMR、带来源和拒答的企业问答、评估）请看《RAG入门指南.md》。

**核心思想**：模型只学过公开资料，**不认识你的私有文档**。RAG 的办法是：**提问时，先从你的文档里找出相关段落，把它们连同问题一起交给模型**。就像**开卷考试**：模型不用背下所有内容，翻书找到答案即可。

#### 4.9.1 RAG 的完整流程

```
【准备阶段（只做一次）】
  文档(PDF/txt/网页)
     ↓ ① 加载 Document Loader
  原始文本
     ↓ ② 切分 Text Splitter
  一个个小片段 (chunk)
     ↓ ③ 向量化 Embedding
  一串串数字（向量）
     ↓ ④ 存入 Vector Store

【问答阶段（每次提问都做）】
  用户问题
     ↓ ⑤ 检索 Retriever：找出最相关的几段
  相关片段
     ↓ ⑥ 拼接成提示词，连同问题一起交给模型
  模型基于资料生成回答
```

#### 4.9.2 代码实现

先安装所需插件：

```powershell
uv add langchain-community langchain-text-splitters langchain-openai
```

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chat_models import init_chat_model

# ① 加载文档（先准备一个 notes.txt 文件，随便写几段文字即可）
docs = TextLoader("notes.txt", encoding="utf-8").load()

# ② 切分文档
#    chunk_size：每段最多多少字符；chunk_overlap：相邻段之间重叠多少字符，防止一句话被从中间切断
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# ③ + ④ 向量化并存入向量库
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(chunks)

# ⑤ 转成检索器：每次返回最相关的 3 段
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# ⑥ 组装问答链
def format_docs(docs):
    """把检索到的多个片段拼成一段文字"""
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_template(
    "请只根据下面的资料回答问题，如果资料中没有答案，就说“资料中没有相关信息”。\n\n"
    "资料：\n{context}\n\n"
    "问题：{question}"
)

model = init_chat_model("openai:gpt-5-nano")

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

print(rag_chain.invoke("我的笔记里提到了哪些学习计划？"))
```

#### 4.9.3 理解 RAG 的几个关键参数

| 参数 | 作用 | 太小会怎样 | 太大会怎样 |
|---|---|---|---|
| `chunk_size` | 每段文字的长度 | 一段话的上下文不完整 | 一段里混入太多无关内容 |
| `chunk_overlap` | 段与段之间的重叠 | 句子可能被切断 | 重复内容多，浪费资源 |
| `k` | 每次检索返回几段 | 资料可能不够 | 无关内容变多，干扰模型 |

**调试技巧**：RAG 答错时，**先别怪模型**。先单独运行 `retriever.invoke("你的问题")`，看检索回来的内容是否真的相关。检索不对，就调整切分参数或换 Embedding 模型。

---

### 4.10 LangGraph 简介（进阶）

**核心思想**：当流程变复杂（需要分支判断、循环重试、人工审批、多个智能体协作）时，用"**状态图**"来描述会更清晰。
- **State（状态）**：一份在各步骤之间传递的"共享笔记本"。
- **Node（节点）**：一个处理步骤，接收状态、返回更新。
- **Edge（边）**：节点之间的连线，决定下一步去哪。

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 定义状态：这份"笔记本"里有一个 text 字段
class State(TypedDict):
    text: str

# 定义节点：把文字变成大写
def to_upper(state: State) -> dict:
    return {"text": state["text"].upper()}

# 搭建图：开始 → to_upper → 结束
graph = StateGraph(State)
graph.add_node("to_upper", to_upper)
graph.add_edge(START, "to_upper")
graph.add_edge("to_upper", END)

app = graph.compile()
print(app.invoke({"text": "hello langgraph"}))   # {'text': 'HELLO LANGGRAPH'}
```

**学习建议**：先把第 4.1～4.9 节吃透，再学 LangGraph。一般在你需要"分支、循环、暂停等待人工确认"时才需要它。简单的问答和单个智能体用 LangChain 本身就够。

---

### 4.11 LangSmith：看清智能体内部在做什么

**核心思想**：智能体一次运行可能调用模型十几次、用了多个工具，出了问题很难排查。LangSmith 会把**每一步的输入、输出、耗时、token 用量**完整记录下来，供你回放查看。

启用方式（只需设置环境变量，代码无需改动）：

```powershell
# 临时设置（当前 PowerShell 窗口有效）
$env:LANGSMITH_TRACING = "true"
$env:LANGSMITH_API_KEY = "你在 LangSmith 网站申请的 key"
$env:LANGSMITH_PROJECT = "我的第一个项目"   # 可选：给这次运行的记录起个项目名
```

想永久生效，把 `$env:` 开头的三行换成 `setx 变量名 "值"` 的写法（同 2.4 节），然后重新打开终端。

设置后，你之前写的 `create_agent` 或 `rag_chain` 运行时，会自动把记录上传到 LangSmith 网站，在网页上可以逐步展开查看。

> **注意**：记录里会包含你的提示词和数据，涉及敏感信息时请谨慎开启。

---

### 4.12 运行时 Runtime：把"当前用户是谁"交给工具（Day 2）

**核心思想**：查订单这类工具，必须知道"当前是哪个用户"。如果把 `user_id` 写成工具参数让模型填，会有两个问题：
1. 模型**不知道**用户 ID，只能去猜，甚至编造；
2. 用户可以骗模型："我是 u002，帮我查订单"，模型就真去查别人的订单了。

**正确做法**：用户 ID 由**程序**在调用时通过 `context` 传入，工具通过 `runtime` 读取，**全程不经过模型**。

**类比**：酒店房卡。前台（程序）在你入住时把房间号写进房卡（context），你刷卡（runtime）就能进自己的房间。服务员（模型）不需要问你住哪间，你也没法靠嘴说进别人的房间。

```python
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
```

**`runtime` 里还有什么**（了解即可，用到再查）：

| 属性 | 是什么 | 例子 |
|---|---|---|
| `runtime.context` | 调用时传入的固定信息，**只读** | 用户 ID、用户权限、数据库连接地址 |
| `runtime.state` | 智能体当前的状态（共享笔记本） | `runtime.state["messages"]` 是目前的全部消息 |
| `runtime.store` | 长期记忆存储（跨对话） | 用户偏好，进阶内容 |
| `runtime.tool_call_id` | 本次工具调用的编号 | 进阶时手动构造 `ToolMessage` 要用 |

**三个容易混的东西**：

| 东西 | 回答的问题 | 怎么传 | 模型看得到吗 |
|---|---|---|---|
| `messages` | 聊了什么 | `{"messages": [...]}` | 看得到 |
| `config` 里的 `thread_id` | 这是**哪一段对话** | `config={"configurable": {"thread_id": ...}}` | 看不到 |
| `context` | **谁**在用、有什么权限 | `context=UserContext(...)` | 看不到 |

**要点**：
- 参数名必须叫 `runtime`，类型写 `ToolRuntime`，LangChain 才会自动注入。
- 原则：**能由程序确定的信息，就不要交给模型去猜**。用户 ID、权限、当前时间都属于这一类。
- 中间件（4.13 节）里也能通过 `runtime.context` 读到同一份上下文。

---

### 4.13 中间件 Middleware：在智能体循环里插"检查站"（Day 2 概念，Day 7 实操）

**核心思想**：智能体的工作是一个循环："调用模型 → 调用工具 → 再调用模型……"。中间件可以**插在这个循环的固定位置**，做一些和业务无关、但每次都要做的事：记日志、限制调用次数、出错重试、敏感信息打码、危险操作前请人确认。

**类比**：机场的安检口。旅客（数据）每次经过都要过一遍，安检员不关心你去哪儿，只负责检查。想加一道检查，就多设一个安检口，不用改登机流程本身。

**中间件能插在哪里**（这些位置叫"钩子 Hook"）：

```
用户输入
   ↓
before_agent        ← 整个任务开始前（只执行一次）
   ↓
┌→ before_model     ← 每次调用模型之前：记日志、裁剪过长的消息
│  wrap_model_call  ← "包住"模型调用：失败重试、换备用模型、动态改提示词
│  after_model      ← 模型返回之后：检查输出、危险操作前请人审批
│  wrap_tool_call   ← "包住"工具调用：记录参数、工具报错时兜底
└─ 还要调工具？是 → 回到 before_model
   ↓ 否
after_agent         ← 整个任务结束后（只执行一次）
   ↓
最终回答
```

**LangChain 自带的常用中间件**（直接拿来用，不用自己写）：

| 中间件 | 作用 | 比喻 |
|---|---|---|
| `SummarizationMiddleware` | 对话太长时自动把旧消息压缩成摘要 | 把厚厚的会议记录整理成纪要 |
| `HumanInTheLoopMiddleware` | 调用指定工具前暂停，等人确认 | 大额转账前要经理签字 |
| `ModelCallLimitMiddleware` | 限制模型调用次数 | 给智能体设"最多想 N 次" |
| `ToolCallLimitMiddleware` | 限制工具调用次数 | 防止反复调用付费接口 |
| `ModelRetryMiddleware` / `ModelFallbackMiddleware` | 模型调用失败时重试 / 换备用模型 | 打不通电话就重拨 / 换个号码 |
| `PIIMiddleware` | 识别并处理手机号、邮箱等敏感信息 | 自动打码 |

**一个最小例子**（Day 2 看懂即可，Day 7 再深入）：

```python
from typing import Any
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, after_model, ModelCallLimitMiddleware
from langchain.tools import tool
from langgraph.runtime import Runtime

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
```

运行后你会看到日志一行一行打印出来，**正好对应智能体循环的每一步**。这是理解"智能体到底在干什么"的好办法。

**要点**：
- 中间件通过 `create_agent(middleware=[...])` 传入，按列表顺序执行。
- 中间件函数返回 `None` 表示不修改；返回字典表示更新状态（进阶用法，Day 7 再学）。
- 中间件和工具的区别：**工具是给模型用的**（模型决定调不调）；**中间件是给开发者用的**（每次必定执行，模型完全不知道它的存在）。

---

### 4.14 MCP：给 AI 应用接入外部能力的"统一插座"（Day 3 上午）

**核心思想**：4.6 节的 `@tool` 是你在**自己程序里**写的工具，只有你的程序能用。MCP（Model Context Protocol，模型上下文协议）是一个**公开标准**：别人按标准写好一个"MCP Server"，任何支持 MCP 的应用（LangChain、Claude Desktop、Cursor 等）都能直接接入它的工具，不用重写。

**类比**：插座和电器。国家规定了统一的插口标准（MCP 协议），电器厂（MCP Server）按标准做插头，你家墙上装好插座（MCP Client），任何品牌的电器插上就能用。

| 角色 | 是什么 | 类比 | 本节的例子 |
|---|---|---|---|
| **MCP Server** | 提供工具的程序 | 电器 | `time_server.py`：提供"查当前时间"、"加法"两个工具 |
| **MCP Client** | 接入并使用工具的一方 | 插座 | LangChain 里的 `MCPAdapter` |
| **传输方式（Transport）** | 两者怎么通信 | 电线 | `stdio`（本机启动子程序）或 `http`（访问网址） |

**安装**（在 `0RAG` 根目录执行）：

```powershell
uv add "langchain[mcp]"
```

> 加引号是因为 PowerShell 里 `[` `]` 有特殊含义。这会同时装上 `fastmcp`（写 MCP Server 的库）。

**第一步：写一个 MCP Server**（保存为 `time_server.py`）

```python
from datetime import datetime
from fastmcp import FastMCP

mcp = FastMCP("时间服务")

# 和 @tool 很像：类型标注 + docstring 会成为工具说明
@mcp.tool
def get_current_time() -> str:
    """获取当前的日期和时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@mcp.tool
def add(a: int, b: int) -> int:
    """计算两个整数的和。"""
    return a + b

if __name__ == "__main__":
    mcp.run()   # 默认使用 stdio：由客户端把本脚本作为子程序启动，通过输入输出通信
```

> 这个文件**不需要你手动运行**，下一步的客户端会自动启动它。

**第二步：写客户端，把 MCP 工具交给智能体**（保存为 `mcp_client.py`，和 `time_server.py` 放在同一个文件夹）

```python
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter

load_dotenv()

# 用"本文件所在文件夹 + 文件名"拼出绝对路径，避免运行目录不对时找不到文件
SERVER = Path(__file__).parent / "time_server.py"

async def main():
    # async with：连接 MCP Server，代码块结束时自动断开
    async with MCPAdapter(SERVER) as adapter:
        tools = await adapter.list_tools()   # 问 Server："你有哪些工具？"
        print("从 MCP Server 拿到的工具：", [t.name for t in tools])

        # 拿到的就是普通的 LangChain 工具，用法和 4.7 节完全一样
        agent = create_agent("openai:gpt-5-nano", tools)
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "现在几点了？顺便算一下 123 加 456"}]}
        )
        print(result["messages"][-1].content)

asyncio.run(main())
```

```powershell
uv run python mcp_client.py
```

**为什么这里出现了 `async` / `await`？** MCP 要和另一个程序通信，需要等待对方回复。`async`（异步）就像**点外卖**：下单后不用站在门口干等，可以先做别的事，外卖到了再处理。现阶段你只需要记住三条规则：
1. 用到 MCP 的代码写在 `async def main():` 里；
2. 调用前面加 `await`，智能体用 `ainvoke`（异步版的 `invoke`）；
3. 最后一行用 `asyncio.run(main())` 启动。

**要点**：
- 运行时会看到一条 `LangChainBetaWarning` 警告，说明 `langchain.mcp` 还是测试版，**属于正常现象，不是报错**。还会看到一个写着 `FastMCP` 的方框横幅和 `Starting MCP server` 日志，这是 Server 被自动启动时打印的，也属于正常现象。
- **过时写法提醒**：网上教程大多用 `langchain-mcp-adapters` 包里的 `MultiServerMCPClient` 和 `get_tools()`。这是 LangChain 1.4 之前的写法，官方已提供迁移指南，新代码用 `MCPAdapter` 和 `list_tools()`。
- 除了本地脚本，`MCPAdapter` 也能接入网上的 MCP 服务，参数换成网址字符串即可，如 `MCPAdapter("https://docs.langchain.com/mcp")`（LangChain 官方文档的 MCP 服务，不需要 Key）。
- MCP 和 `@tool` 怎么选：只给自己的程序用 → `@tool`；想让多个应用共用、或者要接入别人现成的服务 → MCP。

---

### 4.15 Day 2 综合任务思路：带工具、有记忆的对话脚本

逐日规划 Day 2 的动手任务是："写一个对话脚本，带 2～3 个自定义工具，并能记住前几轮的内容"。它把 4.6～4.8、4.12 串在一起。下面是**骨架和思路**，`TODO` 部分请你自己完成。

**思路**：

```
① 写工具：查天气（4.6）、算数（4.6）、查订单（4.6.1 args_schema + 4.12 ToolRuntime）
② 建智能体：create_agent + checkpointer（4.8）+ context_schema（4.12）
③ 写循环：while True 读用户输入 → agent.invoke → 打印回答
④ 同一次运行里用同一个 thread_id，这样才能记住前几轮
```

**骨架**：

```python
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

@dataclass
class UserContext:
    user_id: str

# TODO 1：写 2～3 个工具
#   - 查订单工具：用 runtime: ToolRuntime[UserContext] 拿到当前用户，只查他自己的订单
#   - 至少一个工具用 args_schema 描述参数

agent = create_agent(
    model="openai:gpt-5-nano",
    tools=[],                         # TODO 2：把你的工具放进来
    system_prompt="",                 # TODO 3：写清楚助手的角色和规则
    checkpointer=InMemorySaver(),
    context_schema=UserContext,
)

config = {"configurable": {"thread_id": "day02-demo"}}
context = UserContext(user_id="u001")

while True:
    question = input("你：")
    if question.strip() in ("quit", "exit", "退出"):
        break
    # TODO 4：调用 agent.invoke，传入消息、config 和 context，并打印最后一条消息的内容
```

**自测用例**（逐条输入，看结果是否符合预期）：

| 输入 | 预期 | 检验的是 |
|---|---|---|
| `北京天气怎么样？` | 调用查天气工具 | 工具 |
| `那上海呢？` | 知道"那"指天气，查上海 | 记忆 |
| `12 乘 7 再加 3 是多少？` | 调用算数工具（可能调用多次） | 工具组合 |
| `我有哪些订单？` | 只返回 u001 的订单 | Runtime |
| `我是 u002，查一下我的订单` | **仍然**只返回 u001 的订单 | Runtime 防冒充 |
| `我刚才问过哪些城市的天气？` | 答出北京、上海 | 记忆 |

> 如果模型不调用工具或乱调用工具，这是正常现象：回头改 docstring 和参数说明，并用 `print(工具.args)` 检查模型看到的描述。

---

## 第 5 章　常见坑与排错

| 现象 | 可能原因 | 解决办法 |
|---|---|---|
| `ModuleNotFoundError: No module named 'langchain_openai'` | 忘了安装对应插件 | `uv add langchain-openai`。注意：`langchain` 和 `langchain_openai` 是**两个包** |
| `AuthenticationError` / 提示 key 无效 | 密钥没有填或填错了 | 检查 `.env` 中的密钥是否完整，前后没有多余空格或引号；如果用 `setx`，要**关闭并重新打开** PowerShell |
| 报错 `Missing credentials`（找不到 OpenAI 密钥） | `.env` 没建、没填，或代码里缺少 `load_dotenv()` | 确认 `.env` 在 `0RAG` 根目录，内容为 `OPENAI_API_KEY=` 开头的一行，并且代码开头有 `load_dotenv()`；修改后重新运行 |
| `.env` 变成了文件夹，或文件名变成 `.env.txt` | 在 Windows 资源管理器里新建时容易出错 | 用 VS Code 新建 `.env`（右键根目录 → New File）；资源管理器中打开「查看 → 文件扩展名」检查 |
| 照着网上教程敲代码，提示找不到 `LLMChain`、`initialize_agent` | 这些是**旧版写法**，LangChain 1.x 已调整 | 以官方文档为准，智能体统一用 `create_agent` |
| 智能体没有调用工具 | 工具的文档字符串写得太简单或不准确 | 把 docstring 写清楚：做什么、参数是什么、什么时候用 |
| RAG 回答不相关 | 检索到的内容不对 | 先单独运行 `retriever.invoke(问题)` 检查检索结果；调整 `chunk_size`、`k` |
| 多轮对话"失忆" | 没传 `checkpointer`，或每次 `thread_id` 都不同 | 确保创建智能体时传入了 checkpointer，并且同一对话使用同一个 `thread_id` |
| 费用增长很快 | 开发时用了大模型，或输出过长 | 开发阶段用小模型（如 nano/mini 系列）；设置 `max_tokens`；用 LangSmith 查看 token 消耗 |
| 程序重启后记忆消失 | `InMemorySaver` 只存在内存中 | 正式项目换成持久化的 checkpointer |
| 代码能跑但结果每次不一样 | `temperature` 过高 | 需要稳定结果时设为 `0` |
| `'python' 不是内部或外部命令`，或弹出 Microsoft Store | 安装 Python 时没勾选 "Add python.exe to PATH" | 重新运行 Python 安装包，选择"修改"并勾选 PATH；也可以试试 `py --version` |
| `uv` 无法识别（提示「不是内部或外部命令」或「无法识别」） | uv 所在文件夹没有加入 Windows 用户 PATH | 把 uv 所在文件夹（如 `%USERPROFILE%\.local\bin`）加入用户环境变量 `Path`，然后关闭并重新打开 VS Code / PowerShell |
| 中文输出变成乱码（如 `鍒╃敤`） | Windows 终端编码不是 UTF-8 | 运行前执行 `$env:PYTHONUTF8 = "1"`；cmd 中可先执行 `chcp 65001` |
| `UnicodeDecodeError`，读取 notes.txt 时报错 | 文件不是 UTF-8 编码 | 用记事本"另存为"时编码选 **UTF-8**；本指南代码已写 `encoding="utf-8"` |
| `FileNotFoundError: notes.txt` | 运行目录不对，或文件不在同一文件夹 | 确认 `notes.txt` 与 `.py` 文件在同一文件夹，并在该文件夹中运行 `python xxx.py` |
| 运行后提示 `ModuleNotFoundError`，但明明装过包 | 用 `python 文件名.py` 直接运行，没有使用项目环境 | 改用 `uv run python 文件名.py`；新包用 `uv add 包名` 安装 |
| 工具里 `runtime.context` 是 `None`，或报 `AttributeError` | 创建智能体时没传 `context_schema`，或调用时没传 `context=` | 两处都要写：`create_agent(..., context_schema=...)` 和 `agent.invoke(..., context=...)` |
| 模型把 `runtime` 当成普通参数，要求用户提供 | 参数名不是 `runtime`，或类型没写 `ToolRuntime` | 参数名必须是 `runtime`，类型写 `ToolRuntime[你的上下文类]` |
| 智能体一直调用工具停不下来 | 在智能体里设置了 `tool_choice="any"`；或工具返回的内容让模型以为没完成 | 智能体里不要强制 `tool_choice`；加 `ModelCallLimitMiddleware(run_limit=...)` 兜底 |
| 加了 `tool_choice` 后报错 | 当前模型或第三方接口不支持该参数 | 去掉 `tool_choice` 再试；换支持的模型 |
| `uv add langchain[mcp]` 报错或没装上 | PowerShell 把方括号当成了特殊符号 | 加引号：`uv add "langchain[mcp]"` |
| `ImportError: Please install FastMCP` | 没装 MCP 相关依赖 | `uv add "langchain[mcp]"` |
| 运行 MCP 示例时出现 `LangChainBetaWarning` | `langchain.mcp` 是测试版 | 正常现象，不影响运行 |
| `RuntimeError: asyncio.run() cannot be called from a running event loop` | 在 Jupyter 里运行了 `asyncio.run` | 在 PowerShell 用 `uv run python 文件名.py` 运行；Jupyter 中直接写 `await main()` |
| MCP 客户端报找不到 `time_server.py` | 相对路径和运行目录不一致 | 用 `Path(__file__).parent / "time_server.py"` 拼绝对路径 |

**排错三步法**：
1. **看报错的最后几行**：通常最关键的信息在最后。
2. **拆开看**：把链拆成单个步骤，逐个用 `invoke` 测试，找出是哪一步出错。
3. **查官方文档或开 LangSmith 看记录**。

---

## 第 6 章　学习路线（四周计划）

| 周次 | 学习内容 | 阶段目标（检验标准） |
|---|---|---|
| **第 1 周** | 环境准备、模型、消息、提示词模板、LCEL 管道（第 2～4.4 节） | 能写一个"翻译/润色"小工具，支持输入任意文字 |
| **第 2 周** | 结构化输出、工具、智能体（第 4.5～4.7 节） | 能做一个会查天气、会算数的智能助手 |
| **第 3 周** | 记忆、RAG（第 4.8～4.9 节） | 能用你自己的笔记做问答机器人 |
| **第 4 周** | LangGraph、LangSmith、排错与优化（第 4.10～4.11 节） | 能画一个带分支的流程，并能用 LangSmith 查看运行记录 |

**学习心法**：
- **边敲边学**：不要只看不敲，每个代码块都亲手运行一遍。
- **先跑通，再理解**：遇到不懂的概念先跳过，等用起来了再回头理解。
- **改参数实验**：改一改 `temperature`、`chunk_size`、提示词，观察结果变化，这是最快的学习方式。

> 如果你正在使用本仓库的 `2practice/day01` ～ `day10` 目录练习，可以把上面四周的内容对应地放进去，每天一个小练习，记录在对应文件夹里。

---

## 第 7 章　练习项目

按难度从低到高，建议依次完成。

### 项目 1：多语言翻译器（对应第 1～2 周）
- **功能**：用户输入一句中文和目标语言，输出翻译结果。
- **涉及知识点**：提示词模板、LCEL 管道、`StrOutputParser`。
- **进阶挑战**：加入 `stream` 实现打字机效果。

### 项目 2：带工具的旅行助手（对应第 2 周）
- **功能**：用户问"去北京旅游要带什么衣服？"，智能体自动查询天气，并给出建议。
- **涉及知识点**：`@tool`、`create_agent`、system_prompt。
- **进阶挑战**：再加一个"查询景点门票"的工具，观察智能体如何组合多个工具。

### 项目 3：个人知识库问答（对应第 3 周）
- **功能**：把你的学习笔记（txt 或 PDF）导入，然后可以提问"我上周学了什么？"。
- **涉及知识点**：文档加载、切分、向量化、检索器、RAG 链。
- **进阶挑战**：加入多轮对话记忆，支持追问。

### 项目 4：带人工审批的文章写作流程（对应第 4 周）
- **功能**：智能体写草稿 → 检查草稿质量 → 质量不合格则返工 → 合格后等待人工确认再发布。
- **涉及知识点**：LangGraph 的节点、边、条件分支与循环。
- **进阶挑战**：用 LangSmith 查看每次返工的记录。

---

## 第 8 章　学习资源

| 资源 | 说明 | 地址 |
|---|---|---|
| **LangChain 官方文档** | 最权威的资料，内容会随版本更新 | https://docs.langchain.com/oss/python/langchain/overview |
| **快速入门（Quickstart）** | 官方的第一个示例，建议先读 | https://docs.langchain.com/oss/python/langchain/quickstart |
| **智能体（Agents）章节** | 深入理解 `create_agent` | https://docs.langchain.com/oss/python/langchain/agents |
| **工具（Tools）章节** | 深入理解 `@tool` 与工具设计 | https://docs.langchain.com/oss/python/langchain/tools |
| **模型（Models）章节** | 各家模型的接入方式 | https://docs.langchain.com/oss/python/langchain/models |
| **消息（Messages）章节** | 消息类型的详细说明 | https://docs.langchain.com/oss/python/langchain/messages |
| **运行时（Runtime）章节** | `context_schema`、`ToolRuntime` 的详细说明 | https://docs.langchain.com/oss/python/langchain/runtime |
| **中间件总览** | 中间件能做什么、怎么组合 | https://docs.langchain.com/oss/python/langchain/middleware/overview |
| **自带中间件列表** | 所有内置中间件及参数 | https://docs.langchain.com/oss/python/langchain/middleware/built-in |
| **自定义中间件** | `before_model`、`wrap_tool_call` 等写法 | https://docs.langchain.com/oss/python/langchain/middleware/custom |
| **MCP 章节** | `MCPAdapter` 的用法 | https://docs.langchain.com/oss/python/langchain/mcp |
| **MCP 旧写法迁移指南** | 从 `MultiServerMCPClient` 迁移到 `MCPAdapter` | https://docs.langchain.com/oss/python/migrate/langchain-mcp-adapters |
| **FastMCP 文档** | 写 MCP Server 的库 | https://gofastmcp.com |
| **知识库（Knowledge Base）教程** | 手把手的语义搜索与 RAG 教程 | https://docs.langchain.com/oss/python/langchain/knowledge-base |
| **LangChain GitHub** | 源代码与问题讨论 | https://github.com/langchain-ai/langchain |
| **LangSmith** | 观测与调试平台 | 在官方文档中搜索 LangSmith |

**查资料的小技巧**：
- 遇到不认识的类或函数，直接在官方文档搜索其名字。
- 遇到报错，把**报错信息的最后一行**拿去搜索，通常能找到解决方案。
- 网上旧教程很多，**发布时间较早的文章可能已过时**，优先看官方文档。

---

## 第 9 章　小结与自测清单

### 9.1 全文一图回顾

```
        你的问题 / 文档
              │
              ▼
   ┌──────── LangChain ────────┐
   │  Prompt 模板 → 模型 → 解析  │  ← LCEL 管道 (|)
   │        ↑         ↑        │
   │      工具 Tools  记忆 Memory │  ← 给模型"手和笔记本"
   │        ↑                  │
   │   检索器 Retriever (RAG)   │  ← 给模型"资料库"
   │        ↑                  │
   │  智能体 Agent 自动决策循环   │  ← 把以上组合起来
   └───────────────────────────┘
              │
              ▼
      复杂流程 → LangGraph
      运行观测 → LangSmith
```

### 9.2 自测清单

学完后，对照下面的问题检查自己是否掌握：

- [ ] 我能用一句话说清楚 LangChain、LangGraph、LangSmith 各自是干什么的
- [ ] 我能解释 `init_chat_model` 的作用，并能切换到另一家模型
- [ ] 我能说出 `SystemMessage`、`HumanMessage`、`AIMessage` 的区别
- [ ] 我能写一个带变量的提示词模板，并用 `invoke` 填入内容
- [ ] 我能读懂 `prompt | model | parser` 的执行顺序
- [ ] 我能用 `@tool` 写一个工具，并解释为什么 docstring 很重要
- [ ] 我能用 `create_agent` 创建一个带工具的智能体
- [ ] 我能解释 `thread_id` 和 `checkpointer` 如何实现多轮记忆
- [ ] 我能说清"模型只决定调用哪个工具、传什么参数，真正执行函数的是程序"，并能看懂 `tool_calls`
- [ ] 我知道 `tool_choice` 的三种取值，以及为什么智能体里不要强制调用
- [ ] 我能用 `args_schema` 给工具写参数说明书
- [ ] 我能用 `context_schema` + `ToolRuntime` 把用户 ID 传给工具，并解释为什么不让模型来填
- [ ] 我能说出中间件能插在智能体循环的哪些位置，并举出 2 个用途
- [ ] 我能用一句话说清 MCP Server 和 MCP Client 的关系，并跑通一个 MCP 示例
- [ ] 我能画出 RAG 的六个步骤（加载、切分、向量化、入库、检索、生成）
- [ ] 我知道 RAG 答错时应该先检查"检索结果"，而不是直接怪模型
- [ ] 我知道不能把 API Key 写进代码或提交到 git

### 9.3 术语补充：一句话原则

> **先用 `invoke` 跑通，再用 `|` 组合，再用工具和记忆增强，最后用 RAG 接入你的资料，复杂流程交给 LangGraph。**

---

*本文档为入门学习用途。代码以运行时的官方文档版本为准，遇到接口变化请查阅官方文档。*
