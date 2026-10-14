# RAG 入门完全指南（面向初学者 · 配合逐日规划 Day 3 下午～Day 5）

> **适用版本**：LangChain 1.x（Python）、langchain-milvus 0.4.x、pymilvus 3.x、Milvus 2.5（Docker 版）
> **更新时间**：2026 年 10 月
> **官方文档**：LangChain 知识库教程 https://docs.langchain.com/oss/python/langchain/knowledge-base ；Milvus 文档 https://milvus.io/docs
> **前置知识**：已学完《LangChain入门指南.md》的 4.1～4.8 节（模型、消息、提示词模板、LCEL、结构化输出、工具、智能体、记忆）。
> **阅读建议**：先看第 1 章的「逐日对照表」，每天只看当天对应的小节；第 4 章的代码逐段敲一遍；Day 5 下午按第 7 章项目 4 做成品。

---

## 目录

- [第 0 章　先搞懂：RAG 到底是什么？](#第-0-章先搞懂rag-到底是什么)
- [第 1 章　学习地图与逐日对照](#第-1-章学习地图与逐日对照)
- [第 2 章　环境准备（Windows）](#第-2-章环境准备windows)
- [第 3 章　术语速查表](#第-3-章术语速查表)
- [第 4 章　知识点详解（含代码）](#第-4-章知识点详解含代码)
- [第 5 章　常见坑与排错](#第-5-章常见坑与排错)
- [第 6 章　学习路线（三天计划）](#第-6-章学习路线三天计划)
- [第 7 章　练习项目](#第-7-章练习项目)
- [第 8 章　学习资源](#第-8-章学习资源)
- [第 9 章　小结与自测清单](#第-9-章小结与自测清单)

---

## 第 0 章　先搞懂：RAG 到底是什么？

### 0.1 一个比喻：开卷考试

大模型像一个**学识渊博、但没看过你们公司任何内部资料的考生**。问它"我们公司年假几天？"，它要么说不知道，要么一本正经地编一个（这叫**幻觉**）。

**RAG（Retrieval-Augmented Generation，检索增强生成）就是让它开卷考试**：

1. 考前：把公司资料整理好，做成一本**带索引的参考书**（知识库）；
2. 考试时：先根据题目**翻书找到相关的几页**（检索）；
3. 再把这几页和题目一起交给考生，让它**照着书答题**（生成）。

### 0.2 它解决了什么问题？

| 痛点 | 不用 RAG | 用 RAG |
|---|---|---|
| 模型不知道内部资料 | 答不出，或者编造 | 先查资料再回答 |
| 资料经常更新 | 要重新训练模型（贵、慢） | 只需更新知识库里的文档 |
| 不知道答案对不对 | 无从核实 | 回答附带**来源**（哪个文件第几页），可以核对 |
| 资料太多，塞不进一次对话 | 超出模型的输入长度 | 每次只取最相关的几段 |
| 资料里没有的问题 | 模型照样瞎答 | 可以要求它回答"资料中未找到" |

### 0.3 RAG 用到哪些"零件"？

| 零件 | 干什么 | 本指南用的 | 类比 |
|---|---|---|---|
| **LangChain** | 把所有零件串起来 | `langchain`、`langchain-community` | 总装车间 |
| **文档加载器** | 把 TXT/CSV/JSON/PDF 读进来 | `TextLoader` 等 | 扫描仪 |
| **切分器** | 把长文档切成小段 | `RecursiveCharacterTextSplitter` | 裁纸刀 |
| **Embedding 模型** | 把文字变成一串数字（向量） | Ollama 本地的 `bge-m3` | 给每段文字算 GPS 坐标 |
| **向量库** | 存向量，快速找"最接近"的 | **Milvus**（Docker 版） | 带坐标索引的仓库 |
| **大模型** | 根据找到的资料组织回答 | 你在 Day 1～2 用的模型 | 答题的考生 |

### 0.4 学之前你需要会什么

- [ ] 会用 `init_chat_model` 调用模型（LangChain 指南 4.1）
- [ ] 会写 `ChatPromptTemplate` 和 `prompt | model | parser` 管道（4.3、4.4）
- [ ] 会用 `with_structured_output`（4.5，Day 5 的查询改写会用到）
- [ ] 会用 `@tool` 和 `create_agent`（4.6、4.7，Day 5 的 Agentic RAG 会用到）
- [ ] Day 0 已装好 Docker Desktop，`docker run hello-world` 能成功

---

## 第 1 章　学习地图与逐日对照

### 1.1 知识地图

```
┌───────────────────────────────────────────────────────────────────┐
│ 【第 4 层】评估与进阶    10 题打分表 · 查询改写 · 重排序 · Agentic RAG     │  Day 5
├───────────────────────────────────────────────────────────────────┤
│ 【第 3 层】检索与生成    Retriever · top-k · MMR · 带来源 · 拒答           │  Day 5
├───────────────────────────────────────────────────────────────────┤
│ 【第 2 层】向量化与入库  Embedding · 向量库 · Milvus 集合/字段/索引        │  Day 4
├───────────────────────────────────────────────────────────────────┤
│ 【第 1 层】文档预处理    Document · 4 种加载器 · 切分 chunk_size/overlap  │  Day 3～4
├───────────────────────────────────────────────────────────────────┤
│ 【第 0 层】全链路认知    加载 → 切分 → 向量化 → 入库 → 检索 → 生成        │  Day 3
└───────────────────────────────────────────────────────────────────┘
```

| 学完这一层 | 你就能做出 |
|---|---|
| 第 0 层 | 画出 RAG 流程图，讲清每一步的输入和输出 |
| 第 1 层 | 把 4 种格式的文档读进来，切成大小合适的片段 |
| 第 2 层 | 把片段变成向量存进 Milvus，用一个问题查出最相关的 3 段 |
| 第 3 层 | 一个能回答问题、标注来源、资料里没有就拒答的问答程序 |
| 第 4 层 | 知道它答得好不好，并能针对性地改进 |

### 1.2 逐日对照表（配合逐日规划使用）

| 逐日规划 | 学习要点 | 看本指南哪一节 | 动手任务对应 |
|---|---|---|---|
| **Day 3 上午** | MCP 入门 | 不在本指南，看《LangChain入门指南》**4.14** | 跑通 MCP 示例 |
| **Day 3 下午** | RAG 全链路、Document Loader | **4.1** 全链路 · **4.2** Document · **4.3** 加载器 | 4 种格式文档加载，打印长度和元数据（项目 1） |
| **Day 3 晚上** | 手绘 RAG 流程图 | **4.1** 的流程表 | 标出每一步的输入和输出 |
| **Day 4 上午** | Text Splitter | **4.4** 切分 | 两种切分方式对比（项目 2） |
| **Day 4 下午** | Embedding、Vector Store、Milvus | **4.5** Embedding · **4.6** 向量库 · **4.7** Milvus 基础 | 建集合、写入、相似度检索 |
| **Day 4 晚上** | 全部文档入库 | **4.8** 用 LangChain 写入 Milvus | 记录每份文档的块数（项目 3） |
| **Day 5 上午** | Retrievers、RAG 多种模式 | **4.9** 检索器 · **4.11** 进阶模式 | 选 1～2 种模式动手 |
| **Day 5 下午～晚上** | RAG 企业问答成品、评估 | **4.10** 带来源和拒答 · **4.12** 评估 | 成品（项目 4） |

---

## 第 2 章　环境准备（Windows）

> 本章所有命令都在 **PowerShell** 中执行，默认已经 `cd` 到 `0RAG` 根目录。Python、uv、`.env` 的设置和 LangChain 指南第 2 章相同，这里不重复。

### 2.1 安装本阶段要用的包

你的 `pyproject.toml` 里已经有 `langchain-community`、`langchain-text-splitters`、`langchain-ollama`、`pymilvus`，还需要补 3 个：

```powershell
uv add pypdf jq langchain-milvus
```

| 包 | 干什么 |
|---|---|
| `pypdf` | `PyPDFLoader` 读 PDF 要用 |
| `jq` | `JSONLoader` 解析 JSON 要用（已有 Windows 安装包，可以直接装） |
| `langchain-milvus` | 让 LangChain 能连接 Milvus |

> **为什么不装 Milvus Lite？** 逐日规划 Day 4 列了"方案 B：Milvus Lite"，但它**不支持 Windows**（`langchain-milvus` 在 Windows 上会自动跳过安装它）。规划里也写了方案 B 必须在 WSL2 里运行。你在 Windows 上直接用**方案 A：Docker 版 Milvus**，你仓库里的 `milvus-check` 文件夹就是现成的配置。

### 2.2 准备 Embedding 模型（需要你决定）

Embedding 模型负责把文字变成向量。下面两个方案二选一：

| | **方案 A：Ollama 本地 `bge-m3`（推荐）** | 方案 B：OpenAI 兼容接口 |
|---|---|---|
| 做法 | `ollama pull bge-m3`，代码里用 `OllamaEmbeddings` | 代码里用 `OpenAIEmbeddings`，走你 `.env` 里的接口地址 |
| 费用 | 免费 | 按调用量计费 |
| 中文效果 | 好（专门支持中英文等多语言） | 好 |
| 向量维度 | **1024** | `text-embedding-3-small` 是 **1536** |
| 风险 | 第一次下载约 1.2 GB；要保证 Ollama 服务在运行 | 你用的第三方接口**不一定支持** Embedding，要先测试 |

**推荐方案 A**：你已经装好 Ollama，入库时要对每个文本块都调用一次 Embedding，本地运行免费，反复实验也不心疼。

**方案 A 的准备步骤**：

```powershell
# 下载 Embedding 模型（只需一次）
ollama pull bge-m3

# 确认下载成功：列表里应该能看到 bge-m3
ollama list
```

> **注意**：`qwen2.5:7b` 是**聊天模型**，不能拿来做 Embedding。Embedding 必须用专门的向量模型（如 `bge-m3`）。
>
> 如果 `ollama list` 显示为空，但你确定下载过模型，先回忆之前踩过的坑：`OLLAMA_MODELS` 环境变量指向了错误的目录，需要结束后台 `ollama` 进程后重启服务。

### 2.3 启动 Milvus（Docker 版）

1. 先打开 **Docker Desktop**，等左下角显示 Engine running（引擎运行中）。
2. 在 PowerShell 中执行：

```powershell
cd milvus-check
docker compose up -d     # -d：在后台启动
docker ps                # 应该看到 milvus-standalone、milvus-etcd、milvus-minio 三个容器
cd ..
```

3. 等 1～2 分钟（Milvus 启动比较慢），在浏览器打开 http://localhost:9091/healthz ，页面显示 `OK` 说明启动成功。

> **用完怎么关**：`cd milvus-check` 后执行 `docker compose stop`。数据保存在 `milvus-check/volumes` 里，下次 `docker compose up -d` 还在。**不要随手删 `volumes` 文件夹**，那等于清空数据库。
>
> **内存提醒**：Milvus 三个容器加起来大约占 2～4 GB 内存。电脑卡顿时，不学 RAG 的时候就把它关掉。

### 2.4 准备练习资料

本阶段三天共用一套资料。在 `2practice` 下新建 `data` 文件夹，放入下面 4 个文件。资料内容是一家虚构公司"星辰科技"，后面的示例和测试问题都基于它。

> **为什么放在 `2practice/data`，而不是放进每天的文件夹？** 因为 Day 3～5 都要用同一份资料。代码里用 `Path(__file__).resolve().parents[1] / "data"` 找到它，意思是"本文件所在文件夹（如 `day03`）的上一级里的 `data` 文件夹"。

**① `员工手册.txt`**（用 VS Code 新建，右下角确认编码是 UTF-8）

```
第一章 考勤制度
公司实行弹性工作制，核心工作时间为上午 10:00 至下午 4:00。员工每天需工作满 8 小时。
迟到超过 30 分钟视为旷工半天。每月允许 3 次 15 分钟以内的迟到，不扣款。

第二章 休假制度
员工入职满一年后享有 5 天带薪年假，每多工作一年增加 1 天，最多 15 天。
年假当年有效，未休完的可顺延到次年 3 月 31 日，逾期作废。
病假需提供医院证明，每年前 5 天病假工资照发。

第三章 报销制度
差旅报销需在出差结束后 15 个工作日内提交。
住宿标准：一线城市每晚不超过 500 元，其他城市每晚不超过 350 元。
餐补标准：每天 100 元，无需发票。
```

**② `产品价目.csv`**（用 VS Code 新建，**不要用 Excel 保存**，原因见第 5 章）

```
产品名称,型号,价格,保修期
星辰智能音箱,XC-S1,299,1年
星辰扫地机器人,XC-R2,1999,2年
星辰空气净化器,XC-A3,1299,2年
```

**③ `FAQ.json`**

```json
[
  {"question": "如何申请退货？", "answer": "收货后 7 天内可在 App 中申请无理由退货，商品需保持完好。"},
  {"question": "发票怎么开？", "answer": "订单完成后在 App 订单详情页点击“申请发票”，电子发票 24 小时内发送到邮箱。"},
  {"question": "客服工作时间？", "answer": "每天 9:00 至 21:00，节假日正常服务。"}
]
```

**④ `IT安全规范.pdf`**：用 Word 新建文档，写入下面三段，**每段之间插入一个分页符**（Word 菜单：插入 → 分页），然后"另存为"PDF。这样 PDF 有 3 页，方便后面练习"标注页码"。

```
第 1 页：公司电脑开机密码至少 8 位，必须包含字母和数字，每 90 天更换一次。
第 2 页：禁止在公司电脑上安装未经 IT 部门批准的软件。如需安装，请在 OA 系统提交软件申请单。
第 3 页：发现电脑中病毒或数据泄露，应立即断开网络，并拨打 IT 服务热线 8888。
```

### 2.5 验证环境是否正常

新建 `2practice/day03/check_env.py`：

```python
from pymilvus import MilvusClient
from langchain_ollama import OllamaEmbeddings

# 1. 检查 Milvus
client = MilvusClient(uri="http://localhost:19530")
print("Milvus 连接成功，现有集合：", client.list_collections())

# 2. 检查 Embedding 模型
embeddings = OllamaEmbeddings(model="bge-m3")
vector = embeddings.embed_query("你好")
print("Embedding 成功，向量维度：", len(vector))   # bge-m3 应为 1024
```

```powershell
uv run python 2practice/day03/check_env.py
```

两行都打印出来，环境就准备好了。

---

## 第 3 章　术语速查表

| 英文 | 中文 | 通俗理解 |
|---|---|---|
| **RAG** | 检索增强生成 | 开卷考试：先翻书找资料，再照着资料答题 |
| **Hallucination** | 幻觉 | 模型一本正经地编造不存在的内容 |
| **Document** | 文档对象 | LangChain 里的"一段文字 + 它的来源信息" |
| **page_content** | 正文 | Document 里的文字内容 |
| **metadata** | 元数据 | Document 的"标签"，如来源文件名、页码 |
| **Document Loader** | 文档加载器 | 把各种格式的文件读成 Document 的"扫描仪" |
| **Text Splitter** | 文本切分器 | 把长文档切成小段的"裁纸刀" |
| **Chunk** | 文本块 | 切出来的一小段文字 |
| **chunk_size** | 块大小 | 每块最多多少个字符 |
| **chunk_overlap** | 块重叠 | 相邻两块共用的字符数，防止一句话被切断后丢失上下文 |
| **Embedding** | 向量化 / 嵌入 | 把文字变成一串数字，意思越接近，数字越接近 |
| **Vector** | 向量 | 那一串数字，如 `[0.12, -0.03, ...]` |
| **Dimension** | 维度 | 向量里有几个数字，`bge-m3` 是 1024 个 |
| **Cosine Similarity** | 余弦相似度 | 衡量两个向量方向有多接近，越接近 1 越相似 |
| **Vector Store** | 向量库 | 存放向量、能快速找出"最相似"内容的数据库 |
| **Milvus** | （产品名） | 一款开源的专业向量数据库 |
| **Collection** | 集合 | Milvus 里的"一张表" |
| **Field** | 字段 | 集合里的"一列"，如正文、来源、向量 |
| **Schema** | 结构定义 | 集合有哪些字段、每个字段什么类型 |
| **Index** | 索引 | 让检索变快的"目录"，不建索引就要一条条比对 |
| **Retriever** | 检索器 | 输入问题，返回最相关的几个文本块 |
| **top-k** | 前 k 个 | 每次检索返回几个结果 |
| **MMR** | 最大边际相关性 | 检索时兼顾"相关"和"不重复"，避免返回好几段意思一样的内容 |
| **Query Rewriting** | 查询改写 | 先把用户的口语化问题改写成更适合检索的说法 |
| **Rerank** | 重排序 | 先粗略多找一些，再精细打分，挑出最相关的几个 |
| **Agentic RAG** | 智能体式 RAG | 把"检索"做成工具交给智能体，让它自己决定查不查、查几次 |

---

## 第 4 章　知识点详解（含代码）

> **约定**：
> - 代码文件放在当天的练习文件夹（如 `2practice/day03/`），用 `uv run python 文件路径` 运行。
> - 用到聊天模型的示例，文件最上面要加 `from dotenv import load_dotenv` 和 `load_dotenv()` 两行（同 LangChain 指南）。只用 Ollama 和 Milvus 的示例不需要。
> - 模型名统一写 `"openai:gpt-5-nano"`，换成你自己在用的模型名即可。

### 4.1 RAG 全链路：六步各解决什么问题（Day 3）

**核心思想**：RAG 分成两个阶段。**准备阶段**只做一次（或资料更新时再做）；**问答阶段**每次提问都做。

```
【准备阶段：建知识库】                                【问答阶段：每次提问】

 原始文件（TXT/CSV/JSON/PDF）                          用户问题
      │ ① 加载 Load                                        │ ⑤ 检索 Retrieve
      ▼                                                    │   （问题也要先向量化）
 Document 列表（正文 + 元数据）                            ▼
      │ ② 切分 Split                                  最相关的 k 个文本块
      ▼                                                    │ ⑥ 生成 Generate
 小文本块（chunk）列表                                     │   （文本块 + 问题 → 提示词 → 模型）
      │ ③ 向量化 Embed                                     ▼
      ▼                                               带来源的回答
 向量列表（每块一串数字）
      │ ④ 入库 Store
      ▼
 向量库（Milvus）  ─────────── ⑤ 检索时从这里查 ──────────┘
```

**Day 3 晚上手绘流程图就照这张表画，每一步标出输入和输出**：

| 步骤 | 输入 | 输出 | 解决什么问题 | 对应小节 |
|---|---|---|---|---|
| ① 加载 | 文件路径 | `Document` 列表 | 各种格式的文件统一成一种格式 | 4.2、4.3 |
| ② 切分 | `Document` 列表 | 更小的 `Document` 列表 | 整篇文档太长，塞不进提示词；而且检索要精确到段落 | 4.4 |
| ③ 向量化 | 文本 | 向量（一串数字） | 让电脑能计算"两段话意思有多接近" | 4.5 |
| ④ 入库 | 文本 + 向量 + 元数据 | 向量库里的记录 | 几万段文字也能在毫秒级找到最相近的 | 4.6～4.8 |
| ⑤ 检索 | 用户问题 | 最相关的 k 个文本块 | 从海量资料里只挑出和问题有关的几段 | 4.9 |
| ⑥ 生成 | 文本块 + 问题 | 回答（附来源） | 让模型基于资料、而不是凭记忆回答 | 4.10 |

**要点**：
- RAG 答错时，问题可能出在任何一步。排查顺序是**从前往后**：先看加载出来的文字对不对，再看切分，再看检索结果，最后才怀疑模型。
- 第 ③ 步和第 ⑤ 步**必须用同一个 Embedding 模型**。用 A 模型入库、用 B 模型查询，就像用北京的坐标系去查上海的地图。

---

### 4.2 Document：RAG 里流动的"标准零件"（Day 3）

**核心思想**：不管原始文件是什么格式，加载后都变成 `Document` 对象。它只有两个关键部分：

| 部分 | 内容 | 类比 |
|---|---|---|
| `page_content` | 文字内容 | 一张卡片的正面 |
| `metadata` | 字典，记录来源、页码等 | 卡片背面的标签：来自哪本书、第几页 |

```python
from langchain_core.documents import Document

doc = Document(
    page_content="员工入职满一年后享有 5 天带薪年假。",
    metadata={"source": "员工手册.txt", "chapter": "第二章"},
)

print(doc.page_content)
print(doc.metadata)
print(doc.metadata["source"])
```

**要点**：
- **metadata 是"标注来源"的基础**。Day 5 要求回答附带"文件名 + 页码"，靠的就是加载时保存下来的 metadata。
- 你也可以自己往 metadata 里加字段，如部门、更新日期，之后可以按这些字段过滤检索结果。

---

### 4.3 文档加载器：把 4 种格式读进来（Day 3）

**核心思想**：每种格式配一个"扫描仪"，统一输出 `Document` 列表。**不同加载器"切"文档的粒度不一样**，这是本节最需要注意的地方。

| 格式 | 加载器 | 一个 Document 对应 | metadata 里有什么 | 容易出的问题 |
|---|---|---|---|---|
| TXT | `TextLoader` | **整个文件** | `source` | 编码不是 UTF-8 时报错 |
| CSV | `CSVLoader` | **一行** | `source`、`row`（从 0 开始） | Excel 另存的 CSV 是 GBK 编码 |
| JSON | `JSONLoader` | 由 `jq_schema` 决定，通常是**一条记录** | `source`、`seq_num`（从 1 开始） | `jq_schema` 写错时取不到内容 |
| PDF | `PyPDFLoader` | **一页** | `source`、`page`（**从 0 开始**）、`total_pages` 等 | 扫描版 PDF 取不到文字；排版复杂时文字顺序错乱 |

```python
from pathlib import Path
from langchain_community.document_loaders import TextLoader, CSVLoader, JSONLoader, PyPDFLoader

# 本文件在 2practice/day03/ 里，资料在上一级的 data 文件夹
DATA = Path(__file__).resolve().parents[1] / "data"

# ① TXT：整个文件变成 1 个 Document
txt_docs = TextLoader(DATA / "员工手册.txt", encoding="utf-8").load()

# ② CSV：每一行变成 1 个 Document，正文是"列名: 值"的形式
csv_docs = CSVLoader(DATA / "产品价目.csv", encoding="utf-8").load()

# ③ JSON：jq_schema 是"取数规则"
#    '.[]' 表示"遍历数组里的每一项"；后半段把问题和答案拼成一段文字
json_docs = JSONLoader(
    DATA / "FAQ.json",
    jq_schema='.[] | "问：" + .question + "\\n答：" + .answer',
).load()

# ④ PDF：每一页变成 1 个 Document
pdf_docs = PyPDFLoader(DATA / "IT安全规范.pdf").load()

# 看看 CSV 加载出来长什么样
print(csv_docs[0].page_content)
print(csv_docs[0].metadata)
```

CSV 第一行的输出类似：

```
产品名称: 星辰智能音箱
型号: XC-S1
价格: 299
保修期: 1年
{'source': '...\\2practice\\data\\产品价目.csv', 'row': 0}
```

**Day 3 动手任务提示**：上面的代码已经把 4 种格式都加载好了，你需要自己写一个循环，对每种格式打印：**Document 个数、每个 Document 的正文长度（`len(doc.page_content)`）、metadata**。观察并记录下面几个问题的答案：
1. TXT 有几个 Document？PDF 呢？为什么不一样？
2. PDF 第一页的 `page` 是 0 还是 1？
3. CSV 和 JSON 的正文，和原文件长得一样吗？

**要点**：
- **PDF 的 `page` 从 0 开始**。给用户展示"第几页"时要 `+1`。
- `source` 是完整路径，展示时用 `Path(doc.metadata["source"]).name` 只取文件名。
- **PDF 加载后文字顺序错乱或出现乱码**：先换一个排版简单的 PDF 测试，不必纠结（逐日规划 Day 3 的卡点）。扫描版 PDF（本质是图片）需要 OCR，本阶段不涉及。
- 想一次加载整个文件夹，可以了解 `DirectoryLoader`，本阶段一个个加载更直观。

> **关于 `langchain-community` 的警告**：运行上面的代码时，你可能会看到一条 `DeprecationWarning: langchain-community is being sunset...`。意思是官方已经**停止维护**这个包（加载器大多在这个包里），以后不会再更新，但**现在仍然可以正常使用**，不影响本阶段学习。官方的新方向是：常用的集成拆成独立的包（如 `langchain-milvus`、`langchain-ollama`），简单的加载逻辑**自己写几行代码**就行。

**官方新写法：自己写一个 PDF 加载器**（官方知识库教程现在就是这么写的）。理解了 4.2 节的 Document，你会发现加载器其实很简单：

```python
from pathlib import Path
import pypdf
from langchain_core.documents import Document

def load_pdf_pages(file_path):
    """读取 PDF，每一页变成一个 Document"""
    reader = pypdf.PdfReader(file_path)
    return [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": str(file_path), "page": i},   # 和 PyPDFLoader 一样，页码从 0 开始
        )
        for i, page in enumerate(reader.pages)
    ]

DATA = Path(__file__).resolve().parents[1] / "data"
docs = load_pdf_pages(DATA / "IT安全规范.pdf")
print(len(docs), docs[0].metadata)
```

> 两种写法学会一种即可。本指南后面的代码继续用 `PyPDFLoader` 等现成的加载器，因为网上的教程和项目代码大多这样写，你需要看得懂。

---

### 4.4 文本切分：块大小和重叠怎么影响效果（Day 4 上午）

**核心思想**：整篇文档太长，既塞不进提示词，检索也不精确。所以要切成小块。**类比**：把一本书拆成一张张知识卡片。卡片太大，一张卡上什么都有，找到了也不知道重点在哪；卡片太小，一句话被拆到两张卡上，单看哪张都不完整。

**关键理解（逐日规划原话）**：**块太大会混入无关内容，块太小会丢失上下文**。

下面用同一份文档对比两种切分方式：

```python
from pathlib import Path
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

DATA = Path(__file__).resolve().parents[1] / "data"
text = (DATA / "员工手册.txt").read_text(encoding="utf-8")

# 方式一：按固定字数硬切（separator="" 表示不看标点，数够字数就切）
fixed_splitter = CharacterTextSplitter(separator="", chunk_size=100, chunk_overlap=0)

# 方式二：递归切分——优先在段落处切，不行再在换行处切，再不行在句号处切……
smart_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    # 默认分隔符是为英文设计的（靠空格分词），中文要加上中文标点
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", ""],
    keep_separator="end",   # 标点留在句子末尾，而不是跑到下一块的开头
)

for name, splitter in [("固定字数", fixed_splitter), ("递归切分", smart_splitter)]:
    chunks = splitter.split_text(text)
    print(f"===== {name}：共 {len(chunks)} 块 =====")
    for c in chunks:
        print(repr(c))
```

**实际运行的部分输出**（固定字数切分）：

```
'第二章 休假制度\n员工入职满一年后享有 5 天带薪年假，……\n病假需提供医院证明，每年前 5'
'天病假工资照发。\n\n第三章 报销制度\n……'
```

看到了吗？"每年前 5 天病假工资照发"被**从中间切断**了。有人问"病假工资怎么算"，检索到第一块只看到"每年前 5"，检索到第二块只看到"天病假工资照发"，两块单看都不完整。递归切分则会在句号、换行处断开，基本不会出现这种情况。

**切分 Document（保留元数据）**：实际入库时要切 `Document` 而不是纯文本，用 `split_documents`，每个小块会**继承**原文档的 metadata：

```python
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA = Path(__file__).resolve().parents[1] / "data"
docs = TextLoader(DATA / "员工手册.txt", encoding="utf-8").load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", ""],
    keep_separator="end",
    add_start_index=True,   # 在 metadata 里记录这一块在原文中的起始位置
)
chunks = splitter.split_documents(docs)

print(len(chunks))
print(chunks[1].metadata)   # {'source': '...员工手册.txt', 'start_index': 59}
```

**参数怎么选**：

| 参数 | 太小 | 太大 | 起步建议 |
|---|---|---|---|
| `chunk_size` | 上下文不完整，一句话被拆开 | 一块里混入多个话题，检索不精确，也更费 token | 中文制度类文档 200～500 字 |
| `chunk_overlap` | 切断处的信息丢失 | 重复内容多，浪费存储 | `chunk_size` 的 10%～20% |

**要点**：
- 本节的示例资料很短，所以用了 `chunk_size=100` 方便观察。正式项目从 300 左右起步，用第 4.12 节的测试问题检验效果后再调。
- **CSV、JSON、PDF 一般不用切**：CSV 一行、JSON 一条本来就很短；PDF 一页如果很长才需要切。实际项目里通常只切"长"的 Document。
- `chunk_size` 数的是**字符数**，不是 token 数，一个汉字算 1 个字符。

---

### 4.5 Embedding：把文字变成"坐标"（Day 4 下午）

**核心思想**：电脑不懂"年假"和"休假"意思相近，但它很会算数。Embedding 模型把每段文字变成一串数字（向量），**意思越接近，这串数字就越接近**。

**类比**：给每段文字算一个 GPS 坐标。"年假几天"和"一年能休几天假"虽然用词不同，但坐标挨得很近；"扫地机器人多少钱"的坐标则离得很远。检索就是"找离问题坐标最近的几个点"。

```python
import numpy as np
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="bge-m3")

# 1. 一段文字 → 一个向量
vector = embeddings.embed_query("年假有几天？")
print("维度：", len(vector))       # 1024
print("前 5 个数字：", vector[:5])

# 2. 算一算"意思有多接近"（余弦相似度：越接近 1 越相似）
def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

question = embeddings.embed_query("一年能休几天假？")
texts = [
    "员工入职满一年后享有 5 天带薪年假",
    "住宿标准：一线城市每晚不超过 500 元",
    "星辰扫地机器人，价格 1999 元",
]
vectors = embeddings.embed_documents(texts)   # 一次处理多段文字

for text, vec in zip(texts, vectors):
    print(f"{cosine(question, vec):.3f}  {text}")
```

你会看到第一句的分数明显最高，尽管问题里没有出现"年假"两个字。这就是**语义检索**比"关键词搜索"强的地方。

**如果选了方案 B（OpenAI 兼容接口）**，只需要换两行：

```python
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")   # 维度 1536
```

**要点**：
- `embed_query`：处理**一个问题**；`embed_documents`：处理**一批文档**。有些模型对两者处理方式不同，所以不要混用。
- **向量维度由模型决定**，下一节在 Milvus 建集合时必须填对，否则会报维度不匹配。
- 换了 Embedding 模型，**整个知识库都要重新入库**，旧向量不能和新模型混用。

---

### 4.6 向量库：为什么不用普通数据库（Day 4 下午）

**核心思想**：普通数据库擅长"精确查找"，向量库擅长"找最相似的"。

| | 普通数据库（如 MySQL） | 向量库（如 Milvus） |
|---|---|---|
| 擅长 | `年龄 = 18`、`名字包含"张"` | "和这个问题意思最接近的 3 段话" |
| 搜"一年能休几天假" | 只能匹配到包含这些字的记录，搜不到"带薪年假" | 能找到"带薪年假"那一段 |
| 原理 | 精确匹配、关键词匹配 | 计算向量之间的距离 |
| 数据量大时 | 有成熟索引 | 有专门的向量索引，几百万条也能毫秒级返回 |

**类比**：普通数据库像图书馆按书名查书；向量库像问图书管理员"有没有讲类似内容的书"。

**LangChain 里所有向量库的用法都一样**：`add_documents`（存）、`similarity_search`（查）、`as_retriever`（变成检索器）。所以你学会 Milvus，换成 Chroma 或其他向量库，代码几乎不用改。

---

### 4.7 Milvus 基础：集合、字段、维度、索引（Day 4 下午）

**核心思想**：先不用 LangChain，直接用 Milvus 官方的 Python SDK（`pymilvus`）亲手建一次集合，弄清楚 Milvus 里到底存了什么。逐日规划要求**用 Python SDK 完成建集合、插入、搜索，不讲 SQL**。

**Milvus 的概念和 Excel 对照**：

| Milvus | 类比 Excel | 本节的例子 |
|---|---|---|
| Collection（集合） | 一张工作表 | `day04_demo` |
| Field（字段） | 一列 | `id`、`text`、`source`、`vector` |
| Entity（实体） | 一行 | 一个文本块 |
| Schema（结构） | 表头：有哪些列、每列什么类型 | 下面代码里的 `schema` |
| 向量维度 dim | 向量列里每格放几个数 | 1024（和 `bge-m3` 一致） |
| Index（索引） | 给向量列建的"快速查找目录" | `AUTOINDEX` + `COSINE` |

```python
from pymilvus import MilvusClient, DataType
from langchain_ollama import OllamaEmbeddings

client = MilvusClient(uri="http://localhost:19530")
COLLECTION = "day04_demo"
DIM = 1024   # 必须和 Embedding 模型的输出维度一致（bge-m3 是 1024）

# 练习时每次重新建，先删掉同名的旧集合
if client.has_collection(COLLECTION):
    client.drop_collection(COLLECTION)

# 1. 定义结构（表头）
schema = MilvusClient.create_schema(auto_id=True)   # auto_id：主键自动生成
schema.add_field("id", DataType.INT64, is_primary=True)            # 主键
schema.add_field("text", DataType.VARCHAR, max_length=2000)        # 正文
schema.add_field("source", DataType.VARCHAR, max_length=200)       # 来源
schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)         # 向量

# 2. 给向量字段建索引：AUTOINDEX 让 Milvus 自动选索引类型，COSINE 表示用余弦相似度
index_params = MilvusClient.prepare_index_params()
index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")

# 3. 建集合
client.create_collection(COLLECTION, schema=schema, index_params=index_params)

# 4. 插入数据：先把文字变成向量，再按"字段名: 值"组成一行行数据
embeddings = OllamaEmbeddings(model="bge-m3")
texts = [
    "员工入职满一年后享有 5 天带薪年假，每多工作一年增加 1 天，最多 15 天。",
    "住宿标准：一线城市每晚不超过 500 元，其他城市每晚不超过 350 元。",
    "餐补标准：每天 100 元，无需发票。",
    "迟到超过 30 分钟视为旷工半天。",
]
vectors = embeddings.embed_documents(texts)
rows = [{"text": t, "source": "员工手册.txt", "vector": v} for t, v in zip(texts, vectors)]
result = client.insert(COLLECTION, rows)
print("插入条数：", result["insert_count"])

# 5. 相似度检索：问题也要先变成向量
query_vector = embeddings.embed_query("出差住酒店能报多少钱？")
hits = client.search(
    COLLECTION,
    data=[query_vector],           # 可以一次查多个问题，所以是列表
    limit=3,                       # top-k
    output_fields=["text", "source"],
)
for hit in hits[0]:                # hits[0] 是第一个问题的结果
    print(f"{hit['distance']:.3f}  {hit['entity']['text']}")
```

**要点**：
- 用 `COSINE` 时，`distance` 实际上是**相似度**，**越大越相似**（最大为 1）。如果用 `L2`（欧氏距离），则是**越小越相似**。看分数前先确认用的是哪种。
- **向量维度一旦建好就不能改**。换 Embedding 模型后，要删掉集合重新建。
- 想看集合里有什么，可以用 Milvus 的网页管理工具 Attu（选学，可以搜索"Milvus Attu"了解）。

---

### 4.8 用 LangChain 把全部文档写入 Milvus（Day 4 晚上）

**核心思想**：4.7 节是"手动挡"，要自己建表、自己算向量、自己组装数据。`langchain-milvus` 是"自动挡"：**给它 Document 列表，它自动建集合、算向量、写入**。

**入库前的关键一步：统一元数据**

4 种加载器产生的 metadata 字段各不相同（PDF 有 `page`，CSV 有 `row`，TXT 都没有）。`langchain-milvus` 默认按**第一批数据**的 metadata 来确定集合的字段，后面的数据字段对不上就会报错。所以入库前，先把所有 Document 的 metadata 统一成固定的几个字段：

| 字段 | 含义 | 例子 |
|---|---|---|
| `source` | 文件名 | `IT安全规范.pdf` |
| `location` | 位置描述，给人看的 | `第 2 页`、`第 3 行`、`第 1 条` |

```python
from collections import Counter
from pathlib import Path
from langchain_community.document_loaders import TextLoader, CSVLoader, JSONLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_milvus import Milvus

DATA = Path(__file__).resolve().parents[1] / "data"

# ① 加载（同 4.3 节）
txt_docs = TextLoader(DATA / "员工手册.txt", encoding="utf-8").load()
csv_docs = CSVLoader(DATA / "产品价目.csv", encoding="utf-8").load()
json_docs = JSONLoader(DATA / "FAQ.json", jq_schema='.[] | "问：" + .question + "\\n答：" + .answer').load()
pdf_docs = PyPDFLoader(DATA / "IT安全规范.pdf").load()

# ② 切分：只有 TXT 比较长，需要切
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", ""],
    keep_separator="end",
)
txt_chunks = splitter.split_documents(txt_docs)

# ③ 统一元数据：每个 Document 只保留 source 和 location 两个字段
def normalize(doc, location):
    doc.metadata = {
        "source": Path(doc.metadata["source"]).name,   # 完整路径 → 只留文件名
        "location": location,
    }
    return doc

all_chunks = (
    [normalize(d, f"第 {i + 1} 段") for i, d in enumerate(txt_chunks)]
    + [normalize(d, f"第 {d.metadata['row'] + 1} 行") for d in csv_docs]
    + [normalize(d, f"第 {d.metadata['seq_num']} 条") for d in json_docs]
    + [normalize(d, f"第 {d.metadata['page'] + 1} 页") for d in pdf_docs]   # page 从 0 开始，所以 +1
)

# 统计每份文档切成了几块（Day 4 晚上的任务要求）
print(Counter(d.metadata["source"] for d in all_chunks))

# ④ 入库：自动建集合、算向量、写入
embeddings = OllamaEmbeddings(model="bge-m3")
vector_store = Milvus(
    embedding_function=embeddings,
    collection_name="enterprise_kb",
    connection_args={"uri": "http://localhost:19530"},
    index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE"},
    auto_id=True,      # 主键自动生成（默认是 False，不设置的话需要自己传 ids）
    drop_old=True,     # 先删掉同名旧集合再建。练习时方便反复重跑；正式使用时改成 False
)
ids = vector_store.add_documents(all_chunks)
print("写入条数：", len(ids))

# ⑤ 马上查一下，确认入库成功
for doc in vector_store.similarity_search("电脑中病毒了怎么办？", k=3):
    print(doc.metadata, doc.page_content[:40])
```
**Day 5 怎么"接着用"已经入库的数据**：不用重新加载和入库，创建 `Milvus` 时去掉 `drop_old=True`（或改成 `False`），用同一个 `collection_name` 就能直接连上：

```python
from langchain_ollama import OllamaEmbeddings
from langchain_milvus import Milvus

vector_store = Milvus(
    embedding_function=OllamaEmbeddings(model="bge-m3"),   # 必须和入库时同一个模型
    collection_name="enterprise_kb",
    connection_args={"uri": "http://localhost:19530"},
)
print(vector_store.similarity_search("年假几天", k=2))
```

**保底方案：Milvus 连续 2 小时连不上，先用 Chroma**（逐日规划的规则）

```powershell
uv add langchain-chroma
```

只需要把上面 `vector_store = Milvus(...)` 那一段换成：

```python
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="enterprise_kb",
    embedding_function=embeddings,
    persist_directory="./chroma_db",   # 数据存成本地文件夹，不需要 Docker
)
```

其余代码（`add_documents`、`similarity_search`、`as_retriever`）**完全不用改**。这就是 4.6 节说的"LangChain 统一了向量库的用法"。

**要点**：
- `drop_old=True` 会**清空**同名集合，只在"重新入库"的脚本里用，问答脚本里千万别写。
- 每份文档的块数要记下来（任务要求），它能帮你判断切分参数是否合理：一份短文档切出几十块，说明 `chunk_size` 太小了。

---

### 4.9 检索器：top-k、MMR 和相似度分数（Day 5 上午）

**核心思想**：向量库调用 `as_retriever()` 就变成**检索器（Retriever）**。检索器是一个 Runnable，可以 `invoke`，也可以用 `|` 接进管道。检索策略有三种：

| `search_type` | 含义 | 类比 | 适合场景 |
|---|---|---|---|
| `"similarity"`（默认） | 返回最相似的 k 个 | 找和问题最像的 k 张卡片 | 大多数情况 |
| `"mmr"` | 在"相关"和"不重复"之间取平衡 | 找 k 张卡片，但不要好几张说的是同一件事 | 资料里重复内容多 |
| `"similarity_score_threshold"` | 只返回分数超过阈值的 | 不够像的卡片宁可不要 | 想过滤明显无关的内容 |

```python
from langchain_ollama import OllamaEmbeddings
from langchain_milvus import Milvus

vector_store = Milvus(
    embedding_function=OllamaEmbeddings(model="bge-m3"),
    collection_name="enterprise_kb",
    connection_args={"uri": "http://localhost:19530"},
)

question = "出差的费用标准是什么？"

# 1. 普通相似度检索：取最相似的 3 个
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
for doc in retriever.invoke(question):
    print("[相似度]", doc.metadata["location"], doc.page_content[:30])

# 2. MMR：先找 10 个候选（fetch_k），再从中挑 3 个"相关又不重复"的
#    lambda_mult 越接近 1 越看重相关性，越接近 0 越看重多样性
mmr_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5},
)
for doc in mmr_retriever.invoke(question):
    print("[MMR]", doc.metadata["location"], doc.page_content[:30])

# 3. 带分数检索：看看每个结果"有多像"，用来决定拒答的阈值
for doc, score in vector_store.similarity_search_with_score(question, k=3):
    print(f"[分数 {score:.3f}]", doc.page_content[:30])
```

**`k` 取多少**：

| k | 好处 | 坏处 |
|---|---|---|
| 太小（1～2） | 提示词短、省钱 | 答案分散在多处时会漏 |
| 太大（10+） | 不容易漏 | 无关内容干扰模型、费 token |
| 起步建议 | **3～5** | 用第 4.12 节的测试题检验后再调 |

**要点**：
- **RAG 答错时，第一件事就是单独运行检索器**，看检索回来的内容对不对。检索不对，模型再聪明也答不对。
- 用 COSINE 时 `similarity_search_with_score` 的分数越大越相似。拿几个"资料里有"和"资料里没有"的问题分别打印分数，就能看出阈值大概设多少。

---

### 4.10 RAG 问答：带来源、会拒答（Day 5 核心）

**核心思想**：逐日规划对成品有三条硬要求：①能回答；②**附带来源**（文件名 + 页码或段落）；③资料里没有的，回答**"资料中未找到"**，不能编造。

实现思路：
1. **给每段资料编号**，并带上来源，一起放进提示词；
2. 在系统提示词里**明确规定**：只能根据资料回答、要标编号、没有就说"资料中未找到"；
3. 程序自己把检索到的来源**单独打印出来**，不完全依赖模型。

```python
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings
from langchain_milvus import Milvus

load_dotenv()

vector_store = Milvus(
    embedding_function=OllamaEmbeddings(model="bge-m3"),
    collection_name="enterprise_kb",
    connection_args={"uri": "http://localhost:19530"},
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 1. 把检索结果排成"带编号、带来源"的资料
def format_docs(docs):
    parts = []
    for i, doc in enumerate(docs, start=1):
        parts.append(f"[{i}] 来源：{doc.metadata['source']} {doc.metadata['location']}\n{doc.page_content}")
    return "\n\n".join(parts)

# 2. 提示词：把规则写清楚
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是星辰科技的企业知识库助手。请严格遵守：\n"
     "1. 只能根据【资料】回答，不得使用资料以外的知识；\n"
     "2. 在每个结论后面用 [编号] 标出依据的是哪段资料；\n"
     "3. 如果资料中没有能回答问题的内容，只回答：资料中未找到相关信息。"),
    ("human", "【资料】\n{context}\n\n【问题】{question}"),
])

model = init_chat_model("openai:gpt-5-nano", temperature=0)   # 问答要稳定，温度设为 0
chain = prompt | model | StrOutputParser()

# 3. 问答函数：检索 → 生成 → 打印来源
def ask(question):
    docs = retriever.invoke(question)
    answer = chain.invoke({"context": format_docs(docs), "question": question})
    print(f"问：{question}")
    print(f"答：{answer}")
    print("参考资料：")
    for i, doc in enumerate(docs, start=1):
        print(f"  [{i}] {doc.metadata['source']} {doc.metadata['location']}")
    print()

ask("一线城市出差住宿能报多少？")
ask("电脑中病毒了应该怎么办？")
ask("公司的股票代码是多少？")   # 资料里没有，应该回答"资料中未找到"
```

**为什么这次不像 LangChain 指南 4.9 那样把检索器直接接进管道？** 4.9 节的写法 `{"context": retriever | format_docs, ...} | prompt | ...` 更简洁，但检索结果在管道里"用完就没了"，程序拿不到它们来打印来源。这里先单独调用 `retriever.invoke`，**同一份检索结果既交给模型，又用来打印来源**，调试时也能直接看到检索到了什么。

**要点**：
- 拒答靠的是**提示词规则**。还不放心，可以再加一道程序判断：用 4.9 节的 `similarity_search_with_score`，最高分低于阈值时，直接返回"资料中未找到"，连模型都不用调。
- "参考资料"列的是**检索到的**来源，模型实际引用了哪几条，看回答里的 `[编号]`。
- 回答看起来对、但没有来源时（逐日规划 Day 5 的卡点），检查两处：①提示词里有没有要求标注；②`format_docs` 有没有把来源写进资料。

---

### 4.11 RAG 进阶模式：查询改写、重排序、Agentic RAG（Day 5 上午）

**核心思想**：上一节是**朴素 RAG**（问题直接检索 → 生成）。它最常见的毛病有两个：用户的说法和文档用词差太远，搜不到；搜到的结果排序不理想，最有用的那段没排进前 k。下面三种模式分别对付这些问题。**逐日规划不要求全部实现，讲清原理，选 1～2 种动手即可。**

| 模式 | 解决什么问题 | 做法 | 类比 |
|---|---|---|---|
| 朴素 RAG | — | 问题 → 检索 → 生成 | 直接拿题目去翻书 |
| **查询改写** | 用户说法口语化，和文档用词对不上 | 先让模型把问题改写成规范说法，再检索 | 先把"出去玩住宿能报多少"翻译成"差旅住宿报销标准" |
| **重排序 Rerank** | 检索结果排序不准 | 先多找一些（如 10 个），再精细打分，取前 3 | 海选 10 人，复试选 3 人 |
| **Agentic RAG** | 有的问题不需要查，有的要查好几次 | 把检索做成工具，交给智能体自己决定 | 考生自己决定要不要翻书、翻几次 |

#### 4.11.1 查询改写（推荐动手）

```python
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model("openai:gpt-5-nano", temperature=0)

class RewrittenQuery(BaseModel):
    query: str = Field(description="改写后、适合在公司制度文档中检索的问题")

rewriter = model.with_structured_output(RewrittenQuery)

def rewrite(question):
    result = rewriter.invoke([
        {"role": "system", "content":
            "把用户的口语化问题改写成适合在公司制度文档中检索的规范问题。"
            "使用文档里可能出现的正式用词（如'差旅''报销标准''带薪年假'），补全省略的信息。"},
        {"role": "user", "content": question},
    ])
    return result.query

print(rewrite("出去玩住宿能报多少"))      # 类似：差旅住宿报销标准是多少
print(rewrite("干满一年能歇几天"))        # 类似：入职满一年的带薪年假天数
```

用法：在 4.10 节的 `ask` 函数里，检索时用 `retriever.invoke(rewrite(question))`，生成时仍然用**原始问题**。

> 逐日规划 Day 5 的卡点："问题稍微换个说法就答不出"——这正是引出查询改写的好时机。可以先用朴素 RAG 找一个答不出的说法，加上改写后再试，对比效果。

#### 4.11.2 重排序 Rerank（理解原理，选做）

专业做法是用**重排序模型**（如 `bge-reranker` 系列，或云平台提供的 Rerank 接口），又快又准。为了看懂原理，下面先用大模型打分的方式实现一个简易版：

```python
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model("openai:gpt-5-nano", temperature=0)

class RelevanceScore(BaseModel):
    score: int = Field(description="这段资料对回答问题的帮助程度，0 到 10 分")

scorer = model.with_structured_output(RelevanceScore)

def rerank(question, docs, top_n=3):
    """给每段资料打分，按分数从高到低取前 top_n 个"""
    scored = []
    for doc in docs:
        result = scorer.invoke(f"问题：{question}\n资料：{doc.page_content}\n请为这段资料打分。")
        scored.append((result.score, doc))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [doc for score, doc in scored[:top_n]]

# 用法：先粗召回 10 个，再精排出 3 个
# candidates = vector_store.as_retriever(search_kwargs={"k": 10}).invoke(question)
# best_docs = rerank(question, candidates)
```

**要点**：大模型打分要对每段资料调用一次模型，**慢而且贵**，只适合理解原理。正式项目用专门的重排序模型。

#### 4.11.3 Agentic RAG（了解，Day 8 会用到）

把检索包装成 `@tool`，交给 `create_agent`。模型自己判断：打招呼不用查，问制度就查，一次没查全就再查一次。

```python
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import OllamaEmbeddings
from langchain_milvus import Milvus

load_dotenv()

vector_store = Milvus(
    embedding_function=OllamaEmbeddings(model="bge-m3"),
    collection_name="enterprise_kb",
    connection_args={"uri": "http://localhost:19530"},
)

@tool
def search_company_docs(query: str) -> str:
    """在星辰科技的内部资料（员工手册、产品价目、FAQ、IT 安全规范）中检索。
    用户问到公司制度、产品、售后、IT 相关问题时使用。

    Args:
        query: 检索用的问题，尽量使用正式用词
    """
    docs = vector_store.similarity_search(query, k=3)
    return "\n\n".join(
        f"来源：{d.metadata['source']} {d.metadata['location']}\n{d.page_content}" for d in docs
    )

agent = create_agent(
    model="openai:gpt-5-nano",
    tools=[search_company_docs],
    system_prompt="你是星辰科技的助手。涉及公司信息时必须先检索，并在回答末尾注明来源；检索不到就说资料中未找到。",
)

result = agent.invoke({"messages": [{"role": "user", "content": "扫地机器人保修几年？坏了退货要在几天内？"}]})
print(result["messages"][-1].content)
```

**两步 RAG 和 Agentic RAG 怎么选**：

| | 两步 RAG（4.10 节） | Agentic RAG（本节） |
|---|---|---|
| 每次都检索吗 | 是，固定流程 | 模型自己决定 |
| 可控性 | 高，行为可预测 | 较低，模型可能不查就答 |
| 速度和费用 | 稳定 | 可能多轮调用，更慢更贵 |
| 适合 | 企业问答成品（Day 5） | 复杂、多步的问题；Day 8 的多智能体客服 |

---

### 4.12 RAG 评估：10 题打分表（Day 5 晚上）

**核心思想**：凭感觉说"挺好用的"不算数。准备 10 道题，固定下来，每次改了参数（`chunk_size`、`k`、提示词）就重新跑一遍打分，**用数字说话**。逐日规划要求做**人工评估**，自动化评估框架不在本阶段范围。

**10 道题怎么出**（覆盖不同类型，才能发现不同问题）：

| 类型 | 题数 | 检验什么 | 例子 |
|---|---|---|---|
| 直接问 | 4 | 基本检索能力 | 一线城市出差住宿能报多少？ |
| 换个说法问 | 2 | 语义检索、查询改写的效果 | 干满一年能歇几天？ |
| 跨文档问 | 1 | 一次检索能否覆盖多个来源 | 扫地机器人多少钱？退货要几天内申请？ |
| 看页码 | 1 | PDF 页码是否正确 | 电脑中病毒怎么办？（应标注第 3 页） |
| 资料里没有 | 2 | **拒答能力** | 公司的股票代码是多少？ |

**打分表模板**（复制到 README 或单独的 `评估表.md`，每次评估填一份）：

| 编号 | 问题 | 期望答案 | 实际答案 | 是否命中来源 | 是否通过 |
|---|---|---|---|---|---|
| 1 | 一线城市出差住宿能报多少？ | 每晚不超过 500 元 | | 员工手册.txt | |
| 2 | 干满一年能歇几天？ | 5 天带薪年假 | | 员工手册.txt | |
| 3 | 电脑中病毒怎么办？ | 断网，拨打 8888 | | IT安全规范.pdf 第 3 页 | |
| … | … | … | | | |
| 9 | 公司的股票代码是多少？ | 资料中未找到 | | （无） | |

**用脚本批量跑，结果存成表格再人工打分**：

```python
import csv
from pathlib import Path

# 假设 ask_once(question) 是你仿照 4.10 节写的函数，返回 (回答, 来源列表)
from rag_app import ask_once

questions = [
    "一线城市出差住宿能报多少？",
    "干满一年能歇几天？",
    "公司的股票代码是多少？",
    # …… 补满 10 道
]

out_file = Path(__file__).parent / "评估结果.csv"
# utf-8-sig：带 BOM 的 UTF-8，Windows 上用 Excel 打开中文才不会乱码
with open(out_file, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["编号", "问题", "实际答案", "检索到的来源", "是否通过（人工填）"])
    for i, q in enumerate(questions, start=1):
        answer, sources = ask_once(q)
        writer.writerow([i, q, answer, "；".join(sources), ""])

print("已保存到", out_file)
```

> `from rag_app import ask_once` 的意思是：从同一文件夹下的 `rag_app.py` 中导入 `ask_once` 函数。你需要把 4.10 节的 `ask` 改造成**返回**回答和来源（而不是打印），存成 `rag_app.py`。

**要点**：
- **一次只改一个参数**，然后重跑 10 道题。同时改好几个，就不知道是哪个起了作用。
- 没通过的题，按 4.1 节的顺序从前往后查：加载 → 切分 → 检索 → 提示词 → 模型。

---

## 第 5 章　常见坑与排错

| 现象 | 可能原因 | 解决办法 |
|---|---|---|
| `uv add` 时 `milvus-lite` 装不上 / 找不到 | Milvus Lite 不支持 Windows | 用 Docker 版 Milvus（第 2.3 节），不需要 Milvus Lite |
| 连接 Milvus 报 `Fail connecting to server on localhost:19530`、超时 | Milvus 没启动，或刚启动还没准备好 | ①打开 Docker Desktop；②`docker ps` 看三个容器是否都在运行；③刚启动要等 1～2 分钟；④浏览器打开 http://localhost:9091/healthz 看是否显示 `OK` |
| `docker` 命令卡住没反应，或报 `cannot connect to the Docker daemon` | Docker Desktop 没打开 | 打开 Docker Desktop，等引擎启动完成后再执行 |
| `docker compose up` 报端口被占用（`port is already allocated`） | 19530 / 9000 / 9091 端口被其他程序占了，或旧容器还在 | `docker ps -a` 查看是否有旧的 Milvus 容器，有就先 `docker compose down`；仍不行就重启电脑 |
| `vector dimension mismatch`、维度不一致 | 建集合时的维度和 Embedding 模型输出不一致；或者中途换了 Embedding 模型 | `bge-m3` 是 1024 维。换模型后要删掉集合重新入库（`drop_old=True`） |
| 入库时报字段不存在、`field ... not exist` 等 | 不同加载器的 metadata 字段不同，混在一起入库 | 入库前按 4.8 节统一 metadata |
| 入库时报缺少主键 / `primary key` 相关错误 | `langchain-milvus` 的 `auto_id` 默认是 `False` | 创建 `Milvus(...)` 时加 `auto_id=True` |
| Ollama 报 `model "bge-m3" not found` | 没下载 Embedding 模型 | `ollama pull bge-m3`，下载完用 `ollama list` 确认 |
| `ollama list` 为空，但明明下载过模型 | `OLLAMA_MODELS` 环境变量指向了错误目录（你之前遇到过） | 检查用户级和系统级的 `OLLAMA_MODELS` 是否一致，结束 `ollama app.exe` 和 `ollama.exe` 后重启服务 |
| 用 `qwen2.5:7b` 做 Embedding 效果很差或报错 | 聊天模型不是 Embedding 模型 | 用专门的 Embedding 模型，如 `bge-m3` |
| 读 CSV 报 `UnicodeDecodeError` | 用 Excel 另存的 CSV，在中文 Windows 上默认是 **GBK** 编码 | 用 VS Code 打开，右下角点编码 →"通过编码保存"→ UTF-8；或者 Excel 另存时选 **"CSV UTF-8（逗号分隔）"**；或者加载时改成 `encoding="gbk"` |
| 读 TXT 报 `UnicodeDecodeError` | 文件不是 UTF-8 编码 | 用记事本或 VS Code 另存为 UTF-8；代码里写 `encoding="utf-8"` |
| `JSONLoader` 报 `jq` 相关错误 | 没装 `jq` | `uv add jq` |
| 出现 `DeprecationWarning: langchain-community is being sunset` | 官方已停止维护 `langchain-community` | 只是警告，不影响运行；也可以按 4.3 节自己写加载器 |
| `JSONLoader` 报 `Expected page_content is string` | `jq_schema` 取出来的是对象，不是文字 | 按 4.3 节把字段拼成字符串，或加参数 `text_content=False` |
| PDF 加载后正文是空的 | 扫描版 PDF，里面是图片不是文字 | 换一个能选中文字的 PDF；扫描版需要 OCR，本阶段不涉及 |
| PDF 文字顺序错乱、有乱码 | PDF 排版复杂（多栏、表格） | 先换一个排版简单的 PDF 测试，不必纠结 |
| 标注的页码比实际少 1 | `page` 从 0 开始 | 展示时用 `page + 1` |
| 切分后每块开头都是一个"。" | 分隔符被放到了下一块的开头 | 加参数 `keep_separator="end"` |
| 中文切分效果差，句子被从中间切断 | 默认分隔符是为英文设计的 | 按 4.4 节加上中文标点分隔符 |
| 回答看起来对，但没有来源 | 提示词没要求标注，或者没把来源写进资料 | 检查系统提示词和 `format_docs`（4.10 节） |
| 资料里没有的问题，模型照样编答案 | 提示词规则不够明确，或 `temperature` 太高 | 系统提示词写明"只能根据资料回答"；`temperature=0`；再加分数阈值判断 |
| 换个说法就答不出 | 用户用词和文档差太远 | 加查询改写（4.11.1）；检查切分是否把关键信息切散了 |
| 评估结果 CSV 用 Excel 打开中文乱码 | Excel 打开不带 BOM 的 UTF-8 文件会乱码 | 写文件时用 `encoding="utf-8-sig"`（4.12 节） |
| `FileNotFoundError` 找不到 `员工手册.txt` | 运行目录和代码里的相对路径对不上 | 用 `Path(__file__).resolve().parents[1] / "data"` 拼绝对路径 |
| 中文输出变成乱码（如 `鍒╃敤`） | Windows 终端编码不是 UTF-8 | 运行前执行 `$env:PYTHONUTF8 = "1"` |
| `ModuleNotFoundError: No module named 'langchain_milvus'` | 没装，或者没用项目环境运行 | `uv add langchain-milvus`；用 `uv run python 文件名.py` 运行 |
| 电脑很卡 | Milvus 容器 + Ollama 同时占用大量内存 | 不用时 `docker compose stop` 关掉 Milvus |

**RAG 排错三步法**：
1. **看检索结果**：单独运行 `retriever.invoke(问题)`，打印每段内容和来源。80% 的问题在这一步就能发现。
2. **看提示词**：把 `format_docs(docs)` 的结果打印出来，确认交给模型的资料确实包含答案。
3. **最后才看模型**：资料对、提示词对，模型还答错，再考虑换模型或改提示词的措辞。

---

## 第 6 章　学习路线（三天计划）

> 逐日规划是全天集训，本阶段只有 2.5 天，所以本章按"天"而不是按"周"安排。如果你是在校自学，按规划第十节"1 个集训日 ≈ 2～3 个晚上 + 半个周末"换算。

| 时间 | 学习内容 | 阶段目标（检验标准） |
|---|---|---|
| **Day 3 上午** | LangChain 指南 4.14 MCP | MCP 示例跑通，智能体能调用 MCP Server 的工具 |
| **Day 3 下午** | 第 2 章环境准备 · 4.1～4.3 | `check_env.py` 两行都能打印；4 种格式都能加载，并能说出每种格式"一个 Document 对应什么" |
| **Day 3 晚上** | 4.1 手绘流程图 | 一张标出每一步输入和输出的 RAG 流程图（拍照提交） |
| **Day 4 上午** | 4.4 切分 | 能举出一个"固定字数切分把句子切断"的例子，并说清 `chunk_size` 和 `chunk_overlap` 的作用 |
| **Day 4 下午** | 4.5～4.7 | 用 pymilvus 建集合、插入、检索，能用一个问题查出最相关的 3 段 |
| **Day 4 晚上** | 4.8 | 全部 4 份资料入库，记录每份文档的块数 |
| **Day 5 上午** | 4.9 · 4.11 | 能对比 similarity 和 MMR 的结果；至少动手做一种进阶模式（推荐查询改写） |
| **Day 5 下午** | 4.10 · 第 7 章项目 4 | 成品能回答、标来源、会拒答 |
| **Day 5 晚上** | 4.12 · README | 10 题打分表填完，README 写好运行步骤并截图，提交 Git |

**学习心法**：
- **每一步都打印出来看**：加载完打印 Document，切分完打印 chunk，检索完打印结果。RAG 的问题大多是"中间某一步的数据不对"。
- **先跑通，再调优**：Day 4 先用默认参数把数据入库，Day 5 有了测试题，再回头调 `chunk_size` 和 `k`。
- **按降级规则保底**：Milvus 连续 2 小时连不上就先用 Chroma；Day 5 做不完就先保住"能问答 + 标来源"两项（规划第八节）。

---

## 第 7 章　练习项目

### 项目 1：文档体检器（Day 3 下午）
- **功能**：读取 `data` 文件夹里的 4 种文档，对每种格式输出一张"体检报告"：Document 个数、每个 Document 的正文长度、metadata。
- **涉及知识点**：`Document`、4 种加载器（4.2、4.3）。
- **进阶挑战**：自动遍历 `data` 文件夹，按文件扩展名选择对应的加载器（提示：`Path.suffix`、`Path.glob("*")`）。

### 项目 2：切分参数实验台（Day 4 上午）
- **功能**：对同一份文档，用 3 组不同的 `chunk_size` / `chunk_overlap` 切分，对比块数和切断位置；再对同一个问题，比较不同切分方式检索到的结果。
- **涉及知识点**：切分（4.4）、Embedding（4.5）。检索对比可以先用 `InMemoryVectorStore`（LangChain 指南 4.9），不用等 Milvus。
- **进阶挑战**：把对比结果整理成一张表格，写下你认为最合适的参数和理由。

### 项目 3：知识库入库脚本（Day 4 晚上）
- **功能**：一个可以反复运行的 `ingest.py`：加载 → 切分 → 统一元数据 → 写入 Milvus，最后打印每份文档的块数和总条数。
- **涉及知识点**：4.3、4.4、4.7、4.8。
- **进阶挑战**：给 metadata 增加一个 `category` 字段（如"制度""产品""售后""IT"），检索时用 `expr='category == "IT"'` 只在某一类里查（参考官方 Milvus 集成文档的过滤示例）。

### 项目 4：RAG 企业问答（Day 5，第一个成品）
- **功能**：命令行里循环提问，程序检索 Milvus 并回答，回答附带来源，资料里没有就拒答。
- **涉及知识点**：4.8～4.12，以及 LangChain 指南里的提示词模板、LCEL。
- **验收标准**（来自逐日规划第六节）：
  - [ ] 能加载至少 1 份 PDF 和 1 份 TXT 文档；
  - [ ] 回答附带来源（文件名 + 页码或段落）；
  - [ ] 资料中没有的问题，能回答"资料中未找到"；
  - [ ] 有一张 10 个测试问题的打分表；
  - [ ] README 含运行步骤和截图。
- **建议的文件结构**：

```
2practice/
├── data/                 资料（4 个文件）
└── day05/
    ├── ingest.py         入库（从 Day 4 的项目 3 复制过来）
    ├── rag_app.py        问答：ask_once 函数 + 命令行循环
    ├── evaluate.py       跑 10 道测试题，输出 评估结果.csv
    └── README.md         运行步骤、截图、打分表
```

- **进阶挑战**：加上查询改写（4.11.1），用打分表对比加改写前后的通过率；或者给问答加上多轮记忆（LangChain 指南 4.8），支持追问。

---

## 第 8 章　学习资源

| 资源 | 说明 | 地址 |
|---|---|---|
| **知识库教程（Knowledge Base）** | 官方的"加载 → 切分 → 向量化 → 检索"完整教程，本指南的主要依据 | https://docs.langchain.com/oss/python/langchain/knowledge-base |
| **检索（Retrieval）总览** | 各种 RAG 架构的介绍 | https://docs.langchain.com/oss/python/langchain/retrieval |
| **RAG 教程** | 官方用智能体做 RAG 的教程（偏进阶） | https://docs.langchain.com/oss/python/langchain/rag |
| **文档加载器列表** | 所有支持的加载器 | https://docs.langchain.com/oss/python/integrations/document_loaders |
| **文本切分器列表** | 所有支持的切分器 | https://docs.langchain.com/oss/python/integrations/splitters |
| **Embedding 模型列表** | 所有支持的 Embedding 模型 | https://docs.langchain.com/oss/python/integrations/text_embedding |
| **Ollama Embedding** | `OllamaEmbeddings` 的用法 | https://docs.langchain.com/oss/python/integrations/text_embedding/ollama |
| **Milvus 集成（LangChain）** | `langchain-milvus` 的用法：连接、过滤、MMR、混合检索 | https://docs.langchain.com/oss/python/integrations/vectorstores/milvus |
| **Chroma 集成（LangChain）** | 保底方案 Chroma 的用法 | https://docs.langchain.com/oss/python/integrations/vectorstores/chroma |
| **Milvus 官方文档** | 集合、字段、索引、pymilvus 的详细说明 | https://milvus.io/docs |
| **bge-m3 模型页（Ollama）** | 本指南用的 Embedding 模型 | https://ollama.com/library/bge-m3 |

**查资料的小技巧**：
- 网上很多 RAG 教程用的是旧写法，如 `RetrievalQA`、`load_qa_chain`、`ConversationalRetrievalChain`，这些在 LangChain 1.x 里**已经过时**（被移到了 `langchain-classic` 兼容包）。看到它们就跳过，以官方文档为准。
- Milvus 的老教程常用 `connections.connect()` + `Collection()` 的写法（ORM 风格）。本指南用的 `MilvusClient` 是官方现在主推的写法，两者能做的事一样，不要混着用。

---

## 第 9 章　小结与自测清单

### 9.1 全文一图回顾

```
          【准备阶段 · Day 3～4】                         【问答阶段 · Day 5】

  TXT/CSV/JSON/PDF                                     用户问题
        │ 加载器（4.3）→ Document（正文 + 元数据）            │ （可选）查询改写 4.11.1
        ▼                                                  ▼
  切分器（4.4）→ chunk_size / overlap                    检索器（4.9）
        │                                              similarity / MMR / top-k
        ▼                                                  │ （可选）重排序 4.11.2
  统一元数据（4.8）source + location                        ▼
        │                                              带编号、带来源的资料
        ▼                                                  │
  Embedding（4.5）bge-m3 → 1024 维向量                      ▼
        │                                              提示词：只根据资料、标编号、没有就拒答
        ▼                                                  │
  Milvus（4.7、4.8）集合 / 字段 / 索引 ─── 检索时查这里 ──┘   ▼
                                                       回答 + 来源 → 10 题打分表评估（4.12）
```

### 9.2 自测清单

- [ ] 我能用"开卷考试"的比喻讲清 RAG 是什么、解决什么问题
- [ ] 我能画出 RAG 的六个步骤，并说出每一步的输入和输出
- [ ] 我知道 TXT、CSV、JSON、PDF 四种加载器各自"一个 Document 对应什么"
- [ ] 我知道 PDF 的 `page` 从 0 开始，展示时要加 1
- [ ] 我能举例说明"块太大混入无关内容，块太小丢失上下文"
- [ ] 我知道中文切分要加中文标点分隔符
- [ ] 我能解释为什么"年假"和"一年能休几天假"的向量会很接近
- [ ] 我知道入库和查询必须用同一个 Embedding 模型，建集合的维度必须和模型一致
- [ ] 我能用 pymilvus 建集合、插入、检索，并说出 Collection、Field、Index 的作用
- [ ] 我知道入库前为什么要统一元数据
- [ ] 我能说出 similarity 和 MMR 的区别，以及 `k` 太大、太小各有什么问题
- [ ] 我的问答程序能标注来源，资料里没有的问题能拒答
- [ ] 我能说出查询改写、重排序、Agentic RAG 各自解决什么问题
- [ ] RAG 答错时，我知道第一步应该先看检索结果

### 9.3 一句话原则

> **数据进得对（加载、切分、元数据），才能找得准（Embedding、检索）；找得准，才能答得对（提示词、拒答）；答得对不对，用打分表说话。**

---

*本文档为入门学习用途。代码以写作时的官方文档版本为准，遇到接口变化请查阅第 8 章的官方链接。*
