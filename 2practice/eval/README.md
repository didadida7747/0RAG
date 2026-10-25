# 评测样例（eval）

> 目标：把 0RAG 从"跑通了"升级为"可证明跑对了"——对应简历缺口「≥10 条固定评测样例」。
> 样例文件：`eval_samples.json`（v0.1，12 条：9 答 / 2 拒答 / 1 边界反编造）。

## 指标定义（与简历口径一致）

| 指标 | 判定 |
|---|---|
| 命中 | 检索引用了 `expected_source`，且回答覆盖 `key_points` |
| 误判 | 引用了错误来源，或编造了语料中不存在的细节 |
| 拒答正确 | `scope=refuse` 的样例得到礼貌拒答而非编造 |
| 延迟 | 端到端毫秒（记录 P50 / P95） |

## 跑法（待实现 run_eval.py）

1. 复用 `day02/4.9RAG.py` 里的链，封装成 `answer(query) -> {answer, sources, latency_ms}`；
2. 读 `eval_samples.json` 逐条执行，结果逐行追加 `results.jsonl`（字段：id, hit, source_correct, refusal_correct, latency_ms, answer）；
3. 汇总输出：命中率 / 拒答正确率 / P50·P95 延迟。

## 记录规范（作品集标准）

- 每次跑完把日期、样例版本、三项百分比写进下表；数字不许手编。

| 日期 | 样例版本 | 命中 | 拒答正确 | 误判 | P50 延迟 | P95 延迟 |
|---|---|---|---|---|---|---|
| （待填） | v0.1 | | | | | |
