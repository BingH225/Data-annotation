# Attitude + Intent Error Analysis & Optimization (398070 / 398071)

## 1) 复盘结果
- attitude (N=100): subject=0.78, target=0.73, mechanism=0.34, label=0.15, joint=0.07
- intent (N=100): subject=0.72, target=0.72, mechanism=0.40, label=0.20, joint=0.03

## 2) 主要问题与证据
1. 规则引擎过稀疏（S3.1 rule 激活低）
- 证据: attitude `rule_zero_rate=0.94`，intent `rule_zero_rate=0.97`。
- 影响: 机制判断过度依赖 LLM 波动，导致 mechanism 混淆偏高。

2. S1 占位符过多，导致后续阶段证据链不稳
- 证据: attitude `s1_placeholder_row_rate(>=3)=0.22`，intent `=0.12`。
- 证据: critic 高频问题集中在 `perception_json contains unspecified...`。
- 影响: S2/S3/S5 对同一样本的推理上下文不一致。

3. attitude 标签塌缩在 `dismissive/indifferent`
- 证据: 预测分布 `dismissive=37, indifferent=34`；GT 中 `contemptuous=27, hostile=12, disapproving=16`。
- 证据: 高频混淆 `contemptuous->dismissive(13)`, `disapproving->indifferent(7)`。
- 影响: label acc 被显著拉低。

4. intent 标签塌缩在 `provoke`
- 证据: 预测分布 `provoke=51`，GT `provoke=15`。
- 证据: 高频混淆 `mock->provoke(17)`, `alienate->condemn(8)`。
- 影响: label acc 下降并带来 joint acc 下降。

5. 机制-标签映射未利用数据先验
- 证据: 按全量数据统计，intent 的 `expressive aggression` 并非主要映射到 `provoke`，而是 `mock/alienate`更常见。
- 影响: 仅依赖局部关键词时，易将攻击/嘲讽误归为 provoke。

6. critic 回溯触发率高
- 证据: critic_fail_rate: attitude=0.79, intent=0.69。
- 影响: 多轮回溯对性能帮助有限，且加重不一致。

## 3) 已落地优化（框架级）
1. S1 加入确定性 parser 兜底，替换 `unspecified_*`
- 文件: `src/codar/agents/explicit_perception.py`
- 变更: 新增 `_simple_text_parser` 与 `_is_placeholder`，对 text/audio parser 字段做规则提取补全。

2. ConflictEngine 规则打分从“全量均值”改为“top-k + max + 词面命中”，并加场景 cue
- 文件: `src/codar/agents/conflict_engine.py`
- 变更: keyword score 改为更鲁棒聚合；新增 `_cue_score`（attitude/intent 的强触发词）。

3. attitude 最终决策改为“机制先验 + 词汇边界”联合
- 文件: `src/codar/agents/final_decision.py`
- 变更: 重写 `attitude` mechanism/label heuristic，显式抑制 `contemptuous->dismissive`、`disapproving->indifferent`。

4. intent 最终决策改为“机制先验 + 身份攻击识别 + mock/provoke 解耦”
- 文件: `src/codar/agents/final_decision.py`
- 变更: 重写 `intent` mechanism/label heuristic；新增 `intent subject/target anchor`（如 boss->manager, speaker/coworker 锚定）。

5. Prompt 约束加强
- 文件: `prompts/P1_explicit_perception.md`, `prompts/P6_final_decision.md`
- 变更: 明确要求优先输出可解析字段、强化关键标签边界规则。

6. scenario keyword 扩展
- 文件: `config/scenario_policy.yaml`
- 变更: attitude/intent 增补高频误判触发词（如 `third strike`, `faggot`, `is it a boy or a girl` 等）。

## 4) 相关输出文件
- 诊断 JSON: `analysis/attitude_intent_100_diagnosis_398070_398071.json`
- 诊断摘要: `analysis/attitude_intent_100_diagnosis_398070_398071.md`
