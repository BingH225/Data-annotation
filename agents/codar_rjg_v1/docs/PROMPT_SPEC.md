# CoDAR v1 Prompt 规范

## 通用规则
- Prompt 主语言：英文。
- 禁止 few-shot。
- 输出必须为 JSON 对象。
- 日志仅记录：`prompt_id`、`prompt_vars`、`prompt_hash`。
- 失败重试：同一阶段最多2次。

## P0 Scenario Gate
- 目标：确定 `locked_scenario`。
- 输出字段：`locked_scenario`, `validity_flag`, `reason_short`。

## P1 Explicit Perception
- 目标：仅抽取可观察现象，不做情绪/动机推断。
- 输出字段：
  - `text_components.subject/object/predicate/attribute/adverbial`
  - `image_action.subject/background/behavior/action`
  - `audio_caption.subject/object/predicate/attribute/adverbial`

## P2 Social Context
- 目标：构建局部关系图谱与文化线索。
- 输出字段：`entities`, `relations`, `culture_clues`, `domain_notes`。

## P3 Expected Norm
- 目标：生成该社交场景中的常规预期表达 `e`。
- 输出字段：`expected_behavior`, `norm_assumptions`。

## P3b Conflict Judge
- 目标：输出机制级冲突打分与偏差证据。
- 输出字段：
  - `mechanism_scores`（键为该scenario机制）
  - `conflicts[]`（每项含 `trigger_evidence/deviation_object/deviation_direction/confidence`）

## P3c Null Hypothesis Gate
- 目标：在 H0..H5 空间中选择解释路径。
- 输出字段：
  - `hypotheses[]`（每项含 `evidence_fit/context_fit/parsimony/total_score`）
  - `selected_hypothesis`
  - `need_abduction`
  - `reason`

## P4 Abductive ToT
- 目标：生成并评估4条溯因假设。
- 输出字段：
  - `candidates[]`（包含成本分析、动机倒推、机制还原、评分）
  - `selected_id`
  - `best_hypothesis`

## P5 Critic
- 目标：检验证据链闭环与闭集约束。
- 输出字段：`pass`, `issues`, `revision_instructions`, `backtrack_to`。

## P6 Final Decision
- 目标：闭集选择最终四元组。
- 输出字段：`subject`, `target`, `mechanism`, `label`, `confidence`, `decision_rationale_short`。
