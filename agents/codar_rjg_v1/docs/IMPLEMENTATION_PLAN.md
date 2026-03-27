# CoDAR v1 实施方案

## 1. 目标
- 从零实现三场景（affection/attitude/intent）CoDAR 推理框架。
- 输入固定：`Data/datasetv3.18_hf_319_updatev1.json`。
- 运行形态：批处理 CLI。
- 后端：`vllm` 或 `openai`，单次运行只允许一个后端。
- 本地开发阶段默认 `media_mode=off`，远端使用 `/scratch/e1561245/Implicit_dataset`。

## 2. 分层架构
- S0 ScenarioGateAgent：锁定场景。
- S1 ExplicitPerceptionAgent：客观可观察证据抽取。
- S2 SocialContextAgent：关系图谱与文化线索建模。
- S3 ExpectationAgent：社交常规预期建模。
- S3.1 ConflictEngine：`Delta(x,e)=0.6*rule + 0.4*llm` 冲突计算。
- S3.5 NullHypothesisGateAgent：H0..H5 假设筛选。
- S4 AbductiveToTAgent：4路溯因假设打分择优。
- S5 CriticAgent：一致性审查与回溯（最多2轮）。
- S6 FinalDecisionAgent：闭集约束下输出最终四元组。

## 3. 目录规范
- `docs/`：实施方案、Prompt规范、部署手册。
- `logs/`：`dev_events.jsonl` 与运行日志。
- `config/`：运行配置、场景策略、阈值配置。
- `prompts/`：P0~P6 模板。
- `src/codar/`：框架源码。
- `scripts/`：本地/远端引导与部署脚本。
- `tests/`：单元、组件、集成测试。

## 4. 输入输出契约
### 输入
- `id`
- `input.scenario`
- `input.text`
- `input.media.*`
- `options.subject[]`
- `options.target[]`
- `diversity.*`
- `ground_truth.*`（仅评测使用）

### 输出
- `subject`
- `target`
- `mechanism`
- `label`
- `confidence`
- `stage_artifacts`
- `trace`
- `backend_meta`

## 5. 媒体解析优先级
1. 远端媒体根：`/scratch/e1561245/Implicit_dataset/{scenario}`，按 `sample_id` 映射。
2. 输入样本 `*_path` 字段。
3. `*_url` 兜底。

## 6. 运行接口
- `run-batch --input --output-dir --backend --config`
- `evaluate --predictions --input --output-metrics`
- `smoke --scenario --limit`

## 7. 验收
- 可运行 + 全链路日志 + 可复现实验报告。
- 首版不设硬性精度门槛。
