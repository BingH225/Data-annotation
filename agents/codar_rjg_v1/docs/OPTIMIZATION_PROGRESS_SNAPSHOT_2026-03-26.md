# CoDAR RJG-v1 优化进度快照（配套 AGENT_WORKFLOW_MANUAL）
更新时刻：2026-03-26 13:24 (Asia/Shanghai)  
配套手册：`docs/AGENT_WORKFLOW_MANUAL.md`

## 1) 这份文档怎么用
- 先读工作手册中的 Hyper Rules 和执行循环。
- 再读本快照，直接从“第 7 节：下一轮执行清单”继续，不需要重新摸索上下文。
- 如果状态变化，以远端输出目录中的最新 metrics 为准。

## 2) 当前总体状态（实时核对）
- 集群队列（`qstat -u e1561245`）当前无 `rjg_affection100_402267` 在队列中，说明该作业已结束。
- 最新 affection 作业目录：  
  `/scratch/e1561245/cot_yz/codar_output/codar_rjg_v1/rjg_affection100_402267.hopper-m-02`
- 该作业已完整产出：
  - `predictions.jsonl`：100 条
  - `tuned/metrics.eval.json`：已生成
  - `metrics.formula_metrics.json`、`metrics.formula_table_ready.json`、`metrics.nus_compat.json`：已生成

## 3) 关键分数看板（用于下一轮比较）

### 3.1 Baseline（固定对照）
来源：`/scratch/e1561245/cot_yz/codar_output/codar_v1/baseline300_internvl38b_399187.hopper-m-02/metrics.json`

- affection: subject `0.52`, target `0.76`, mechanism `0.36`, label `0.33`, joint `0.02`
- attitude: subject `0.56`, target `0.82`, mechanism `0.46`, label `0.14`, joint `0.03`
- intent: subject `0.53`, target `0.74`, mechanism `0.50`, label `0.27`, joint `0.11`

### 3.2 RJG-v1 当前 best（单 scenario）
- affection（label best）：`rjg_affection100_401916.hopper-m-02`
  - subject `0.96`, target `0.94`, mechanism `0.39`, label `0.23`, joint `0.11`
- affection（mechanism best）：`rjg_affection100_402205.hopper-m-02` / `402267.hopper-m-02`
  - subject `0.96`, target `0.94`, mechanism `0.41`, label `0.22`, joint `0.11`
- intent（当前 best）：`rjg_intent100_401917.hopper-m-02`
  - subject `0.97`, target `0.93`, mechanism `0.30`, label `0.21`, joint `0.14`
- attitude（RJG 100 条）：当前未形成稳定 best 记录，需补提单 scenario run 并建同口径看板。

## 4) 最近一轮 affection 连续优化结论（402019 → 402267）
- `402019`: mechanism 提升到 `0.40`，label `0.22`，joint `0.11`
- `402092`: 回归（label `0.16`, joint `0.05`），已回退激进策略
- `402150`: 回到 `0.40/0.22/0.11`
- `402205`: mechanism 提升到 `0.41`，label `0.22`，joint `0.11`
- `402267`: 与 `402205` 持平（`0.41/0.22/0.11`），无新增提升

结论：
- 已稳定提升 mechanism（相对 baseline affection +0.05）。
- label 仍低于 baseline affection（`0.22` vs `0.33`），是当前首要瓶颈。

## 5) 已验证的瓶颈证据（不要重复走弯路）
- affection 候选覆盖并不差，主要问题在“打分选择”：
  - GT label 在候选中出现：`86/100`
  - GT mechanism 在候选中出现：`90/100`
  - GT mechanism+label 同时出现：`44/100`
  - “正确 label 已在候选但没被选中”：`64` 例
- 说明优先优化 M3 judge + M4 fusion，而不是盲目扩召回。

## 6) 已落地代码方向（下一轮应基于此继续）
核心文件：
- `src/codar/rjg/fusion.py`
- `src/codar/rjg/pipeline.py`
- `tests/test_rjg_fusion.py`
- `tests/test_rjg_pipeline.py`

已做且保留的关键改动：
- affection anti-collapse 规则（含惩罚项与修复分支）
- affection 多视角 candidate 多样化
- label judge 权重从“硬匹配优先”调整为“信号+启发式优先”

已证伪并回退：
- 过强 disgust 先验与激进 cue 映射（导致 `402092` 明显回归）

## 7) 下一轮执行清单（接棒即可跑）
1. 目标固定为 affection：只追 `label/joint`，机制分不作为主目标。  
2. 仅改 M3+M4（judge/fusion）：
   - 增加“高频塌缩标签”动态惩罚（按当前样本 top-k 竞争差距触发，不做全局硬罚）。
   - 在 tie-break 阶段强制引入“与 GT 无关的证据完整性分”（避免单 cue 短路）。
3. 本地跑测试：
   - `python -m unittest discover tests -v`
4. 部署并单 scenario 提交（1 卡）：
   - `powershell -ExecutionPolicy Bypass -File scripts/deploy_scp_mirror.ps1 -HostAlias nus_hopper`
   - `ssh nus_hopper "cd /scratch/e1561245/cot_yz/codar_rjg_v1 && qsub scripts/run_rjg_affection100_vllm_1gpu.pbs"`
5. 轮询到第一条推理并记录：
   - `predictions.jsonl` 行数 > 0
6. 完成后比较：
   - 必须同时对比 `401916`（label best）与 `402205/402267`（mechanism best）
   - 若 `label` 无提升，立即进入下一轮，不停留。

## 8) 风险与约束提醒
- 不删除历史输出目录；输出与代码目录分离，禁止重置覆盖旧结果。
- 禁用 URL 媒体慢路径，继续使用本地/远端真实路径。
- 按手册要求每 10 分钟写一条 `logs/optimization_worklog.jsonl`。
- 若不确定下一步，回到：
  1) `docs/AGENT_WORKFLOW_MANUAL.md`
  2) `logs/optimization_worklog.jsonl`
  3) 本文第 7 节
