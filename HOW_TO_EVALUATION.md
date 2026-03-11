# 评估器操作说明（评估员）

本说明用于当前 `video_app_evaluation.py` 评估器。

## 1. 启动与导入

1. 进入.venv环境（如有）
2. 在终端启动应用：
  `streamlit run video_app_evaluation.py`
3. 在页面顶部 `Input JSON/JSONL path` 输入待评估数据路径（支持 `.json`、`.jsonl`，也支持目录）。
4. 点击 `start_import`。
5. 导入完成后，系统会自动：
  - 生成评估输出文件：`<input_stem>_evaluation.json`
  - 生成进度文件：`<input_stem>_evaluation_process.json`
  - 下载媒体文件到 `images/`（如已存在且有效会复用）

## 2. 页面结构

- 左上：媒体预览（图片/视频）
- 右上：只读 `ID` 与 `Input`
- 下方：4 列字段评估区（每个字段右侧有一个 `F` 勾选框）
  - 第 1 列：`scenario` + 3 个 `mechanism`
  - 第 2 列：3 个 `label`
  - 第 3 列：`subject` + `subject1/2/3`
  - 第 4 列：`target` + `target1/2/3`
- 不显示字段：`domain`、`culture`、`rationale`

说明：

- 字段内容是只读，不能编辑。
- 勾选字段旁 `F` 表示该字段评估不通过。
- 勾选 `F` 后字段会标红。

## 3. 按钮与评估规则（重点）

### 3.1 仅导航，不计完成

- `Go`：跳转到指定页
- `Previous`：上一条
- `Next`（在没有勾选任何 `F` 且没有勾选 `Abandon` 时）：仅下一条导航

以上操作都不算“完成评估”。

### 3.2 评估为“不通过”

1. 勾选 >=1 个字段的 `F`
2. 点击 `Next`

结果：

- 当前条目记为已评估（`evaluated=true`）
- 失败字段写入 `failed_fields`
- 自动保存到输出 JSON
- 自动跳到下一条

### 3.3 评估为“全通过”

1. 确保没有任何 `F` 勾选
2. 确保 `Abandon` 未勾选
3. 点击大按钮 `Accept (All Fields Pass)`

结果：

- 当前条目记为全通过（`overall_pass=true`）
- 15 个核心字段都写为通过
- 自动保存到输出 JSON
- 自动跳到下一条

### 3.4 标记为 `Abandon`

1. 勾选 `Abandon`
2. 点击 `Next`

结果：

- 当前条目标记为 `abandon=true`
- 该条计入完成数
- 自动保存并跳下一条

注意：

- 只勾选 `Abandon` 但不点 `Next`，不会落盘。
- 勾选 `Abandon` 时，`Accept` 不可用。

## 4. 完成数如何计算

页面中的 `Completed` 由以下两类条目组成：

- `abandon=true`
- 或 `evaluation.evaluated=true`

因此，“纯跳转”不会增加完成数。

## 5. 自动保存机制

每次真正提交评估（以下任一）都会立即保存：

- `Accept`（全通过）
- 勾选 `F` 后点 `Next`（部分不通过）
- 勾选 `Abandon` 后点 `Next`（舍弃）

保存内容包括：

- `*_evaluation.json`（逐条评估结果）
- `*_evaluation_process.json`（`total/completed/remaining/current_index/updated_at`）

## 6. 结果数据结构（输出 JSON）

每条样本关键字段如下：

- `input`: 原始输入
- `output.abandon`: 是否舍弃（true/false）
- `output.evaluation.field_pass`: 15 个核心字段逐项通过状态（true/false/null）
- `output.evaluation.failed_fields`: 不通过字段列表
- `output.evaluation.overall_pass`: 本条是否全通过
- `output.evaluation.evaluated`: 本条是否已评估
- `output.evaluation.evaluated_at`: 评估时间

## 7. 修订已评估条目

可通过 `Go` 回到历史条目修改状态：

- 改为全通过：取消所有 `F`，取消 `Abandon`，点击 `Accept`
- 改为部分不通过：按需勾选 `F`，点击 `Next`
- 改为舍弃：勾选 `Abandon`，点击 `Next`

提示：勾选框状态会随当前条目已有结果自动回填。