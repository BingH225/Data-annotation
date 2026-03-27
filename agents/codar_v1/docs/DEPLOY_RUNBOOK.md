# CoDAR v1 部署运行手册

## 1. 本地开发目录
- 根目录：`D:/NUS/ACMm/Data-annotation/agents/codar_v1`

## 2. 远端目录
- 代码目录：`/scratch/e1561245/cot_yz/codar_v1`
- 媒体目录：`/scratch/e1561245/Implicit_dataset`
- 独立输出目录：`/scratch/e1561245/cot_yz/codar_output/codar_v1`（不随代码部署覆盖）

## 3. 本地准备
```powershell
cd D:\NUS\ACMm\Data-annotation\agents\codar_v1
python -m venv .venv
.\.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

## 4. 同步到 Hopper（不重置远端项目目录）
```powershell
powershell -ExecutionPolicy Bypass -File scripts\deploy_scp_mirror.ps1 -HostAlias nus_hopper
```

## 5. 远端初始化
```bash
ssh nus_hopper
cd /scratch/e1561245/cot_yz/codar_v1
bash scripts/bootstrap_remote.sh
```

## 6. Smoke 测试
```bash
source .venv/bin/activate
python -m codar.cli smoke \
  --input /scratch/e1561245/cot_yz/codar_v1/data/datasetv3.18_hf_319_updatev1.json \
  --scenario affection \
  --limit 3 \
  --backend vllm \
  --config config/runtime.yaml
```

## 7. 批处理推理
```bash
python -m codar.cli run-batch \
  --input /scratch/e1561245/cot_yz/codar_v1/data/datasetv3.18_hf_319_updatev1.json \
  --output-dir /scratch/e1561245/cot_yz/codar_output/codar_v1/run_001 \
  --backend vllm \
  --config config/runtime.yaml
```

## 8. 评测
```bash
python -m codar.cli evaluate \
  --predictions /scratch/e1561245/cot_yz/codar_output/codar_v1/run_001/predictions.jsonl \
  --input /scratch/e1561245/cot_yz/codar_v1/data/datasetv3.18_hf_319_updatev1.json \
  --output-metrics /scratch/e1561245/cot_yz/codar_output/codar_v1/run_001/metrics.json
```

## 9. 注意事项
- vLLM 端点需提前启动，框架只负责连接。
- 运行前必须补齐配置中的模型名、endpoint、鉴权信息。
- 本地默认 `media_mode=off`，远端建议 `media_mode=local`。