# CoDAR v1

自研多阶段 CoDAR agent 推理框架（affection/attitude/intent）。

## RJG-v1（新框架）
不使用标注答案记忆库，基于输入特征检索 + 多裁判融合：
1. `python -m codar.cli build-memory-index --input data/eval300_shared_aff_att_int_100_each.json --out data/rjg_memory_300.json`
2. `python -m codar.cli run-rjg-batch --input data/eval300_shared_aff_att_int_100_each.json --memory data/rjg_memory_300.json --output-dir output/rjg_run_001 --backend vllm --config config/runtime.internvl.local.yaml`
3. `python -m codar.cli evaluate --predictions output/rjg_run_001/predictions.jsonl --input data/eval300_shared_aff_att_int_100_each.json --output-metrics output/rjg_run_001/metrics.json`
4. `python -m codar.cli tune-rjg --predictions output/rjg_run_001/predictions.jsonl --input data/eval300_shared_aff_att_int_100_each.json --output-dir output/rjg_run_001/tuned --search-budget 120`

## 快速开始
1. 创建并激活虚拟环境。
2. `pip install -U pip && pip install -e .`
3. 复制 `config/runtime.template.yaml` 为 `config/runtime.yaml` 并填写模型信息。
4. 运行：
   - `python -m codar.cli smoke --input ../../Data/datasetv3.18_hf_319_updatev1.json --scenario affection --limit 3 --backend vllm --config config/runtime.yaml`
   - `python -m codar.cli run-batch --input ../../Data/datasetv3.18_hf_319_updatev1.json --output-dir output/run_001 --backend vllm --config config/runtime.yaml`
   - `python -m codar.cli evaluate --predictions output/run_001/predictions.jsonl --input ../../Data/datasetv3.18_hf_319_updatev1.json --output-metrics output/run_001/metrics.json`
