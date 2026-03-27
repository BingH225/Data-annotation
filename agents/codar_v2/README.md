# codar_v2 (independent script)

This version is an independent script forked from `experiment/nus_evalfin_319.py`.

Changes in v2:
- Added a dedicated subject/target agent framework at `agents/codar_v2/agent_framework/`.
- Pipeline for role scoring only:
  1. explicit perception decomposition (text/image/audio structured fields)
  2. social-context knowledge graph (Subject Relations)
  3. role slot recommendation + post-decode role correction
- No chain-of-thought output is requested.
- Mechanism/label taxonomy and matching logic remain unchanged.

Default run:
```powershell
python agents/codar_v2/nus_evalfin_319_v2.py
```

Default input:
- `agents/codar_v2/data/eval300_affection100_from_baseline300.json`

Key env overrides:
- `INPUT_JSON_PATH`
- `OUTPUT_DIR`
- `MODEL_NAME` or `EVAL_MODEL_NAME`
- `SILICONFLOW_BASE_URL` / `VLLM_BASE_URL`
- `SILICONFLOW_API_KEY` / `VLLM_API_KEY` / `OPENAI_API_KEY`
- `SUBJECT_TARGET_PIPELINE_STRATEGY` (`anchor_bias` | `model_first` | `anchor_only`)
- `SUBJECT_TARGET_ANCHOR_MIN_CONF` (default `0.55`)
