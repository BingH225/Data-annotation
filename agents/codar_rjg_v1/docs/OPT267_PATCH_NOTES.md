# CoDAR v1 Opt267 Patch Notes

Date: 2026-03-25

## Scope
This patch implements the six requested optimizations and prepares a reproducible 267-sample dual-GPU experiment run.

## Changes
1. 267-sample subset builder
- Added `scripts/build_eval_subset.py`
- Generated `data/eval267_current_plus_baseline_overlap.json` with deterministic backfill.

2. Evaluation rule alignment + detailed output
- `codar.cli evaluate` now supports `--output-detailed`
- Added detailed record export in `src/codar/eval/metrics.py` and `src/codar/cli.py`

3. Semantic matching (BERT-style) for rule scoring and closed-set mapping
- Added `src/codar/semantic_matcher.py` (lazy transformer encoder, fallback-safe)
- Integrated into:
  - `ConflictEngine` rule scoring (`src/codar/agents/conflict_engine.py`)
  - `FinalDecisionAgent` closed-set mapping (`src/codar/agents/final_decision.py`)

4. Critic loop strengthened
- Increased configured backtrack rounds to 4
- Added consistency-based branch selection in `src/codar/orchestrator/pipeline.py`

5. Subject anchor rule
- Added first-person -> `speaker` anchor in `FinalDecisionAgent`

6. S1 parser hard constraints + temperature override
- Updated S1 prompt to require non-empty parser fields (`prompts/P1_explicit_perception.md`)
- Added code-level non-empty enforcement in `ExplicitPerceptionAgent`
- Added stage-level temperature override plumbing and set S1 to 0.

## Runtime and script updates
- Updated runtime configs in `config/*.yaml`:
  - `max_backtrack_rounds: 4`
  - `alpha_rule: 0.2`, `alpha_llm: 0.8`
  - semantic matcher settings
  - subject anchor enable flag
- Added dual-GPU runner:
  - `scripts/run_267_internvl38b_vllm_2gpu_opt.pbs`

## Local validation
- `python -m unittest discover tests -v` passed
- `python -m codar.cli smoke ... --limit 3 --backend mock` passed
- Verified S1 outputs contain no empty parser fields in smoke output.
