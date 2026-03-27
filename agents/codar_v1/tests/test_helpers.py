from pathlib import Path

from codar.agents.abductive_tot import AbductiveToTAgent
from codar.agents.conflict_engine import ConflictEngine
from codar.agents.critic import CriticAgent
from codar.agents.explicit_perception import ExplicitPerceptionAgent
from codar.agents.expectation import ExpectationAgent
from codar.agents.final_decision import FinalDecisionAgent
from codar.agents.null_hypothesis_gate import NullHypothesisGateAgent
from codar.agents.scenario_gate import ScenarioGateAgent
from codar.agents.social_context import SocialContextAgent
from codar.media import MediaResolver
from codar.orchestrator.pipeline import CoDARPipeline
from codar.prompting import PromptStore


def build_pipeline(backend, runtime_cfg, scenario_policy, max_backtrack_rounds=2):
    root = Path(__file__).resolve().parents[1]
    prompt_store = PromptStore(root / "prompts")
    media_resolver = MediaResolver(runtime_cfg=runtime_cfg)
    max_retries = int(runtime_cfg.get("pipeline", {}).get("max_stage_retries", 1))
    alpha_rule = float(runtime_cfg.get("pipeline", {}).get("alpha_rule", 0.6))
    alpha_llm = float(runtime_cfg.get("pipeline", {}).get("alpha_llm", 0.4))
    max_frames = int(runtime_cfg.get("pipeline", {}).get("max_video_frames", 4))
    return CoDARPipeline(
        scenario_gate=ScenarioGateAgent(backend, prompt_store, max_retries),
        explicit_perception=ExplicitPerceptionAgent(backend, prompt_store, media_resolver, max_retries, max_frames),
        social_context=SocialContextAgent(backend, prompt_store, max_retries),
        expectation=ExpectationAgent(backend, prompt_store, max_retries),
        conflict_engine=ConflictEngine(
            backend,
            prompt_store,
            max_retries,
            alpha_rule=alpha_rule,
            alpha_llm=alpha_llm,
            scenario_policy=scenario_policy,
        ),
        null_gate=NullHypothesisGateAgent(backend, prompt_store, max_retries),
        abductive_tot=AbductiveToTAgent(backend, prompt_store, max_retries),
        critic=CriticAgent(backend, prompt_store, max_retries),
        final_decision=FinalDecisionAgent(backend, prompt_store, max_retries),
        max_backtrack_rounds=max_backtrack_rounds,
    )
