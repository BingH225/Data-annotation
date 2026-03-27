from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from .utils import normalize_token


class SemanticClosedSetMatcher:
    """Optional BERT-based matcher with lexical fallback safety.

    The matcher lazily loads a sentence encoder. If dependencies/model are
    unavailable, it silently degrades to lexical-only behavior (returns None).
    """

    def __init__(
        self,
        enabled: bool = True,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.35,
    ):
        self.enabled = enabled
        self.model_name = model_name
        self.similarity_threshold = float(similarity_threshold)
        self._init_done = False
        self._init_error = ""
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._cache: Dict[str, List[float]] = {}

    def _lazy_init(self) -> bool:
        if self._init_done:
            return self._model is not None
        self._init_done = True
        if not self.enabled:
            return False
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self._torch = torch
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()
        except Exception as exc:  # pragma: no cover
            self._init_error = str(exc)
            self._tokenizer = None
            self._model = None
            self._torch = None
            return False
        return True

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)
        return summed / denom

    def _encode(self, text: str) -> Optional[List[float]]:
        key = normalize_token(text)
        if not key:
            return None
        if key in self._cache:
            return self._cache[key]
        if not self._lazy_init():
            return None
        assert self._tokenizer is not None and self._model is not None and self._torch is not None
        with self._torch.no_grad():
            batch = self._tokenizer(
                [key],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64,
            )
            outputs = self._model(**batch)
            pooled = self._mean_pool(outputs.last_hidden_state, batch["attention_mask"])
            vec = pooled[0]
            vec = vec / vec.norm(p=2).clamp(min=1e-9)
            out = vec.cpu().tolist()
            self._cache[key] = out
            return out

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        return float(sum(x * y for x, y in zip(a, b)))

    def match(self, candidate: str, allowed: Iterable[str]) -> Optional[str]:
        candidate_norm = normalize_token(candidate)
        if not candidate_norm:
            return None
        allowed_list = [str(x) for x in allowed]
        if not allowed_list:
            return None
        cand_vec = self._encode(candidate_norm)
        if cand_vec is None:
            return None
        scored: List[Tuple[float, str]] = []
        for opt in allowed_list:
            opt_vec = self._encode(opt)
            if opt_vec is None:
                continue
            scored.append((self._cosine(cand_vec, opt_vec), opt))
        if not scored:
            return None
        best_score, best_opt = max(scored, key=lambda x: x[0])
        if best_score < self.similarity_threshold:
            return None
        return best_opt

    def similarity(self, text_a: str, text_b: str) -> Optional[float]:
        a = self._encode(text_a)
        b = self._encode(text_b)
        if a is None or b is None:
            return None
        return self._cosine(a, b)

    def metadata(self) -> Dict[str, str]:
        return {
            "enabled": str(self.enabled),
            "model_name": self.model_name,
            "init_error": self._init_error,
        }
