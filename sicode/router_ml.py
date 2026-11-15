"""Runtime-trained contextual ML router for model, effort, cost, and efficiency optimization."""
import json
import math
import os
import threading
import time

MODEL_PATH = os.path.join(os.path.expanduser("~"), ".sicode", "router_model.json")

# Approximate relative cost per 1K tokens (USD-equivalent weight; 0.0 for local GPU).
PROVIDER_COST = {
    "local": 0.0,
    "claude-code": 0.008,
    "codex": 0.008,
    "agy": 0.005,
    "gemini": 0.004,
    "codebuddy": 0.004,
    "tcloud": 0.004,
    "api": 0.015,
}

EFFORT_MULT = {"": 1.0, "low": 0.7, "medium": 1.0, "high": 1.35, "xhigh": 1.7, "max": 2.0}


def extract_features(task_text, media=None, cloud_ratio=10):
    """Extract normalized task feature vector [bias, len, code, design, multi, img, vid, cloud_ratio]."""
    low = (task_text or "").lower()
    n = min(1.0, len(low) / 800.0)
    is_code = 1.0 if any(w in low for w in ("fix", "bug", "test", "refactor", "implement", "build", "compile", "error", "api", "function")) else 0.0
    is_design = 1.0 if any(w in low for w in ("ui", "design", "css", "html", "docs", "readme", "explain", "review", "search", "plan", "architecture")) else 0.0
    is_multi = 1.0 if any(w in low for w in ("all", "across", "multiple", "project", "entire", "system", "migrate")) else 0.0
    has_vid = 1.0 if any(str(m.get("mime", "")).startswith("video/") for m in (media or [])) else 0.0
    has_img = 1.0 if any(str(m.get("mime", "")).startswith("image/") for m in (media or [])) else 0.0
    cr = max(0.0, min(1.0, float(cloud_ratio) / 100.0))
    return [1.0, n, is_code, is_design, is_multi, has_img, has_vid, cr]


class RuntimeRouterML:
    """Online contextual bandit (SGD linear model + UCB exploration) trained at runtime."""

    def __init__(self, path=MODEL_PATH):
        self.path = path
        self.lock = threading.Lock()
        self.arms = {}  # arm_key ("provider:effort") -> {"w": [float...], "n": int, "avg_eff": float, "avg_tok_s": float, "total_cost": float}
        self._load()

    def _load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("arms"), dict):
                self.arms = data["arms"]
        except (OSError, ValueError):
            self.arms = {}

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"arms": self.arms}, f)
        except OSError:
            pass

    @staticmethod
    def _prior(provider, effort, x, quota_left, tok_rate):
        _, _, is_code, is_design, is_multi, has_img, has_vid, _ = x
        score = 0.55 + 0.25 * (max(0.0, min(100.0, quota_left)) / 100.0)
        if has_vid > 0.5 and provider in ("gemini", "agy"):
            score += 0.35
        elif has_img > 0.5 and provider in ("agy", "gemini", "claude-code", "codex", "api"):
            score += 0.15
        if is_code > 0.5 and provider in ("codex", "claude-code", "codebuddy"):
            score += 0.12
        if is_design > 0.5 and provider in ("agy", "gemini", "claude-code"):
            score += 0.12
        cost_w = PROVIDER_COST.get(provider, 0.008) * EFFORT_MULT.get(effort, 1.0)
        score -= cost_w * (4.0 if is_multi < 0.5 else 2.0)
        # Penalize high recent token burn rate to balance load
        score -= min(0.20, tok_rate / 50000.0)
        return score

    def score_arm(self, provider, effort, x, quota_left=100.0, tok_rate=0.0):
        key = f"{provider}:{effort or 'default'}"
        arm = self.arms.get(key)
        prior = self._prior(provider, effort, x, quota_left, tok_rate)
        if not arm or arm.get("n", 0) == 0:
            return prior + 0.08  # small exploration bonus for untried combinations
        w = arm["w"]
        pred = sum(wi * xi for wi, xi in zip(w, x))
        n = arm["n"]
        ucb = 0.12 / math.sqrt(n + 1.0)
        blend = min(0.8, n / (n + 5.0))
        return (1.0 - blend) * prior + blend * pred + ucb

    def recommend(self, task_text, candidates, media=None, cloud_ratio=10, usage_map=None, default_efforts=None):
        """Return (best_provider, best_effort, ranked_list) for the given task."""
        x = extract_features(task_text, media, cloud_ratio)
        usage_map = usage_map or {}
        default_efforts = default_efforts or {}
        efforts = ("low", "medium", "high")
        scored = []
        with self.lock:
            for p in candidates:
                u = usage_map.get(p) or {}
                quota = float(u.get("quota_left", 100.0))
                tok_rate = float(u.get("tok_per_min", 0.0))
                pref_ef = default_efforts.get(p) or ""
                cand_efforts = [pref_ef] if pref_ef else list(efforts)
                for ef in cand_efforts:
                    sc = self.score_arm(p, ef, x, quota_left=quota, tok_rate=tok_rate)
                    scored.append((sc, p, ef))
        scored.sort(key=lambda t: -t[0])
        if not scored:
            return "claude-code", "medium", []
        _, best_p, best_ef = scored[0]
        return best_p, best_ef, [{"provider": p, "effort": ef or "default", "score": round(sc, 3)} for sc, p, ef in scored[:5]]

    def observe(self, provider, effort, task_text, media, cloud_ratio, success, elapsed_s, tokens):
        """Update the online ML model after a completed run."""
        x = extract_features(task_text, media, cloud_ratio)
        cost = (max(0, tokens) / 1000.0) * PROVIDER_COST.get(provider, 0.008) * EFFORT_MULT.get(effort or "", 1.0)
        speed_bonus = 1.0 / (1.0 + max(0.0, elapsed_s) / 60.0)
        cost_penalty = min(0.5, cost * 2.0)
        reward = (0.75 if success else 0.05) + 0.25 * speed_bonus - cost_penalty
        reward = max(0.0, min(1.2, reward))
        key = f"{provider}:{effort or 'default'}"
        with self.lock:
            arm = self.arms.setdefault(key, {"w": [0.0] * len(x), "n": 0, "avg_eff": 0.0, "avg_tok_s": 0.0, "total_cost": 0.0})
            w = arm["w"]
            pred = sum(wi * xi for wi, xi in zip(w, x))
            lr = 0.15 / (1.0 + 0.05 * arm["n"])
            err = reward - pred
            for i in range(len(w)):
                w[i] = max(-2.0, min(2.0, w[i] + lr * (err * x[i] - 0.01 * w[i])))
            n = arm["n"] + 1
            arm["n"] = n
            arm["avg_eff"] = round(((n - 1) * arm["avg_eff"] + reward) / n, 4)
            tok_s = tokens / max(1.0, elapsed_s)
            arm["avg_tok_s"] = round(((n - 1) * arm["avg_tok_s"] + tok_s) / n, 1)
            arm["total_cost"] = round(arm["total_cost"] + cost, 5)
            self._save()
        return cost, reward

    def summary(self):
        with self.lock:
            out = []
            for k, arm in self.arms.items():
                out.append({"arm": k, "runs": arm.get("n", 0), "efficiency": arm.get("avg_eff", 0.0),
                            "tok_s": arm.get("avg_tok_s", 0.0), "cost": arm.get("total_cost", 0.0)})
            out.sort(key=lambda r: (-r["efficiency"], -r["runs"]))
            return out
