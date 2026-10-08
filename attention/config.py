"""Configuration and feature flags for the attention layer.

Everything is opt-in. With ATTENTION_MODE=off (the default) the engine
behaves exactly as it did before this package existed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

VALID_MODES = ("off", "observe", "active")


def _env_int(env: dict, name: str, default: int) -> int:
    try:
        return int(env.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(env: dict, name: str, default: float) -> float:
    try:
        return float(env.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class AttentionConfig:
    mode: str = "off"                       # off | observe | active

    # Repetition / deferral
    duplicate_window: int = 5               # recent thoughts compared against
    near_duplicate_threshold: float = 0.55  # shingle Jaccard counted as "near duplicate"
    paraphrase_threshold: float = 0.5       # content-word Jaccard for a likely paraphrase (guarded)
    deferral_threshold: int = 2             # stalled observations before redirecting

    # Progress
    progress_threshold: float = 0.35        # score at/above which a cycle counts as progress

    # Redirection / idle
    max_consecutive_redirects: int = 1
    idle_backoff_seconds: int = 120
    idle_backoff_max_seconds: int = 1800
    revisit_cooldown_cycles: int = 5
    max_thread_revisits: int = 2
    max_open_threads: int = 25
    thread_expiry_cycles: int = 60

    # Hard budgets (0 = unlimited). These override every nomination.
    max_cycles: int = 0
    max_runtime_seconds: int = 0
    max_approx_tokens: int = 0
    stop_file: str = ""

    # Storage
    log_path: str = "./logs/attention.jsonl"
    state_path: str = "./logs/attention_state.json"

    random_seed: int = 7
    resume: bool = True                     # reload controller state snapshot on (re)start

    @property
    def enabled(self) -> bool:
        return self.mode in ("observe", "active")

    @property
    def active(self) -> bool:
        return self.mode == "active"

    @classmethod
    def from_env(cls, env: dict | None = None, base_dir: Path | None = None) -> "AttentionConfig":
        env = dict(os.environ if env is None else env)
        mode = str(env.get("ATTENTION_MODE", "off")).strip().lower()
        if mode not in VALID_MODES:
            mode = "off"

        def _path(name: str, default: str) -> str:
            raw = env.get(name, default)
            p = Path(raw)
            if base_dir is not None and not p.is_absolute():
                p = base_dir / p
            return str(p)

        return cls(
            mode=mode,
            duplicate_window=max(1, _env_int(env, "ATTENTION_DUPLICATE_WINDOW", 5)),
            near_duplicate_threshold=_env_float(env, "ATTENTION_NEAR_DUPLICATE_THRESHOLD", 0.55),
            paraphrase_threshold=_env_float(env, "ATTENTION_PARAPHRASE_THRESHOLD", 0.5),
            deferral_threshold=max(1, _env_int(env, "ATTENTION_DEFERRAL_THRESHOLD", 2)),
            progress_threshold=_env_float(env, "ATTENTION_PROGRESS_THRESHOLD", 0.35),
            max_consecutive_redirects=max(0, _env_int(env, "ATTENTION_MAX_CONSECUTIVE_REDIRECTS", 1)),
            idle_backoff_seconds=max(1, _env_int(env, "ATTENTION_IDLE_BACKOFF_SECONDS", 120)),
            idle_backoff_max_seconds=max(1, _env_int(env, "ATTENTION_IDLE_BACKOFF_MAX_SECONDS", 1800)),
            revisit_cooldown_cycles=max(0, _env_int(env, "ATTENTION_REVISIT_COOLDOWN_CYCLES", 5)),
            max_thread_revisits=max(1, _env_int(env, "ATTENTION_MAX_THREAD_REVISITS", 2)),
            max_open_threads=max(1, _env_int(env, "ATTENTION_MAX_OPEN_THREADS", 25)),
            thread_expiry_cycles=max(1, _env_int(env, "ATTENTION_THREAD_EXPIRY_CYCLES", 60)),
            max_cycles=max(0, _env_int(env, "ATTENTION_MAX_CYCLES", 0)),
            max_runtime_seconds=max(0, _env_int(env, "ATTENTION_MAX_RUNTIME_SECONDS", 0)),
            max_approx_tokens=max(0, _env_int(env, "ATTENTION_MAX_APPROX_TOKENS", 0)),
            stop_file=_path("ATTENTION_STOP_FILE", "") if env.get("ATTENTION_STOP_FILE") else "",
            log_path=_path("ATTENTION_LOG_PATH", "./logs/attention.jsonl"),
            state_path=_path("ATTENTION_STATE_PATH", "./logs/attention_state.json"),
            random_seed=_env_int(env, "ATTENTION_RANDOM_SEED", 7),
            resume=str(env.get("ATTENTION_RESUME", "true")).strip().lower() in ("1", "true", "yes"),
        )


@dataclass(frozen=True)
class RunLimits:
    """Mode-independent run caps for baseline/benchmark runs (PR 0).

    All default to 0 (= unlimited) so production behaviour is unchanged.
    """
    max_thoughts: int = 0
    max_runtime_seconds: int = 0
    history_token_budget: int = 0           # 0 = keep legacy MAX_HISTORY-only trimming
    benchmark_mode: bool = False            # disables self-modifying built-in tools
    record: bool = False                    # write a per-cycle run record (always on in benchmark mode)
    record_dir: str = "./logs/runs"

    @property
    def recording(self) -> bool:
        return self.record or self.benchmark_mode

    def exceeded(self, thoughts_done: int, elapsed_seconds: float) -> str | None:
        """Reason to stop before starting another thought, or None."""
        if self.max_thoughts and thoughts_done >= self.max_thoughts:
            return f"RUN_MAX_THOUGHTS reached ({self.max_thoughts})"
        if self.max_runtime_seconds and elapsed_seconds >= self.max_runtime_seconds:
            return f"RUN_MAX_SECONDS reached ({self.max_runtime_seconds}s)"
        return None

    @classmethod
    def from_env(cls, env: dict | None = None, base_dir: Path | None = None) -> "RunLimits":
        env = dict(os.environ if env is None else env)
        record_dir = Path(env.get("RUN_RECORD_DIR", "./logs/runs"))
        if base_dir is not None and not record_dir.is_absolute():
            record_dir = base_dir / record_dir
        return cls(
            max_thoughts=max(0, _env_int(env, "RUN_MAX_THOUGHTS", 0)),
            max_runtime_seconds=max(0, _env_int(env, "RUN_MAX_SECONDS", 0)),
            history_token_budget=max(0, _env_int(env, "HISTORY_TOKEN_BUDGET", 0)),
            benchmark_mode=str(env.get("BENCHMARK_MODE", "false")).strip().lower() in ("1", "true", "yes"),
            record=str(env.get("RUN_RECORD", "false")).strip().lower() in ("1", "true", "yes"),
            record_dir=str(record_dir),
        )
