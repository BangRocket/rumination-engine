import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(ROOT))

from attention.config import AttentionConfig  # noqa: E402


def load_fixture(name: str) -> list[dict]:
    data = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    return [{"cycle": i, "thought": t["thought"], "tool_trace": t.get("tool_trace", [])}
            for i, t in enumerate(data["thoughts"], 1)]


@pytest.fixture
def cfg():
    return AttentionConfig(mode="active", log_path="", state_path="")
