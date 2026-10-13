"""The pilot harness's tools task: generated tasks are fixed by the seed and
consistent with their own answers, and the checker accepts only the right
call. These tests need no GPU or keys."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "length_pilot_run", Path(__file__).resolve().parents[1] / "experiments/length_pilot/run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def test_tools_task_is_fixed_by_seed():
    assert run.make_tools_task(1000, 150) == run.make_tools_task(1000, 150)
    assert run.make_tools_task(1000, 150) != run.make_tools_task(1001, 150)


@pytest.mark.parametrize("seed", range(1000, 1015))
def test_tools_answers_follow_the_account(seed):
    """Replay the correct calls: no quantity goes below zero, and every refund
    takes exactly what the order holds."""
    intro, steps, truth = run.make_tools_task(seed, 150)
    assert len(steps) == len(truth) == 150
    held = {}
    for line in intro.splitlines()[1:]:
        order, rest = line.split(": ", 1)
        items = rest.split("; ")[0].split(", ")
        held[order] = {i.split(" x ")[1]: int(i.split(" x ")[0]) for i in items}
    for call in truth:
        h = held[call["order"]]
        if call["tool"] == "add_item":
            h[call["item"]] = h.get(call["item"], 0) + call["qty"]
        elif call["tool"] == "remove_item":
            assert 1 <= call["qty"] <= h[call["item"]]
            h[call["item"]] -= call["qty"]
        elif call["tool"] == "refund_item":
            assert call["qty"] == h.pop(call["item"])


WANT = {"tool": "refund_item", "order": "B7", "item": "mug", "qty": 3}


@pytest.mark.parametrize("reply,ok", [
    ('{"tool": "refund_item", "order": "B7", "item": "mug", "qty": 3}', True),
    ('```json\n{"tool": "refund_item", "order": "b7", "item": "Mug", "qty": "3"}\n```', True),
    ('{"tool": "refund_item", "args": {"order": "B7", "item": "mug", "qty": 3}}', True),
    ('{"tool": "refund_item", "order": "B7", "item": "mug", "qty": 2}', False),
    ('{"tool": "remove_item", "order": "B7", "item": "mug", "qty": 3}', False),
    ('{"tool": "refund_item", "order": "B7", "item": "mug"}', False),
    ("I will refund the mugs.", False),
])
def test_checker(reply, ok):
    assert run.call_correct(run.parse_call(reply), WANT) is ok


def test_order_task_matches_committed_logs():
    """The order task behind the committed logs must not change."""
    import json
    log = Path(__file__).resolve().parents[1] / "experiments/length_pilot/results/ollama__qwen3_8b__d0.jsonl"
    for line in log.open():
        r = json.loads(line)
        _, _, truth = run.make_task(r["seed"], r["lmax"], r["difficulty"])
        assert truth == [s["true"] for s in r["steps"]]
