"""A rate-capped provider must not shrink the cascade.

Observed live 2026-09-19: every OpenRouter free model answered 429 during a
ranking run and the rebuilt cascade went from 14 free entries to 5. A model that
could not be probed is UNKNOWN, not bad.

Each test below ranks the SECOND cascade entry, so a correct result reorders the
cascade -- otherwise the assertions could not tell retention from a no-op.
"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from hello_operator.ranking import apply_ranking

EP = "https://provider.example/v1"


def _cfg(**router):
    return {
        "models": {
            "a": {"id": "vendor/a:free", "endpoint": EP},
            "b": {"id": "vendor/b:free", "endpoint": EP},
            "paid": {"id": "vendor/paid-model", "endpoint": EP},
        },
        "roles": {"chat": {"cascade": ["a", "b", "paid"]}},
        "router": dict(router),
        # without this the ranking builds no variants and refuses, which would
        # make every assertion below vacuously true
        "ranking": {"provider_order": [EP]},
    }


def _row(mid, tools=2, score=5):
    return {"id": mid, "score": score, "tok_s": 10.0, "latency": 1.0,
            "context": 262144, "tools": tools}


def _apply(tmp_path, cfg, ranked):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    changed, note = apply_ranking(str(path), ranked, cfg)
    assert changed, f"apply_ranking refused, so this test proves nothing: {note}"
    return yaml.safe_load(path.read_text())["roles"]["chat"]["cascade"], note


def test_unprobeable_model_is_retained_below_the_proven_one(tmp_path):
    """'a' was rate-capped this run; only 'b' probed. 'a' must survive, demoted."""
    casc, note = _apply(tmp_path, _cfg(), {EP: [_row("vendor/b:free")]})
    assert casc == ["b", "a", "paid"], f"got {casc} ({note})"


def test_denylisted_model_is_not_retained(tmp_path):
    """Retention must not become a back door for a banned model."""
    casc, note = _apply(tmp_path, _cfg(denylist=["vendor/a"]),
                        {EP: [_row("vendor/b:free")]})
    assert "a" not in casc, f"denylisted model came back via retention: {casc}"
    assert casc == ["b", "paid"], f"got {casc} ({note})"


def test_the_paid_tail_keeps_its_place(tmp_path):
    casc, _ = _apply(tmp_path, _cfg(), {EP: [_row("vendor/b:free")]})
    assert casc[-1] == "paid"


def test_tool_score_outranks_raw_probe_score(tmp_path):
    """A model that can use tools must outrank a higher-scoring one that cannot."""
    cfg = _cfg()
    ranked = {EP: [_row("vendor/a:free", tools=0, score=5),
                   _row("vendor/b:free", tools=2, score=3)]}
    casc, note = _apply(tmp_path, cfg, ranked)
    assert casc.index("b") < casc.index("a"), (
        f"a tool-incapable model outranked a tool-capable one: {casc} ({note})")
