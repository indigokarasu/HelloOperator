"""Ranking + free-tier breaker (owner directives, 2026-09-19).

Pure-function tests: no network, no live provider. What they pin down is the
behaviour that was actually wrong at some point today.
"""
import time

import yaml

from hello_operator import ranking
from hello_operator.server import free_exhausted, mark_free_exhausted, _is_free


NOUS, OR = "https://nous.example/v1", "https://openrouter.example/v1"


def _cfg(tmp_path, models, cascade):
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump({
        "router": {"listen": "127.0.0.1:8800"},
        "ranking": {"provider_order": [NOUS, OR]},
        "models": models,
        "roles": {"chat": {"cascade": cascade},
                  "vision": {"cascade": ["moon"], "requires": ["vision"]}}}))
    return str(p)


def _row(mid, score, tok_s, ctx=262144):
    return {"id": mid, "score": score, "tok_s": tok_s, "latency": 1.0, "context": ctx}


def test_model_first_keeps_a_models_providers_adjacent(tmp_path):
    """A model offered by two providers must sit back-to-back: its second
    provider is then one hop away when the first provider's free tier dies."""
    models = {"moon": {"id": "moondream", "endpoint": "http://local/v1", "context_window": 2048},
              "paid": {"id": "vendor/paid", "endpoint": OR, "context_window": 1000}}
    path = _cfg(tmp_path, models, ["paid"])
    ranked = {NOUS: [_row("a:free", 5, 10.0), _row("b:free", 5, 99.0)],
              OR: [_row("b:free", 5, 50.0), _row("a:free", 5, 1.0)]}
    changed, note = ranking.apply_ranking(path, ranked, yaml.safe_load(open(path)))
    assert changed, note
    out = yaml.safe_load(open(path))
    ids = [out["models"][k]["id"] for k in out["roles"]["chat"]["cascade"]]
    assert ids == ["b:free", "b:free", "a:free", "a:free", "vendor/paid"], ids
    # best provider for that model comes first: b was faster on Nous (99 vs 50)
    eps = [out["models"][k]["endpoint"] for k in out["roles"]["chat"]["cascade"][:2]]
    assert eps == [NOUS, OR], eps


def test_same_id_on_two_providers_keeps_two_entries(tmp_path):
    """Keying by id alone collapsed cross-provider duplicates and re-pointed a
    row at the wrong endpoint on the second run."""
    models = {"n-a": {"id": "a:free", "endpoint": NOUS, "context_window": 1},
              "o-a": {"id": "a:free", "endpoint": OR, "context_window": 1},
              "paid": {"id": "vendor/paid", "endpoint": OR, "context_window": 1}}
    path = _cfg(tmp_path, models, ["n-a", "o-a", "paid"])
    ranked = {NOUS: [_row("a:free", 5, 10.0)], OR: [_row("a:free", 5, 20.0)]}
    ranking.apply_ranking(path, ranked, yaml.safe_load(open(path)))
    out = yaml.safe_load(open(path))
    pairs = {(out["models"][k]["id"], out["models"][k]["endpoint"])
             for k in out["roles"]["chat"]["cascade"] if ":free" in out["models"][k]["id"]}
    assert pairs == {("a:free", NOUS), ("a:free", OR)}, pairs


def test_zero_usable_free_models_is_refused(tmp_path):
    """Every probe failing (dead key, provider outage) must not be read as
    'the free tier is empty' and silently demote everything to paid."""
    models = {"paid": {"id": "vendor/paid", "endpoint": OR, "context_window": 1},
              "n-a": {"id": "a:free", "endpoint": NOUS, "context_window": 1}}
    path = _cfg(tmp_path, models, ["n-a", "paid"])
    before = open(path).read()
    changed, note = ranking.apply_ranking(path, {NOUS: [], OR: []}, yaml.safe_load(open(path)))
    assert changed is False and "no usable free models" in note
    assert open(path).read() == before


def test_paid_tail_is_never_reordered(tmp_path):
    models = {"p1": {"id": "vendor/one", "endpoint": OR, "context_window": 1},
              "p2": {"id": "vendor/two", "endpoint": OR, "context_window": 1}}
    path = _cfg(tmp_path, models, ["p1", "p2"])
    ranking.apply_ranking(path, {NOUS: [_row("a:free", 5, 1.0)], OR: []},
                          yaml.safe_load(open(path)))
    out = yaml.safe_load(open(path))
    tail = [out["models"][k]["id"] for k in out["roles"]["chat"]["cascade"]
            if not out["models"][k]["id"].endswith(":free")]
    assert tail == ["vendor/one", "vendor/two"], tail


def test_free_breaker_expires(monkeypatch):
    """Provider-wide, but timed: 'often, but not always' means it must re-test."""
    assert not free_exhausted(NOUS)
    mark_free_exhausted(NOUS, cooldown=60)
    assert free_exhausted(NOUS)
    monkeypatch.setattr(time, "monotonic", lambda: 1e9)   # jump past the cooldown
    assert not free_exhausted(NOUS)


def test_is_free_only_matches_free_ids():
    class S:
        def __init__(self, i, free=False): self.id, self.free = i, free
    assert _is_free(S("vendor/model:free"))
    assert not _is_free(S("vendor/model"))
    assert _is_free(S("stealth/space-bunny-alpha", free=True))


def test_zero_priced_catalogue_entries_count_as_free():
    zero = {"prompt": "0", "completion": "0"}
    assert ranking.is_free_entry({"id": "vendor/model:free"})
    assert ranking.is_free_entry({"id": "stealth/space-bunny-alpha", "pricing": zero})
    assert not ranking.is_free_entry({"id": "openrouter/auto", "pricing": zero})
    assert not ranking.is_free_entry({"id": "vendor/paid", "pricing": {"prompt": "0.0000002", "completion": "0"}})
    assert not ranking.is_free_entry({"id": "vendor/unpriced"})


def test_zero_priced_model_joins_the_free_block_flagged_free(tmp_path):
    models = {"p1": {"id": "vendor/paid", "endpoint": OR, "context_window": 1},
              "moon": {"id": "moondream", "endpoint": "http://local/v1", "context_window": 2048}}
    path = _cfg(tmp_path, models, ["p1"])
    row = dict(_row("stealth/space-bunny-alpha", 5, 1.0), zero_priced=True)
    ranking.apply_ranking(path, {NOUS: [], OR: [row]}, yaml.safe_load(open(path)))
    out = yaml.safe_load(open(path))
    cascade = out["roles"]["chat"]["cascade"]
    first = out["models"][cascade[0]]
    assert first["id"] == "stealth/space-bunny-alpha" and first.get("free") is True
    assert out["models"][cascade[-1]]["id"] == "vendor/paid"


def test_stealth_model_ignores_the_free_tier_breaker():
    from hello_operator.server import _shares_free_quota
    class S:
        def __init__(self, i, free=False): self.id, self.free = i, free
    assert _shares_free_quota(S("vendor/model:free"))
    assert not _shares_free_quota(S("stealth/space-bunny-alpha", free=True))
