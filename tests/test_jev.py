"""JEV judges 'free' and 'meets the requirement' in the daily ranking (2026-09-25).

No network: jev.ask and the provider calls are replaced with fakes.
"""
import asyncio

import yaml

from hello_operator import jev, ranking

OR = "https://openrouter.example/v1"
JS = {**jev.DEFAULTS, "api_key": "k", "enabled": True}


def _entry(mid, prompt="0", completion="0", ins=("text",), outs=("text",), ctx=262144):
    return {"id": mid, "context_length": ctx, "supported_parameters": ["tools"],
            "pricing": {"prompt": prompt, "completion": completion},
            "architecture": {"input_modalities": list(ins), "output_modalities": list(outs)}}


def test_settings_needs_a_key(monkeypatch):
    monkeypatch.delenv("JEVER_API_KEY", raising=False)
    assert not jev.settings({})["enabled"]
    monkeypatch.setenv("JEVER_API_KEY", "from-env")
    assert jev.settings({})["api_key"] == "from-env"
    s = jev.settings({"ranking": {"jev": {"api_key": "abc"}}})
    assert s["enabled"] and s["api_key"] == "abc"
    assert not jev.settings({"ranking": {"jev": {"api_key": "${NOT_SET_ANYWHERE_X}"}}})["enabled"]


def test_jev_decides_free_and_meets():
    e = _entry("stealth/bunny")
    d = jev.decide(e, {"free": 0.93, "meets": 0.82, "image_in": 0.99}, JS, True)
    assert d == {"free": True, "meets": True, "vision": True, "by": "jev", "note": ""}
    d = jev.decide(e, {"free": 0.14, "meets": 0.9, "image_in": 0.1}, JS, True)
    assert d["free"] is False and "JEV said paid" in d["note"]
    d = jev.decide(e, {"free": 0.9, "meets": 0.2, "image_in": 0.0}, JS, True)
    assert d["meets"] is False


def test_listed_price_overrides_a_free_verdict():
    """JEV saying 'free' must not put a billed model into the free block."""
    e = _entry("vendor/paid", prompt="0.0000003")
    d = jev.decide(e, {"free": 0.9, "meets": 0.9, "image_in": 0.0}, JS, False)
    assert d["free"] is False and "kept as paid" in d["note"]
    assert jev.listed_price({"pricing": {"prompt": "-1", "completion": "-1"}}) == 1.0
    assert jev.listed_price({"pricing": {"prompt": "0", "completion": "0"}}) == 0.0
    assert jev.listed_price({}) is None


def test_no_verdict_falls_back_to_rules():
    img_only = _entry("vendor/painter", outs=("image",))
    d = jev.decide(img_only, None, JS, True)
    assert d["by"] == "rules" and d["free"] is True and d["meets"] is False


def test_judge_caches_and_stops_after_repeated_failures(monkeypatch):
    calls = []

    async def ok(http, s, attrs):
        calls.append(attrs["id"])
        return {"free": 0.9, "meets": 0.9, "image_in": 0.1}

    monkeypatch.setattr(jev, "ask", ok)
    cache = {}
    entries = [_entry("a"), _entry("b")]
    v, c = asyncio.run(jev.judge(None, entries, JS, cache))
    assert set(v) == {"a", "b"} and c["asked"] == 2 and len(cache) == 2
    v, c = asyncio.run(jev.judge(None, entries, JS, cache))
    assert c == {"asked": 0, "cached": 2, "failed": 0, "skipped": 0} and len(calls) == 2

    async def down(http, s, attrs):
        return None

    monkeypatch.setattr(jev, "ask", down)
    many = [_entry(f"m{i}") for i in range(20)]
    v, c = asyncio.run(jev.judge(None, many, {**JS, "concurrency": 1, "max_failures": 3}, {}))
    assert v == {} and c["failed"] == 3 and c["skipped"] == 17


def test_rank_uses_jev_verdicts(monkeypatch, tmp_path):
    cat = [_entry("stealth/bunny", ins=("text", "image")),       # JEV: free, meets
           _entry("vendor/looks-free"),                          # JEV: paid
           _entry("vendor/painter:free", outs=("image",)),       # JEV: fails requirement
           _entry("vendor/unjudged:free")]                       # JEV unavailable -> rules
    answers = {"stealth/bunny": {"free": 0.95, "meets": 0.8, "image_in": 0.99},
               "vendor/looks-free": {"free": 0.1, "meets": 0.9, "image_in": 0.0},
               "vendor/painter:free": {"free": 0.97, "meets": 0.1, "image_in": 0.0}}

    async def fake_ask(http, s, attrs):
        return answers.get(attrs["id"])

    async def fake_catalogue(http, endpoint, key):
        return cat

    async def fake_probe(http, endpoint, key, mid, s):
        return 5, 10.0, 1.0

    async def fake_tools(http, endpoint, key, mid, s):
        return 2

    monkeypatch.setattr(jev, "ask", fake_ask)
    monkeypatch.setattr(ranking, "_catalogue", fake_catalogue)
    monkeypatch.setattr(ranking, "_probe", fake_probe)
    monkeypatch.setattr(ranking, "_probe_tools", fake_tools)
    raw = {"ranking": {"provider_order": [OR], "jev": {"api_key": "k"}}}
    cache_path = str(tmp_path / "jev.json")
    ranked = asyncio.run(ranking.rank(None, raw, {}, jev_cache_path=cache_path))
    rows = {r["id"]: r for r in ranked[OR]}
    assert set(rows) == {"stealth/bunny", "vendor/unjudged:free"}, rows
    assert rows["stealth/bunny"]["vision"] and rows["stealth/bunny"]["judged_by"] == "jev"
    assert rows["vendor/unjudged:free"]["judged_by"] == "rules"
    assert len(jev.cache_load(cache_path)) == 3


def test_jev_vision_verdict_reaches_a_new_model(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump({
        "router": {"listen": "127.0.0.1:8800"}, "ranking": {"provider_order": [OR]},
        "models": {"paid": {"id": "vendor/paid", "endpoint": OR, "context_window": 1}},
        "roles": {"chat": {"cascade": ["paid"]}}}))
    row = {"id": "stealth/bunny", "score": 5, "tok_s": 1.0, "latency": 1.0,
           "context": 1000000, "tools": 2, "zero_priced": True, "vision": True}
    ranking.apply_ranking(str(p), {OR: [row]}, yaml.safe_load(open(p)))
    out = yaml.safe_load(open(p))
    first = out["models"][out["roles"]["chat"]["cascade"][0]]
    assert first["id"] == "stealth/bunny" and "vision" in first["capabilities"]
    assert first["free"] is True
