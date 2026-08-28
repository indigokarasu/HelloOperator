"""CLI entry point: serve (default), --check, --discover.

--check   validates without serving (FR-10)
--discover emits a proposed config from detection without serving (3.1a)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

import aiohttp
from aiohttp import web

from . import config as config_mod
from . import discovery
from .config import Config, ConfigError
from .server import build_app

log = logging.getLogger("router")


async def _check(cfg: Config, probe: bool) -> int:
    """Startup validation: endpoints respond; detection vs declaration
    disagreements surfaced with provenance. Fail loudly here, never silently
    at request time (FR-10)."""
    rc = 0
    async with aiohttp.ClientSession() as http:
        detected = await discovery.detect_all(http, cfg, probe=probe)
        # Re-resolve with detection so declared-vs-detected warnings surface.
        cfg2 = config_mod.load(cfg.source_path, detected_cache=detected)
        print(config_mod.provenance_table(cfg2))
        print()
        for w in cfg2.warnings:
            print(f"warning: {w}")
        specs = list(cfg.models.values()) + ([cfg.embedding] if cfg.embedding else [])
        seen = set()
        for spec in specs:
            if spec.endpoint in seen:
                continue
            seen.add(spec.endpoint)
            served = await discovery.list_backend_models(http, spec.endpoint)
            if served is None:
                print(f"error: endpoint {spec.endpoint} did not answer /v1/models")
                rc = 1
            else:
                print(f"ok: {spec.endpoint} answers ({len(served)} model(s) served)")
        for note in await discovery.drift_check(http, cfg):
            print(f"warning: {note}")
    print("check: FAILED" if rc else "check: OK")
    return rc


async def _discover(cfg: Config, probe: bool, out: str) -> int:
    async with aiohttp.ClientSession() as http:
        proposal = await discovery.generate_proposal(http, cfg, probe=probe)
    if out:
        with open(out, "w") as f:
            f.write(proposal)
        print(f"proposal written to {out} — review and edit; nothing routes on "
              "it until you adopt it")
    else:
        print(proposal)
    return 0


async def _pre_serve(cfg: Config) -> Config:
    """Startup detection, endpoint validation, and drift flagging.

    Returns the detection-resolved config the server ROUTES on (spec 3.1:
    declared > detected > default applies everywhere, so a model declared with
    only id+endpoint serves with the backend's real context window and
    capabilities, not the heuristic defaults). Detection failing entirely
    leaves the router fully usable on declarations alone (FR-15).

    Unreachable endpoints are reported loudly (FR-10) but do not refuse
    startup: under a service manager the backends may simply boot later
    (NFR-6), and FR-9 failover covers a dead backend per request.
    """
    try:
        async with aiohttp.ClientSession() as http:
            detected = await discovery.detect_all(http, cfg)
            cfg2 = config_mod.load(cfg.source_path, detected_cache=detected)
            for w in cfg2.warnings:
                log.warning("%s", w)
            seen = set()
            specs = list(cfg.models.values()) + ([cfg.embedding] if cfg.embedding else [])
            for spec in specs:
                if spec.endpoint in seen:
                    continue
                seen.add(spec.endpoint)
                if await discovery.list_backend_models(http, spec.endpoint) is None:
                    log.error("endpoint %s did not answer /v1/models — serving "
                              "anyway; requests to it will fail over (FR-9/FR-10)",
                              spec.endpoint)
            for note in await discovery.drift_check(http, cfg):
                log.warning("%s", note)
            return cfg2
    except Exception as e:  # noqa: BLE001 — startup detection is best-effort
        log.warning("startup detection skipped: %s", e)
        return cfg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hermes-model-router",
        description="Local-first routing layer for OpenAI-compatible models")
    ap.add_argument("-c", "--config", default="config.yaml",
                    help="path to the single config file (default: ./config.yaml)")
    ap.add_argument("--check", action="store_true",
                    help="validate config and endpoints without serving")
    ap.add_argument("--discover", action="store_true",
                    help="emit a proposed registry-and-roles config from detection")
    ap.add_argument("--probe", action="store_true",
                    help="with --check/--discover: run opt-in active probes "
                         "(forces a load on on-demand models unless disabled per model)")
    ap.add_argument("--out", default="", help="with --discover: write proposal here")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    try:
        cfg = config_mod.load(args.config)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2

    for w in cfg.warnings:
        log.warning("%s", w)

    if args.check:
        return asyncio.run(_check(cfg, probe=args.probe))
    if args.discover:
        return asyncio.run(_discover(cfg, probe=args.probe, out=args.out))

    cfg = asyncio.run(_pre_serve(cfg))
    app = build_app(cfg)
    mode = "degenerate (single model, selection bypassed)" if cfg.degenerate \
        else f"{len(cfg.models)} models, {len(cfg.roles)} roles"
    log.info("hermes-model-router serving %s on %s:%d — logical model '%s' (%s)",
             mode, cfg.settings.listen_host, cfg.settings.listen_port,
             cfg.settings.logical_model,
             "loopback" if not cfg.settings.allow_non_loopback else "NON-LOOPBACK opt-in")
    web.run_app(app, host=cfg.settings.listen_host, port=cfg.settings.listen_port,
                print=None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
