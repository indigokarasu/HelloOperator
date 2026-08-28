"""Request property extraction and capability filtering (FR-3).

Hard-filters candidates on observable request properties. History-wide for
media, because an image introduced at turn 1 remains in context at turn 20.
Zero inference cost: everything here is a linear scan of the request body.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .config import ModelSpec, Settings

IMAGE_PART_TYPES = {"image_url", "input_image", "image"}
AUDIO_PART_TYPES = {"input_audio", "audio", "audio_url"}


@dataclass
class RequestProps:
    has_image: bool
    has_audio: bool
    wants_tools: bool
    tool_choice_required: bool
    est_tokens: int
    max_tokens: int
    tools_sig: str          # stable signature of the tool set (FR-6 transition input)
    last_user_text: str     # classification input (FR-4)
    msg_count: int


def _part_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                out.append(str(part.get("text", "")))
        return "\n".join(out)
    return ""


def extract_props(body: dict, settings: Settings) -> RequestProps:
    messages = body.get("messages") or []
    has_image = has_audio = False
    last_user_text = ""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                t = part.get("type")
                if t in IMAGE_PART_TYPES:
                    has_image = True
                elif t in AUDIO_PART_TYPES:
                    has_audio = True
        if msg.get("role") == "user":
            txt = _part_text(content)
            if txt:
                last_user_text = txt

    tools = body.get("tools") or []
    tool_names = sorted(
        t.get("function", {}).get("name", "") for t in tools if isinstance(t, dict))
    tools_sig = hashlib.sha1(json.dumps(tool_names).encode()).hexdigest()[:12] if tools else ""

    tc = body.get("tool_choice")
    tool_choice_required = tc == "required" or (isinstance(tc, dict) and tc.get("type") == "function")

    # Coarse token estimate: text chars / 4 plus a flat per-media estimate.
    # Media payloads (base64 data URIs) must NOT be counted as text — a 500KB
    # image serializes to ~680KB of base64 (~170k "tokens"), which would
    # falsely eliminate every model. Vision models count an image as roughly
    # 1-2k tokens, so a flat estimate is the honest bound (FR-3).
    est_chars = 0
    media_parts = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            est_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                t = part.get("type")
                if t in IMAGE_PART_TYPES or t in AUDIO_PART_TYPES:
                    media_parts += 1
                elif t == "text":
                    est_chars += len(str(part.get("text", "")))
                else:
                    try:
                        est_chars += len(json.dumps(part, ensure_ascii=False))
                    except (TypeError, ValueError):
                        pass
        est_chars += 32  # role/framing overhead per message
    est = est_chars // 4 + media_parts * settings.media_token_estimate
    if tools:
        try:
            est += len(json.dumps(tools, ensure_ascii=False)) // 4
        except (TypeError, ValueError):
            pass

    max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") \
        or settings.assumed_completion_tokens
    try:
        max_tokens = int(max_tokens)
    except (TypeError, ValueError):
        max_tokens = settings.assumed_completion_tokens

    return RequestProps(
        has_image=has_image, has_audio=has_audio,
        wants_tools=bool(tools), tool_choice_required=tool_choice_required,
        est_tokens=est, max_tokens=max_tokens, tools_sig=tools_sig,
        last_user_text=last_user_text, msg_count=len(messages))


def model_ok(spec: ModelSpec, props: RequestProps, settings: Settings) -> tuple[bool, str]:
    """(eligible, reason-if-not). Reasons are surfaced verbatim in FR-3/FR-9
    errors, so they name the unsatisfiable constraint."""
    if props.has_image and "vision" not in spec.capabilities:
        return False, f"request contains image content but model '{spec.key}' does not declare vision"
    if props.has_audio and "audio" not in spec.capabilities:
        return False, f"request contains audio content but model '{spec.key}' does not declare audio"
    if props.wants_tools and "tools" not in spec.capabilities:
        return False, f"request supplies tool definitions but model '{spec.key}' does not declare tools"
    needed = props.est_tokens + props.max_tokens + settings.context_safety_margin
    if needed > spec.context_window:
        return False, (f"estimated {props.est_tokens} prompt tokens + {props.max_tokens} "
                       f"completion tokens exceed model '{spec.key}' context window "
                       f"of {spec.context_window}")
    return True, ""
