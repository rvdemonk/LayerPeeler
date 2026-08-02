"""One judge call per provider, normalised to the same return shape.

Each adapter returns (parsed_dict, raw_response_dict, usage_dict). The raw
response is persisted verbatim by the runner — the responses ARE the corpus, and
a bench that keeps only its own summary of them cannot be re-analysed.

MODELS is resolved against each provider's live model list, not from memory. A
substitution is recorded in the `note` field rather than silently upgraded.
"""

import base64
import json
import os
from pathlib import Path

import prompt as P

# Load the research-root .env and ~/.env so API keys are available without
# manual sourcing. Only sets vars that are not already in the environment.
for _env_path in [Path(__file__).resolve().parents[3] / ".env",
                  Path.home() / ".env"]:
    if _env_path.exists():
        for line in _env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip("\"'")
                if k and k not in os.environ:
                    os.environ[k] = v

# Hard ceiling: bounds worst-case output cost exactly rather than forecasting it.
# 1200 was set at checkpoint 1 to bring the four-model worst case ($3.73 at 2000)
# inside the $3.50 approved envelope, at $2.74. These are thinking models and
# thinking bills as output, so a long deliberation can truncate — a truncated
# response fails to parse and is scored UNUSABLE, never as a PASS. Truncation can
# only cost a model points; it cannot manufacture the false PASS that rules §5.2.
MAX_OUTPUT = 1200

# Resolved 2026-08-02 against each provider's model-list endpoint.
MODELS = {
    "opus-5": {
        "vendor": "anthropic", "id": "claude-opus-5",
        "in": 5.00, "out": 25.00, "note": "",
    },
    "sonnet-5": {
        "vendor": "anthropic", "id": "claude-sonnet-5",
        "in": 2.00, "out": 10.00,
        "note": "intro pricing to 2026-08-31; list $3/$15",
    },
    "haiku-4.5": {
        "vendor": "anthropic", "id": "claude-haiku-4-5-20251001",
        "in": 1.00, "out": 5.00,
        "note": "dated id — the only Haiku 4.5 entry on /v1/models",
    },
    "gemini-3.1-pro": {
        "vendor": "gemini", "id": "gemini-3.1-pro-preview",
        "in": 2.00, "out": 12.00,
        "note": "SUBSTITUTION: spec said 'Gemini 3.1 Pro'; the only 3.1 Pro id "
                "the API lists is the -preview one. Same tier, not an upgrade.",
    },
    "gemini-3.6-flash": {
        "vendor": "gemini", "id": "gemini-3.6-flash",
        "in": 1.50, "out": 7.50, "note": "",
    },
    "gpt-5": {
        "vendor": "openai", "id": "gpt-5",
        "in": 1.25, "out": 10.00, "note": "",
    },
    "gpt-5-mini": {
        "vendor": "openai", "id": "gpt-5-mini",
        "in": 0.25, "out": 2.00, "note": "",
    },
    "deepseek-v4-flash": {
        "vendor": "deepseek", "id": "deepseek-v4-flash",
        "in": 0.14, "out": 0.28, "note": "",
    },
}


def _b64(path):
    return base64.standard_b64encode(path.read_bytes()).decode()


# ---------------------------------------------------------------- anthropic --

def _anthropic(model, images, client):
    content = [{"type": "text", "text": P.USER_PREFIX}]
    for slot, frame, path in images:
        content.append({"type": "text", "text": P.image_caption(slot, frame)})
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": _b64(path)}})
    r = client.messages.create(
        model=model["id"],
        max_tokens=MAX_OUTPUT,
        system=P.SYSTEM,
        messages=[{"role": "user", "content": content}],
        output_config={"format": {"type": "json_schema", "schema": P.SCHEMA}},
    )
    raw = r.model_dump(mode="json")
    if r.stop_reason == "refusal":
        return None, raw, _usage_anthropic(r)
    text = next((b.text for b in r.content if b.type == "text"), None)
    return (json.loads(text) if text else None), raw, _usage_anthropic(r)


def _usage_anthropic(r):
    u = r.usage
    return {"input": u.input_tokens, "output": u.output_tokens,
            "stop_reason": r.stop_reason}


# ------------------------------------------------------------------- openai --

def _openai(model, images, client):
    content = [{"type": "input_text", "text": P.USER_PREFIX}]
    for slot, frame, path in images:
        content.append({"type": "input_text",
                        "text": P.image_caption(slot, frame)})
        content.append({"type": "input_image",
                        "image_url": "data:image/png;base64," + _b64(path)})
    r = client.responses.create(
        model=model["id"],
        max_output_tokens=MAX_OUTPUT,
        instructions=P.SYSTEM,
        input=[{"role": "user", "content": content}],
        text={"format": {"type": "json_schema", "name": "identity_judgement",
                         "strict": True, "schema": P.SCHEMA}},
    )
    raw = r.model_dump(mode="json")
    txt = getattr(r, "output_text", None)
    return (json.loads(txt) if txt else None), raw, {
        "input": r.usage.input_tokens, "output": r.usage.output_tokens,
        "stop_reason": r.status,
    }


# ------------------------------------------------------------------- gemini --

def _gemini(model, images, client):
    from google.genai import types
    parts = [types.Part.from_text(text=P.USER_PREFIX)]
    for slot, frame, path in images:
        parts.append(types.Part.from_text(text=P.image_caption(slot, frame)))
        parts.append(types.Part.from_bytes(data=path.read_bytes(),
                                           mime_type="image/png"))
    r = client.models.generate_content(
        model=model["id"],
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(
            system_instruction=P.SYSTEM,
            max_output_tokens=MAX_OUTPUT,
            response_mime_type="application/json",
            response_json_schema=P.SCHEMA,
        ),
    )
    raw = r.model_dump(mode="json")
    u = r.usage_metadata
    usage = {"input": u.prompt_token_count or 0,
             "output": (u.candidates_token_count or 0)
             + (u.thoughts_token_count or 0),
             "stop_reason": str(r.candidates[0].finish_reason)
             if r.candidates else None}
    try:
        return json.loads(r.text), raw, usage
    except Exception:
        return None, raw, usage


# ---------------------------------------------------------------- deepseek --

def _deepseek(model, images, client):
    content = [{"type": "text", "text": P.USER_PREFIX}]
    for slot, frame, path in images:
        content.append({"type": "text", "text": P.image_caption(slot, frame)})
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + _b64(path)}})
    r = client.chat.completions.create(
        model=model["id"],
        max_tokens=MAX_OUTPUT,
        messages=[
            {"role": "system", "content": P.SYSTEM},
            {"role": "user", "content": content},
        ],
        response_format={"type": "json_object"},
        extra_body={"thinking": {"type": "disabled"}},
    )
    raw = r.model_dump(mode="json")
    u = r.usage
    usage = {"input": u.prompt_tokens,
             "output": u.completion_tokens,
             "stop_reason": r.choices[0].finish_reason if r.choices else None}
    try:
        return json.loads(r.choices[0].message.content), raw, usage
    except Exception:
        return None, raw, usage


DISPATCH = {"anthropic": _anthropic, "openai": _openai, "gemini": _gemini,
            "deepseek": _deepseek}


def make_clients():
    import anthropic
    import openai
    from google import genai
    return {
        "anthropic": anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"]),
        "openai": openai.OpenAI(
            api_key=os.environ["OPENAI_API_SECRET_KEY"]),
        "gemini": genai.Client(
            api_key=os.environ["GEMINI_API_SECRET_KEY"]),
        "deepseek": openai.OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com"),
    }


def judge(model_key, images, clients):
    m = MODELS[model_key]
    return DISPATCH[m["vendor"]](m, images, clients[m["vendor"]])


# Pairwise variant — uses pairwise_prompt instead of the full prompt.
# Same dispatch, different images (only 2: anchor + comparison).
PAIRWISE_DISPATCH = {}

import pairwise_prompt as PP  # noqa: E402


def _pairwise_anthropic(model, images, client):
    content = [{"type": "text", "text": PP.USER_PREFIX}]
    for slot, frame, path in images:
        content.append(
            {"type": "text", "text": PP.image_caption(slot, frame)})
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/png",
            "data": _b64(path)}})
    r = client.messages.create(
        model=model["id"],
        max_tokens=MAX_OUTPUT,
        system=PP.SYSTEM,
        messages=[{"role": "user", "content": content}],
        output_config={"format": {"type": "json_schema",
                                   "schema": PP.SCHEMA}},
    )
    raw = r.model_dump(mode="json")
    if r.stop_reason == "refusal":
        return None, raw, _usage_anthropic(r)
    text = next((b.text for b in r.content if b.type == "text"), None)
    return (json.loads(text) if text else None), raw, _usage_anthropic(r)


def _pairwise_openai(model, images, client):
    content = [{"type": "input_text", "text": PP.USER_PREFIX}]
    for slot, frame, path in images:
        content.append({"type": "input_text",
                        "text": PP.image_caption(slot, frame)})
        content.append({"type": "input_image",
                        "image_url": "data:image/png;base64," + _b64(path)})
    r = client.responses.create(
        model=model["id"],
        max_output_tokens=MAX_OUTPUT,
        instructions=PP.SYSTEM,
        input=[{"role": "user", "content": content}],
        text={"format": {"type": "json_schema", "name": "identity_pair",
                         "strict": True, "schema": PP.SCHEMA}},
    )
    raw = r.model_dump(mode="json")
    txt = getattr(r, "output_text", None)
    return (json.loads(txt) if txt else None), raw, {
        "input": r.usage.input_tokens, "output": r.usage.output_tokens,
        "stop_reason": r.status,
    }


def _pairwise_gemini(model, images, client):
    from google.genai import types
    parts = [types.Part.from_text(text=PP.USER_PREFIX)]
    for slot, frame, path in images:
        parts.append(
            types.Part.from_text(text=PP.image_caption(slot, frame)))
        parts.append(types.Part.from_bytes(data=path.read_bytes(),
                                           mime_type="image/png"))
    r = client.models.generate_content(
        model=model["id"],
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(
            system_instruction=PP.SYSTEM,
            max_output_tokens=MAX_OUTPUT,
            response_mime_type="application/json",
            response_json_schema=PP.SCHEMA,
        ),
    )
    raw = r.model_dump(mode="json")
    u = r.usage_metadata
    usage = {"input": u.prompt_token_count or 0,
             "output": (u.candidates_token_count or 0)
             + (u.thoughts_token_count or 0),
             "stop_reason": str(r.candidates[0].finish_reason)
             if r.candidates else None}
    try:
        return json.loads(r.text), raw, usage
    except Exception:
        return None, raw, usage


def _pairwise_deepseek(model, images, client):
    content = [{"type": "text", "text": PP.USER_PREFIX}]
    for slot, frame, path in images:
        content.append(
            {"type": "text", "text": PP.image_caption(slot, frame)})
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/png;base64,"
                                      + _b64(path)}})
    r = client.chat.completions.create(
        model=model["id"],
        max_tokens=MAX_OUTPUT,
        messages=[
            {"role": "system", "content": PP.SYSTEM},
            {"role": "user", "content": content},
        ],
        response_format={"type": "json_object"},
        extra_body={"thinking": {"type": "disabled"}},
    )
    raw = r.model_dump(mode="json")
    u = r.usage
    usage = {"input": u.prompt_tokens,
             "output": u.completion_tokens,
             "stop_reason": (r.choices[0].finish_reason
                             if r.choices else None)}
    try:
        return json.loads(r.choices[0].message.content), raw, usage
    except Exception:
        return None, raw, usage


PAIRWISE_DISPATCH = {
    "anthropic": _pairwise_anthropic,
    "openai": _pairwise_openai,
    "gemini": _pairwise_gemini,
    "deepseek": _pairwise_deepseek,
}


def pairwise_judge(model_key, images, clients):
    m = MODELS[model_key]
    return PAIRWISE_DISPATCH[m["vendor"]](m, images, clients[m["vendor"]])


def cost(model_key, usage):
    m = MODELS[model_key]
    return (usage["input"] / 1e6 * m["in"]
            + usage["output"] / 1e6 * m["out"])
