"""Stand-in for the customer's `gdm_binary_probability.GdmBinaryProbabilityMiddleware`.

The customer's module was not shared, so this is a best guess at what it does, built from
their description: "The model generates exactly one token, limited to Safe (39316) or
Un (2805). Our score is p_unsafe = sigmoid(logP(Un) - logP(Safe))", served with
`--max-logprobs=20`.

Pure ASGI middleware, loaded with `--middleware=gdm_binary_probability.GdmBinaryProbabilityMiddleware`
(vLLM calls `app.add_middleware(cls)`, so the class takes only `app`). For POST requests to
/v1/chat/completions and /v1/completions it:

  1. buffers and parses the request JSON and fills in the one-token classification settings
     if the client left them out (1 output token, logprobs on, top-20, allowed_token_ids =
     [Safe, Un], tokens returned as ids);
  2. buffers the JSON response and replaces it with the binary score:
     {"id", "p_unsafe", "logp_un", "logp_safe", "top1_token_id", "both_in_topk", "usage"}.

Streaming requests, non-200 responses and non-JSON responses pass through unchanged, and so
does any request carrying the header `x-gdm-bypass: 1` (used by the benchmark to measure the
middleware's cost without restarting the server). Nothing from the request or the response
is logged.

Environment knobs: GDM_MW_PATHS (comma list), GDM_MW_INJECT (1/0), GDM_MW_TOPK (20).
"""
import json
import math
import os

SAFE_ID, UN_ID = 39316, 2805
PATHS = frozenset(
    p for p in os.environ.get("GDM_MW_PATHS", "/v1/chat/completions,/v1/completions").split(",") if p
)
INJECT = os.environ.get("GDM_MW_INJECT", "1") == "1"
TOPK = int(os.environ.get("GDM_MW_TOPK", "20"))
BYPASS_HEADER = b"x-gdm-bypass"


def _token_id(token):
    """Token as returned by vLLM: 'token_id:N' with return_tokens_as_token_ids, else text."""
    if not isinstance(token, str):
        return None
    if token.startswith("token_id:"):
        try:
            return int(token[9:])
        except ValueError:
            return None
    return {"Safe": SAFE_ID, "Un": UN_ID}.get(token)


def _first_token_logprobs(resp, is_chat):
    """Return (top1_id, {token_id: logprob}) for the first generated token."""
    choice = resp["choices"][0]
    lp = choice.get("logprobs") or {}
    entries = {}
    top1 = None
    if is_chat:
        content = lp.get("content") or []
        first = content[0] if content else {}
        top1 = _token_id(first.get("token"))
        for e in first.get("top_logprobs") or []:
            tid = _token_id(e.get("token"))
            if tid is not None and e.get("logprob") is not None:
                entries[tid] = float(e["logprob"])
        if top1 is not None and first.get("logprob") is not None:
            entries.setdefault(top1, float(first["logprob"]))
    else:
        tops = lp.get("top_logprobs") or []
        for tok, val in (tops[0] or {}).items() if tops else ():
            tid = _token_id(tok)
            if tid is not None and val is not None:
                entries[tid] = float(val)
        toks = lp.get("tokens") or []
        tlp = lp.get("token_logprobs") or []
        if toks:
            top1 = _token_id(toks[0])
            if top1 is not None and tlp and tlp[0] is not None:
                entries.setdefault(top1, float(tlp[0]))
    return top1, entries


def score(raw, is_chat):
    """Turn a completion response body into the binary-probability body."""
    try:
        resp = json.loads(raw)
        top1, entries = _first_token_logprobs(resp, is_chat)
    except (ValueError, KeyError, IndexError, TypeError):
        return raw  # not something we understand; hand it back untouched
    lu, ls = entries.get(UN_ID), entries.get(SAFE_ID)
    if lu is not None and ls is not None:
        p = 1.0 / (1.0 + math.exp(-(lu - ls)))
    elif lu is not None:
        p = 1.0  # Safe fell out of the top-k: it is far below Un
    elif ls is not None:
        p = 0.0
    else:
        p = None
    out = {
        "id": resp.get("id"),
        "p_unsafe": p,
        "logp_un": lu,
        "logp_safe": ls,
        "top1_token_id": top1,
        "both_in_topk": lu is not None and ls is not None,
        "usage": resp.get("usage"),
    }
    return json.dumps(out, separators=(",", ":")).encode("utf-8")


def inject_defaults(body, is_chat):
    """Fill in the classification settings the client left out. Returns (body, is_stream)."""
    try:
        req = json.loads(body)
    except ValueError:
        return body, False
    if not isinstance(req, dict):
        return body, False
    if req.get("stream"):
        return body, True
    if is_chat:
        if "max_tokens" not in req and "max_completion_tokens" not in req:
            req["max_completion_tokens"] = 1
        req["logprobs"] = True
        req.setdefault("top_logprobs", TOPK)
    else:
        req.setdefault("max_tokens", 1)
        if not req.get("logprobs"):
            req["logprobs"] = TOPK
    # GDM_NO_ALLOWED_TOKEN_IDS=1: never send allowed_token_ids (tpu-inference's sampler path for it
    # crashes the engine: input_batch.py:451 "zeros() got multiple values for argument 'dtype'").
    # The score only needs logP(Un) and logP(Safe) from the top-k, which the mask does not change.
    if os.environ.get("GDM_NO_ALLOWED_TOKEN_IDS") == "1":
        req.pop("allowed_token_ids", None)
    else:
        req.setdefault("allowed_token_ids", [SAFE_ID, UN_ID])
    req.setdefault("return_tokens_as_token_ids", True)
    return json.dumps(req, separators=(",", ":")).encode("utf-8"), False


async def _read_body(receive):
    parts = []
    while True:
        msg = await receive()
        if msg["type"] == "http.disconnect":
            return b"".join(parts)
        parts.append(msg.get("body", b""))
        if not msg.get("more_body"):
            return b"".join(parts)


def _replay(body, receive):
    """Hand the buffered body to the app once, then defer to the real receive().

    vLLM watches receive() for http.disconnect to cancel abandoned requests, so after the
    body we must block on the real channel rather than fabricate a disconnect.
    """
    done = False

    async def _recv():
        nonlocal done
        if not done:
            done = True
            return {"type": "http.request", "body": body, "more_body": False}
        return await receive()

    return _recv


def _with_content_length(scope, n):
    headers = [(k, v) for k, v in scope.get("headers", []) if k.lower() != b"content-length"]
    headers.append((b"content-length", str(n).encode("ascii")))
    new_scope = dict(scope)
    new_scope["headers"] = headers
    return new_scope


def _bypass(scope):
    for k, v in scope.get("headers", []):
        if k.lower() == BYPASS_HEADER:
            return v.strip() in (b"1", b"true", b"yes")
    return False


class GdmBinaryProbabilityMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") not in PATHS
            or _bypass(scope)
        ):
            await self.app(scope, receive, send)
            return

        is_chat = scope["path"].endswith("/chat/completions")
        body = await _read_body(receive)
        streaming = False
        if INJECT:
            new_body, streaming = inject_defaults(body, is_chat)
            if new_body is not body:
                body = new_body
                scope = _with_content_length(scope, len(body))
        if streaming:
            await self.app(scope, _replay(body, receive), send)
            return

        start_msg = None
        chunks = []
        passthrough = False

        async def send_wrapper(msg):
            nonlocal start_msg, passthrough
            if passthrough:
                await send(msg)
                return
            mtype = msg["type"]
            if mtype == "http.response.start":
                ctype = b""
                for k, v in msg.get("headers", []):
                    if k.lower() == b"content-type":
                        ctype = v
                        break
                if msg.get("status") != 200 or not ctype.startswith(b"application/json"):
                    passthrough = True
                    await send(msg)
                    return
                start_msg = msg
                return
            if mtype == "http.response.body":
                chunks.append(msg.get("body", b""))
                if msg.get("more_body"):
                    return
                out = score(b"".join(chunks), is_chat)
                headers = [
                    (k, v) for k, v in start_msg.get("headers", []) if k.lower() != b"content-length"
                ]
                headers.append((b"content-length", str(len(out)).encode("ascii")))
                await send({"type": "http.response.start", "status": 200, "headers": headers})
                await send({"type": "http.response.body", "body": out, "more_body": False})
                return
            await send(msg)

        await self.app(scope, _replay(body, receive), send_wrapper)
