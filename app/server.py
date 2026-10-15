"""Jarvis HTTP API.

Bound to loopback (127.0.0.1) by default and authenticated with a bearer
token. Remote access is meant to arrive through Tailscale Serve terminating
TLS in front of this process, not by binding a wider interface.

Every /api/* route requires `Authorization: Bearer $JARVIS_TOKEN`. There is
deliberately no CORS middleware: the PWA is same-origin, so cross-origin
access is not needed and would only widen exposure of personal data.
"""
from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from contextlib import asynccontextmanager
from typing import Literal, Optional
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from app.core import (
    auth, config, pwa, storage, utils, search as search_mod, recall,
    personal_context, context_engine, memory_candidates, memory_judge, interaction_policy, conversation_continuity, context_topics,
    web,
)
from app.providers.llm import OllamaCompatible


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Installed here rather than at import so it runs after uvicorn has set up
    # its logging, and it applies to every process that serves the API.
    auth.install_access_log_redaction()
    # uvicorn configures only its own loggers and leaves application logs below
    # WARNING unhandled, so bounded web-retrieval diagnostics would be dropped.
    # Attach one stderr handler to the "app" namespace only -- this keeps
    # application INFO logs visible without enabling noisy third-party INFO.
    app_logger = logging.getLogger("app")
    if not app_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s - %(message)s"))
        app_logger.addHandler(handler)
    app_logger.setLevel(logging.INFO)
    yield


app = FastAPI(
    title="Jarvis API",
    version="0.2.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
BASE = config.BASE
_log = logging.getLogger(__name__)


def _cfg():
    """Legacy accessor retained for the config dict shape used by callers."""
    return config.load_config()


def _paths(cfg: dict):
    raw = config.raw_dir()
    dbp = config.db_path()
    raw.mkdir(parents=True, exist_ok=True)
    dbp.parent.mkdir(parents=True, exist_ok=True)
    return raw, dbp


@app.middleware("http")
async def enforce_api_auth(request: Request, call_next):
    """Gate /api/* before routing, so no route can be added unprotected by
    accident. HTTPException raised here is outside the route context and so
    bypasses FastAPI's handlers; it is converted to a response directly."""
    if auth.is_api_path(request.url.path):
        try:
            auth.require_token(request)
        except HTTPException as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
                headers=exc.headers,
            )
    return await call_next(request)


class AddReq(BaseModel):
    text: str
    source: str = "cli"
    origin_kind: str = "self"


class ChatTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatReq(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=4000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=8)


def _load_chat_inputs(message: str, policy: interaction_policy.InteractionPolicy, history=None):
    """Load only the context classes requested by the deterministic policy."""
    cfg = _cfg()
    _, dbp = _paths(cfg)
    try:
        db = storage.JarvisDB(dbp)
    except Exception:
        # A local context-store failure must not prevent a general chat answer.
        # Do not log exception strings: sqlite errors may contain local paths.
        _log.warning("Local context database unavailable; continuing without stored context")
        return ([], "", "unavailable" if policy.propose_memory_candidate else None,
                None, None, False, [], web.WebRetrieval([]))
    try:
        evidence = []
        context = ""
        web_result = web.WebRetrieval([])
        memory_action = None
        continuity_thread = None
        continuity_score = 0.0
        continuity_siblings = []
        continuity_operation = None
        continuity_update = False
        directive = conversation_continuity.directive(message)
        explicit_restore = directive in {"where", "resume"}
        try:
            repo = conversation_continuity.ContinuityRepository(db)
            if directive == "where":
                continuity_thread, continuity_score = repo.find_relevant(message, explicit="where")
            elif directive in {"pause", "resolve", "forget"}:
                continuity_thread, _ = repo.find_relevant(
                    message, explicit="active" if directive == "pause" else "current"
                )
                if continuity_thread:
                    if directive == "pause":
                        repo.pause(continuity_thread["thread_id"], continuity_thread["revision"])
                        continuity_operation = "paused"
                    elif directive == "resolve":
                        repo.resolve(continuity_thread["thread_id"], continuity_thread["revision"])
                        continuity_operation = "resolved"
                    else:
                        repo.forget(continuity_thread["thread_id"], continuity_thread["revision"])
                        continuity_operation = "forgotten"
                    continuity_thread = None
            elif directive == "resume":
                continuity_thread, continuity_score = repo.find_relevant(message, explicit="resume")
                if continuity_thread and continuity_thread["status"] == "PAUSED":
                    continuity_thread = repo.resume(continuity_thread["thread_id"], continuity_thread["revision"])
                    continuity_operation = "resumed"
                elif continuity_thread and continuity_thread["status"] == "ACTIVE":
                    continuity_thread = repo.update_snapshot(
                        continuity_thread["thread_id"], continuity_thread["revision"], {
                            key: continuity_thread[key] for key in (
                                "topic", "current_problem_or_goal", "decisions", "open_questions",
                                "temporary_commitments", "expires_at",
                            )
                        },
                    )
                    continuity_operation = "resumed"
                elif continuity_thread is None:
                    continuity_operation = "not_found"
            elif directive == "temporary":
                continuity_thread, _ = repo.find_relevant(message, explicit="current")
                continuity_thread = conversation_continuity.add_temporary_commitment(db, continuity_thread, message)
                if continuity_thread:
                    continuity_operation = "temporary_saved"
                else:
                    continuity_operation = "not_found"
            else:
                continuity_thread, continuity_score = repo.find_relevant(
                    message, explicit="continue" if directive == "continue" else None
                )
                continuity_update = bool(continuity_thread or directive == "start")
            if directive == "start":
                continuity_update = True
            if directive in {"pause", "resolve", "forget"} and continuity_operation is None:
                continuity_operation = "not_found"
        except Exception:
            _log.warning("Conversation continuity unavailable; continuing without it")
            continuity_thread = None
            continuity_operation = None
            continuity_update = False
            continuity_score = 0.0
        if continuity_thread and directive not in {"temporary", "pause", "resolve", "forget"}:
            try:
                continuity_siblings = context_topics.relevant_sibling_threads(
                    db, continuity_thread, message,
                    primary_match_score=continuity_score,
                    explicit_restore=explicit_restore,
                )
            except Exception:
                # Topic-derived continuity is optional; retain the selected
                # thread if this additional lookup fails.
                _log.warning("Conversation topic context unavailable; continuing with matched thread")
                continuity_siblings = []
        if policy.retrieve_evidence:
            if policy.email_itinerary:
                # Private itinerary lookup: retrieve booking confirmations and
                # updates directly, newest first, instead of generic keyword recall
                # that a terse booking email would never surface.
                evidence = recall.retrieve_external_itinerary(db, k=5)
            else:
                recall_options = {"include_external": policy.retrieve_external_evidence}
                if policy.retrieve_external_evidence:
                    recall_options.update({"external_provider": "gmail", "external_only": True})
                    date_window = interaction_policy.gmail_date_window(message)
                    if date_window:
                        recall_options["external_date_start"] = date_window["start_utc"]
                        recall_options["external_date_end"] = date_window["end_utc"]
                    sender = interaction_policy.gmail_sender_constraint(message)
                    if sender:
                        recall_options["external_sender"] = sender
                evidence = recall.recall(
                    db, message, k=5, **recall_options,
                )
        if policy.retrieve_personal_context:
            broad = policy.personal_context_mode == "broad"
            try:
                if config.context_engine_enabled():
                    package = context_engine.assemble(
                        db, message,
                        evidence=evidence,
                        external_only=policy.retrieve_external_evidence,
                        broad=broad,
                    )
                    evidence = package.evidence
                    context = package.render_personal_lane()
                elif broad:
                    context = personal_context.build_personal_overview(db)
                else:
                    context = personal_context.build_personal_context(db, message)
            except Exception:
                # Memory context is an optional enhancement. Keep diagnostics
                # non-sensitive: never log claims, prompts, or exception strings.
                _log.warning("Personal context retrieval failed; continuing without it")
                context = ""
        if policy.web_needed and config.web_enabled():
            started = time.monotonic()
            try:
                # Public pages only; no personal data leaves the device. The raw
                # message is normalized into a concise query (or a contained URL
                # is fetched directly) rather than sent verbatim. Any failure
                # degrades to no evidence, and the caller supplies a deterministic
                # "cannot verify" line so the answer stays honest either way.
                web_result = web.retrieve_for_message(
                    message, max_items=config.web_max_results()
                )
            except Exception:
                _log.warning("Web retrieval failed; continuing without live sources")
                web_result = web.WebRetrieval([], status="error")
            # Bounded, non-sensitive diagnostics: provider, outcome category,
            # result count and duration. The query and page text are never logged.
            # Failures log at WARNING so they reach the default log even without an
            # explicit logging configuration; successes are informational.
            _level = (logging.INFO if web_result.status in {"ok", "url"}
                      else logging.WARNING)
            _log.log(
                _level,
                "web retrieval provider=%s status=%s sources=%d duration_ms=%d",
                config.web_search_provider(), web_result.status,
                len(web_result.evidence), int((time.monotonic() - started) * 1000),
            )
        if policy.propose_memory_candidate and directive != "temporary":
            try:
                assessments = memory_candidates.extract_conversation_candidates(
                    db, [{"role": "user", "content": interaction_policy.candidate_text(message)}]
                )
                if any(item.get("candidate") for item in assessments):
                    memory_action = "candidate"
                elif any(item.get("relationship") == "DUPLICATE" for item in assessments):
                    memory_action = "already_present"
                else:
                    memory_action = "not_created"
            except Exception:
                # Explicit candidate capture is best-effort and never blocks chat.
                _log.warning("Explicit memory candidate processing failed")
                memory_action = "unavailable"
        if config.memory_candidates_enabled() and not policy.propose_memory_candidate and directive != "temporary":
            try:
                # Silent, deterministic background producer: only explicit, strong
                # statements become unconfirmed CANDIDATE records. It never alters
                # the reply and never confirms anything.
                memory_candidates.extract_deterministic_candidates(
                    db, [{"role": "user", "content": message}]
                )
            except Exception:
                # Background candidate capture is best-effort and never blocks chat.
                _log.warning("Background memory candidate processing failed")
        if config.memory_judge_enabled() and config.memory_judge_apply():
            try:
                # Gated Memory Judge: decides pending candidates through the
                # existing lifecycle and records decisions under judge/v1. Both
                # flags default to off, so live memory is never mutated unless
                # explicitly opted in.
                memory_judge.judge_candidates(db, apply=True)
            except Exception:
                _log.warning("Memory judge processing failed")
        return (evidence, context, memory_action, continuity_thread,
                continuity_operation, continuity_update, continuity_siblings,
                web_result)
    finally:
        db.close()


def _sse(event: str, payload: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def _public_evidence(items: list[dict]) -> list[dict]:
    """Only return citation/provenance fields needed by the UI; omit paths."""
    fields = ("id", "date", "status", "facet_status", "attribution", "source",
              "external_provider", "external_author",
              "category", "importance", "text")
    return [{key: item.get(key) for key in fields} for item in items]


@app.post("/api/chat")
async def chat(req: ChatReq, request: Request):
    if not req.message.strip():
        raise HTTPException(422, "message must not be blank")

    policy = interaction_policy.classify(req.message)
    history = [turn.model_dump() for turn in req.history]
    (evidence, personal, memory_action, continuity_thread, continuity_operation,
     continuity_update, continuity_siblings, web_result) = await run_in_threadpool(
        _load_chat_inputs, req.message, policy, history
    )
    web_evidence = web_result.evidence
    cfg = dict(_cfg())
    llm_cfg = dict(cfg.get("llm") or {})
    # Honor the existing environment-aware config accessor, then use the shared
    # provider factory for model, temperature, token and timeout settings.
    llm_base_url = config.ollama_base_url()
    if (urlsplit(llm_base_url).hostname or "").lower() not in {"localhost", "127.0.0.1", "::1"}:
        raise HTTPException(503, "Chat provider must be local to this device")
    llm_cfg["base_url"] = llm_base_url
    cfg["llm"] = llm_cfg
    provider = OllamaCompatible.from_config(cfg)
    messages = recall.answer_prompt(
        req.message, evidence, history,
        include_empty_evidence=False,
    )
    policy_content = policy.guidance
    if policy.kind == interaction_policy.InteractionKind.MEMORY_QUERY and not personal and not evidence:
        policy_content += " No matching personal context or source records were retrieved; say so once, without guessing."
    elif policy.kind == interaction_policy.InteractionKind.PERSONAL_QUERY and not personal and not evidence:
        policy_content += (
            " No personal context or stored records were retrieved for this user; say plainly that no local "
            "record is available, and do not claim the underlying model remembers them."
        )
    elif policy.retrieve_external_evidence and not evidence:
        policy_content += " No matching external records were retrieved; say so once, without guessing."
    if memory_action == "candidate":
        policy_content += (
            " The explicit request created an unconfirmed memory candidate. Acknowledge briefly; "
            "do not say it is permanently saved or confirmed."
        )
    elif memory_action == "already_present":
        policy_content += " An equivalent confirmed memory already exists; acknowledge briefly without reciting it."
    elif memory_action == "not_created":
        policy_content += (
            " No candidate was created, so do not claim the information was saved. If needed, ask one "
            "concise clarification about what durable detail the user wants remembered."
        )
    elif memory_action == "unavailable":
        policy_content += " Memory candidate processing was unavailable; do not claim the information was saved."
    if conversation_continuity.directive(req.message) == "where" and not continuity_thread:
        policy_content += " No ongoing conversation thread was available to restore; say so briefly without guessing."
    if continuity_operation:
        if continuity_operation == "not_found":
            policy_content += " No matching conversation thread was found; say so briefly and do not claim the action succeeded."
        else:
            policy_content += (
                f" The requested conversation-continuity action completed ({continuity_operation}); "
                "acknowledge it briefly and do not claim a reminder or notification was scheduled."
            )
    messages.insert(0, {"role": "system", "content": policy_content})
    continuity_message = conversation_continuity.format_context(
        continuity_thread, related_threads=continuity_siblings
    )
    if continuity_message:
        messages.insert(1, {"role": "system", "content": continuity_message})
    personal_message = personal_context.as_system_message(personal)
    if personal_message is not None:
        # Keep canonical personal context separate from source evidence and
        # place its guidance before the conversation/evidence messages.
        messages.insert(0, personal_message)
    web_message = web.format_web_context(
        web_evidence, transport=interaction_policy.is_transport_query(req.message)
    )
    if web_message:
        # Append last so the current sources sit closest to generation, the same
        # position recall uses for evidence; small local models weight it more.
        messages.append({"role": "user", "content": web_message})
    public_evidence = _public_evidence(evidence)
    public_web = web.public_evidence(web_evidence)
    # When a current-information answer depends on the public web, buffer it so
    # dangling/fabricated [Wn] citations can be removed before the user sees
    # them. Ordinary chat keeps streaming token-by-token.
    web_attempted = bool(policy.web_needed and config.web_enabled())

    async def stream():
        yield _sse("evidence", {"evidence": public_evidence})
        if public_web:
            yield _sse("web_evidence", {"evidence": public_web})
        chunks: queue.Queue = queue.Queue(maxsize=8)
        stopped = threading.Event()
        register_cancel = getattr(provider, "set_stream_cancel_event", None)
        if callable(register_cancel):
            register_cancel(stopped)

        def produce():
            emitted = False

            def publish(item):
                while not stopped.is_set():
                    try:
                        chunks.put(item, timeout=0.2)
                        return True
                    except queue.Full:
                        continue
                return False

            try:
                for piece in provider.chat_stream(messages):
                    if stopped.is_set():
                        break
                    if piece:
                        emitted = True
                        if not publish(("delta", piece)):
                            return
                if not stopped.is_set():
                    publish(("complete", None) if emitted else
                            ("error", ("Provider returned an empty answer", False)))
            except Exception:
                if not stopped.is_set():
                    # Do not include provider exception details: they can contain
                    # URLs or request data and are not useful to the client.
                    failure = "Provider stream failed"
                    if emitted:
                        failure = "Provider stream ended early"
                    publish(("error", (failure, emitted)))

        worker = threading.Thread(target=produce, name="jarvis-llm-stream", daemon=True)
        worker.start()
        answer_parts: list[str] = []
        try:
            while True:
                if await request.is_disconnected():
                    stopped.set()
                    cancel_stream = getattr(provider, "cancel_chat_stream", None)
                    if callable(cancel_stream):
                        cancel_stream()
                    return
                try:
                    kind, value = await asyncio.to_thread(chunks.get, True, 0.2)
                except queue.Empty:
                    continue
                if kind == "delta":
                    answer_parts.append(value)
                    if not web_attempted:
                        yield _sse("delta", {"text": value})
                elif kind == "error":
                    message, incomplete = value
                    yield _sse("error", {"error": message, "incomplete": incomplete})
                    return
                else:
                    answer = "".join(answer_parts)
                    if web_attempted and not stopped.is_set():
                        # Fail closed: never deliver a citation that has no
                        # matching retrieved source, and guarantee an honest
                        # "cannot verify" line when nothing was retrieved.
                        answer = web.sanitize_web_citations(answer, web_evidence)
                        if not web_evidence:
                            note = web.empty_result_message(web_result.status)
                            answer = f"{answer.rstrip()} {note}".strip() if answer.strip() else note
                        answer = recall.scrub_internal_labels(answer)
                        if answer:
                            yield _sse("delta", {"text": answer})
                    else:
                        answer = recall.scrub_internal_labels(answer)
                    if continuity_update and not stopped.is_set():
                        # Update after one successful answer. The local JSON
                        # completion is optional; failures never alter the
                        # streamed answer or overwrite a valid snapshot.
                        try:
                            await asyncio.to_thread(
                                _persist_continuity_update,
                                req.message, history, continuity_thread, provider,
                            )
                        except Exception:
                            _log.warning("Conversation continuity update failed; continuing without it")
                    citations = recall.check_citations(answer, evidence)
                    web_citations = web.check_web_citations(answer, web_evidence)
                    yield _sse("done", {"answer": answer,
                                        "citations": citations,
                                        "citation_valid": citations["ok"],
                                        "web_citations": web_citations,
                                        "web_status": web_result.status,
                                        "web_query": web_result.query,
                                        "web_attempted": web_attempted})
                    return
        finally:
            stopped.set()
            cancel_stream = getattr(provider, "cancel_chat_stream", None)
            if callable(cancel_stream):
                cancel_stream()

    return StreamingResponse(
        stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


def _persist_continuity_update(message, history, thread, provider):
    """Run gated local structured continuity update outside the event loop."""
    cfg = _cfg()
    _, dbp = _paths(cfg)
    db = storage.JarvisDB(dbp)
    try:
        turns = list(history or []) + [{"role": "user", "content": message}]
        return conversation_continuity.update_from_local_model(
            db, provider, turns, thread=thread,
            start_new=(conversation_continuity.directive(message) == "start" and thread is None),
        )
    finally:
        db.close()

@app.get("/health")
def health():
    return {"ok": True, "version": "0.2.0"}

@app.post("/api/add")
def add(req: AddReq):
    if not req.text or not req.text.strip():
        raise HTTPException(400, "empty text")
    cfg = _cfg()
    raw_dir, dbp = _paths(cfg)
    db = storage.JarvisDB(dbp)
    dt = utils.utc_now()
    fpath, relpath, ch = utils.write_raw(raw_dir, req.text, dt)
    eid = db.insert_entry(text=req.text, source=req.source, origin_kind=req.origin_kind, source_type='text', raw_path=fpath, raw_relpath=relpath, content_hash=ch)
    from app.core import extract
    ex = extract.HeuristicExtractor().extract(req.text, utils.utc_iso(dt))
    db.add_structured(eid, ex, inferred_by='heuristic')
    db.close()
    return {"id": eid, "raw_relpath": relpath}

@app.get("/api/search")
def search(q: str = Query(..., min_length=1), limit: int = Query(10, ge=1, le=100), offset: int = Query(0, ge=0)):
    cfg = _cfg()
    raw_dir, dbp = _paths(cfg)
    db = storage.JarvisDB(dbp)
    res = db.search_fts(q, limit=limit+offset)
    res = search_mod.rank_results(res, q=q, self_boost=cfg.get('search',{}).get('self_boost',0.25), recency_boost=cfg.get('search',{}).get('recency_boost',0.15))
    res = res[offset:offset+limit]
    db.close()
    return {"results": res}

@app.get("/api/show/{eid}")
def show(eid: int):
    cfg = _cfg()
    raw_dir, dbp = _paths(cfg)
    db = storage.JarvisDB(dbp)
    g = db.get(eid)
    db.close()
    if not g or not g['entry']:
        raise HTTPException(404, "not found")
    return g

@app.get("/api/recent")
def recent(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)):
    cfg = _cfg()
    raw_dir, dbp = _paths(cfg)
    db = storage.JarvisDB(dbp)
    rows = db.recent(limit=limit+offset)
    rows = rows[offset:offset+limit]
    db.close()
    return {"results": rows}

@app.get("/api/stats")
def stats():
    cfg = _cfg()
    raw_dir, dbp = _paths(cfg)
    db = storage.JarvisDB(dbp)
    st = db.stats()
    db.close()
    return st

@app.get("/")
def root(token: Optional[str] = None):
    """The app shell, or the one-time token bootstrap.

    ``/?token=...`` is the single sanctioned place a token may arrive as a query
    parameter. It is validated here with the same constant-time comparison the
    API uses, and on success the response is a minimal page that hands the token
    to ``localStorage`` and immediately replaces the URL. The token is never
    placed in the HTML, never returned as JSON, and never echoed back.

    ``/api/*`` continues to reject query-string tokens outright (see
    ``auth.require_token``): URLs end up in access logs and Referer headers.
    Group 1's access-log redaction keeps the token out of the log even for this
    one allowed URL.
    """
    if pwa.is_bootstrap("/", token):
        if not auth.tokens_match(token, config.token()):
            # Deliberately opaque, and no-store like the success path so a
            # failed attempt is not cached either.
            raise HTTPException(401, "unauthorized", headers=dict(pwa.BOOTSTRAP_HEADERS))
        return Response(
            content=pwa.BOOTSTRAP_HTML,
            media_type="text/html; charset=utf-8",
            headers=dict(pwa.BOOTSTRAP_HEADERS),
        )
    return FileResponse(BASE / 'static' / 'index.html')

@app.get("/{asset}")
def pwa_asset(asset: str):
    """Serve a PWA asset from the allowlist.

    An explicit allowlist rather than a StaticFiles mount: a mount would become
    a catch-all for any unrouted path and would serve whatever files exist on
    disk. Here an unknown name is simply a 404, so no crafted path can reach
    ``config.yaml`` or escape ``static/``. Kept unauthenticated because the
    shell, manifest and icons must load before the user has a token, and
    because they contain no personal data.
    """
    media_type = pwa.ASSETS.get(asset)
    if media_type is None:
        raise HTTPException(404, "not found")
    path = pwa.STATIC_DIR / asset
    if not path.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(path, media_type=media_type, headers=pwa.asset_headers(asset))
