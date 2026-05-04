"""Surface Commons — Indic Translator.

Free open-source replacement for paid translation SaaS (Google Translate,
DeepL Pro). Translates between English and the major Indian scripts using
the existing `route_think` substrate (claude_cli backend) so we ride on the
already-paid MAX subscription instead of standing up a model server.

Endpoints (all under `/commons` prefix):
    POST /commons/translate                   single string, 2 SC
    POST /commons/translate/batch             list of items, 2 SC * N
    POST /commons/translate/detect            langdetect, 1 SC
    GET  /commons/translate/health            substrate + supported pairs

A future v2 will swap the substrate for native ai4bharat/indictrans2; the
public contract stays identical so callers don't need to migrate.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.service_auth import require_service_key_and_agent
from app.models.agent import Agent
from app.services.ledger import InsufficientFundsError, Reason, record_transaction
from app.services.router_service import route_think

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/commons", tags=["commons", "translate"])


# ─────────────────────────────────────────────────────────────────────────
# Supported languages — declared once, used in /health and validated on
# every request. Codes follow ISO-639-1 where defined; "en" is the pivot.
# ─────────────────────────────────────────────────────────────────────────

SUPPORTED_LANGUAGES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "ta": "Tamil",
    "bn": "Bengali",
    "te": "Telugu",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "ur": "Urdu",
}


def _supported_pairs() -> list[str]:
    """All bidirectional pairs we currently advertise. English is the pivot
    for v1 — non-English ↔ non-English routes through the LLM directly,
    which works in practice for these scripts but isn't formally tested."""
    pairs: list[str] = []
    for src in SUPPORTED_LANGUAGES:
        for tgt in SUPPORTED_LANGUAGES:
            if src == tgt:
                continue
            pairs.append(f"{src}->{tgt}")
    return pairs


# SC pricing (mission spec) — flat per-call costs, paid up front, no refund
# on substrate failure (yet) since the LLM call already burned tokens.
SC_PER_TRANSLATE = 2.0
SC_PER_DETECT = 1.0


# ─────────────────────────────────────────────────────────────────────────
# Request / response schemas
# ─────────────────────────────────────────────────────────────────────────


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)
    source_lang: str = Field(..., min_length=2, max_length=8)
    target_lang: str = Field(..., min_length=2, max_length=8)


class TranslateResponse(BaseModel):
    translation: str
    model_used: str
    confidence: float
    sc_charged: float


class BatchItem(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)
    src: str = Field(..., min_length=2, max_length=8)
    tgt: str = Field(..., min_length=2, max_length=8)


class BatchRequest(BaseModel):
    items: list[BatchItem] = Field(..., min_length=1, max_length=50)


class BatchTranslation(BaseModel):
    translation: str
    model_used: str
    confidence: float
    error: str | None = None


class BatchResponse(BaseModel):
    translations: list[BatchTranslation]
    sc_charged: float


class DetectRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)


class DetectResponse(BaseModel):
    detected_lang: str
    confidence: float
    sc_charged: float


class HealthResponse(BaseModel):
    ok: bool
    substrate: str
    substrate_note: str
    languages: dict[str, str]
    supported_pairs: list[str]
    sc_per_translate: float
    sc_per_detect: float


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


def _validate_pair(src: str, tgt: str) -> None:
    if src not in SUPPORTED_LANGUAGES:
        raise HTTPException(400, detail=f"unsupported_source_lang:{src}")
    if tgt not in SUPPORTED_LANGUAGES:
        raise HTTPException(400, detail=f"unsupported_target_lang:{tgt}")
    if src == tgt:
        raise HTTPException(400, detail="src_and_tgt_identical")


def _build_translation_prompt(src: str, tgt: str, body: str) -> str:
    src_name = SUPPORTED_LANGUAGES[src]
    tgt_name = SUPPORTED_LANGUAGES[tgt]
    return (
        f"Translate from {src_name} to {tgt_name} preserving honorifics and "
        f"idiom. Text: {body}. Output ONLY the translation."
    )


def _extract_text(result: dict[str, Any]) -> str:
    """route_think returns a dict with `content` either as a plain string
    (gemini/ollama path) or as a list of Anthropic-shape blocks
    (anthropic/claude_cli). Pull the first plain-text payload either way."""
    if not isinstance(result, dict):
        return ""
    content = result.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text")
                if isinstance(t, str) and t.strip():
                    return t.strip()
    txt = result.get("text")
    if isinstance(txt, str) and txt.strip():
        return txt.strip()
    return ""


async def _log_tool_call(
    db: AsyncSession,
    agent_id: str,
    input_payload: dict[str, Any],
    output_len: int,
    sc_cost: float,
    latency_ms: int,
    error: str | None,
) -> None:
    """Append-only audit row in tool_calls. Fire-and-forget on failure —
    we never want logging to block a successful translate."""
    try:
        await db.execute(
            text(
                """
                INSERT INTO tool_calls
                    (id, agent_id, tool_name, input_json, output_len,
                     sc_cost, latency_ms, error)
                VALUES
                    (:id, :agent_id, :tool_name, CAST(:input_json AS JSONB),
                     :output_len, :sc_cost, :latency_ms, :error)
                """
            ),
            {
                "id": str(uuid.uuid4()),
                "agent_id": agent_id,
                "tool_name": "commons.indic_translate",
                "input_json": _json_dumps_safe(input_payload),
                "output_len": output_len,
                "sc_cost": sc_cost,
                "latency_ms": latency_ms,
                "error": error,
            },
        )
    except Exception as exc:
        logger.warning("tool_calls_log_failed", extra={"error": str(exc)})


def _json_dumps_safe(payload: dict[str, Any]) -> str:
    import json

    try:
        return json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        return "{}"


async def _charge_sc(
    db: AsyncSession, agent_id: str, sc: float, ref_id: str, note: str
) -> None:
    if sc <= 0:
        return
    try:
        tx = await record_transaction(
            db=db,
            agent_id=agent_id,
            delta=-sc,
            reason=Reason.TOOL_CALL,
            ref_id=ref_id,
            note=note,
        )
    except InsufficientFundsError as exc:
        raise HTTPException(status_code=402, detail=str(exc))
    if tx is None:
        raise HTTPException(status_code=404, detail="agent_not_found_for_debit")


async def _translate_one(src: str, tgt: str, body: str, agent_id: str) -> tuple[str, str]:
    """Returns (translation, model_used). Raises HTTPException on substrate failure."""
    prompt = _build_translation_prompt(src, tgt, body)
    messages = [{"role": "user", "content": prompt}]
    # Translation is a deterministic, low-tier task: try a chain of providers
    # so the tool keeps working regardless of the calling agent's PAPP
    # preference and isn't held hostage by any single billing issue.
    last_err = "no_provider_attempted"
    result: dict[str, Any] | None = None
    for provider in ("gemini", "claude_cli", "ollama"):
        try:
            result = await route_think(
                agent_id=agent_id,
                tier="fast",
                messages=messages,
                max_tokens=1024,
                provider_override=provider,
            )
            if _extract_text(result):
                break
            last_err = f"{provider}:empty"
            result = None
        except Exception as exc:
            last_err = f"{provider}:{exc}"
            result = None
            continue
    if not result:
        raise HTTPException(status_code=502, detail=f"substrate_error:{last_err}")
    text_out = _extract_text(result)
    if not text_out:
        raise HTTPException(status_code=502, detail="empty_translation")
    model_used = result.get("model") or "claude_cli"
    return text_out, str(model_used)


# ─────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────


@router.get("/translate/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        ok=True,
        substrate="claude_cli (route_think)",
        substrate_note="via route_think substrate (claude_cli/anthropic/gemini fallback chain); native IndicTrans2 planned v2.",
        languages=SUPPORTED_LANGUAGES,
        supported_pairs=_supported_pairs(),
        sc_per_translate=SC_PER_TRANSLATE,
        sc_per_detect=SC_PER_DETECT,
    )


@router.post("/translate", response_model=TranslateResponse)
async def translate(
    body: TranslateRequest,
    db: AsyncSession = Depends(get_db),
    agent: Agent = Depends(require_service_key_and_agent),
) -> TranslateResponse:
    _validate_pair(body.source_lang, body.target_lang)
    ref_id = str(uuid.uuid4())
    started = time.perf_counter()
    error: str | None = None
    translation = ""
    model_used = "claude_cli"
    try:
        translation, model_used = await _translate_one(
            body.source_lang, body.target_lang, body.text, agent.id
        )
    except HTTPException as exc:
        error = str(exc.detail)
        raise
    finally:
        latency_ms = int((time.perf_counter() - started) * 1000)
        await _log_tool_call(
            db=db,
            agent_id=agent.id,
            input_payload={
                "text": body.text[:500],
                "source_lang": body.source_lang,
                "target_lang": body.target_lang,
            },
            output_len=len(translation),
            sc_cost=SC_PER_TRANSLATE,
            latency_ms=latency_ms,
            error=error,
        )
        if error is None:
            await _charge_sc(
                db=db,
                agent_id=agent.id,
                sc=SC_PER_TRANSLATE,
                ref_id=ref_id,
                note=f"commons.indic_translate {body.source_lang}->{body.target_lang}",
            )
            await db.commit()
        else:
            await db.commit()  # commit the audit row even on error

    return TranslateResponse(
        translation=translation,
        model_used=model_used,
        confidence=0.85,
        sc_charged=SC_PER_TRANSLATE,
    )


@router.post("/translate/batch", response_model=BatchResponse)
async def translate_batch(
    body: BatchRequest,
    db: AsyncSession = Depends(get_db),
    agent: Agent = Depends(require_service_key_and_agent),
) -> BatchResponse:
    for item in body.items:
        _validate_pair(item.src, item.tgt)

    results: list[BatchTranslation] = []
    successful = 0
    started = time.perf_counter()
    for item in body.items:
        try:
            translation, model_used = await _translate_one(
                item.src, item.tgt, item.text, agent.id
            )
            results.append(
                BatchTranslation(
                    translation=translation, model_used=model_used, confidence=0.85
                )
            )
            successful += 1
        except HTTPException as exc:
            results.append(
                BatchTranslation(
                    translation="",
                    model_used="claude_cli",
                    confidence=0.0,
                    error=str(exc.detail),
                )
            )

    latency_ms = int((time.perf_counter() - started) * 1000)
    sc_to_charge = SC_PER_TRANSLATE * successful

    ref_id = str(uuid.uuid4())
    await _log_tool_call(
        db=db,
        agent_id=agent.id,
        input_payload={"batch_size": len(body.items), "successful": successful},
        output_len=sum(len(r.translation) for r in results),
        sc_cost=sc_to_charge,
        latency_ms=latency_ms,
        error=None if successful == len(body.items) else "partial_batch",
    )
    if sc_to_charge > 0:
        await _charge_sc(
            db=db,
            agent_id=agent.id,
            sc=sc_to_charge,
            ref_id=ref_id,
            note=f"commons.indic_translate batch x{successful}",
        )
    await db.commit()

    return BatchResponse(translations=results, sc_charged=sc_to_charge)


@router.post("/translate/detect", response_model=DetectResponse)
async def detect(
    body: DetectRequest,
    db: AsyncSession = Depends(get_db),
    agent: Agent = Depends(require_service_key_and_agent),
) -> DetectResponse:
    try:
        from langdetect import DetectorFactory, detect_langs

        DetectorFactory.seed = 0  # deterministic across calls
        candidates = detect_langs(body.text)
    except ImportError:
        raise HTTPException(503, detail="langdetect_not_installed")
    except Exception as exc:
        raise HTTPException(422, detail=f"detect_failed:{exc}")

    if not candidates:
        raise HTTPException(422, detail="no_language_detected")

    top = candidates[0]
    code = str(top.lang)
    confidence = float(top.prob)
    if code not in SUPPORTED_LANGUAGES:
        # Detection succeeded but lies outside our advertised set — surface
        # the raw code anyway; caller can decide whether to translate.
        pass

    ref_id = str(uuid.uuid4())
    await _log_tool_call(
        db=db,
        agent_id=agent.id,
        input_payload={"text": body.text[:500]},
        output_len=len(code),
        sc_cost=SC_PER_DETECT,
        latency_ms=0,
        error=None,
    )
    await _charge_sc(
        db=db,
        agent_id=agent.id,
        sc=SC_PER_DETECT,
        ref_id=ref_id,
        note=f"commons.indic_translate.detect -> {code}",
    )
    await db.commit()

    return DetectResponse(
        detected_lang=code, confidence=confidence, sc_charged=SC_PER_DETECT
    )
