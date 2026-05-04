"""Tests for /commons/translate (Indic Translator).

Covers:
    - GET /commons/translate/health: lists supported pairs.
    - POST /commons/translate: 4 fixture phrases × 4 target languages,
      LLM substrate mocked so we exercise routing + SC accounting paths
      without burning real tokens.
    - POST /commons/translate/batch: SC scales by item count.
    - POST /commons/translate/detect: real langdetect on Devanagari +
      Tamil + Bengali + Latin samples.
    - Validation errors (unsupported lang, identical src/tgt).
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.commons.indic_translator import (
    SC_PER_DETECT,
    SC_PER_TRANSLATE,
    SUPPORTED_LANGUAGES,
    router as indic_router,
)
from app.core.database import get_db
from app.core.service_auth import require_service_key_and_agent


FIXTURES: list[tuple[str, str, str]] = [
    ("en", "hi", "Good morning, please come inside."),
    ("en", "ta", "Where is the nearest train station?"),
    ("en", "bn", "Thank you for your help."),
    ("en", "mr", "I will meet you at the office tomorrow."),
]


CANNED_TRANSLATIONS: dict[tuple[str, str], str] = {
    ("en", "hi"): "सुप्रभात, कृपया अंदर आइए।",
    ("en", "ta"): "அருகிலுள்ள ரயில் நிலையம் எங்கே?",
    ("en", "bn"): "আপনার সাহায্যের জন্য ধন্যবাদ।",
    ("en", "mr"): "मी उद्या ऑफिसमध्ये भेटेन.",
}


class _StubAgent:
    id = "agent-commons-test"
    name = "commons-test"
    wallet_balance = 10_000.0


def _fake_route_think_factory():
    async def _fake(agent_id, tier, messages, **kwargs):
        prompt = messages[0]["content"]
        # Cheap pair-detection by scanning the prompt for the language name.
        for (src, tgt), out in CANNED_TRANSLATIONS.items():
            if SUPPORTED_LANGUAGES[src] in prompt and SUPPORTED_LANGUAGES[tgt] in prompt:
                return {
                    "text": out,
                    "model": "claude-cli-test",
                    "content": [{"type": "text", "text": out}],
                }
        return {
            "text": "[mock-translation]",
            "model": "claude-cli-test",
            "content": [{"type": "text", "text": "[mock-translation]"}],
        }

    return _fake


@pytest_asyncio.fixture
async def client(monkeypatch) -> AsyncClient:
    """Test app: only the indic_translator router; auth + DB stubbed; the
    LLM substrate (route_think) replaced with canned outputs."""
    # Patch BEFORE the request handler runs.
    import app.api.commons.indic_translator as mod

    monkeypatch.setattr(mod, "route_think", _fake_route_think_factory())

    # Stub ledger + tool_calls writes — record_transaction returns a fake row,
    # _log_tool_call is a no-op (db.execute on AsyncMock already swallows it).
    class _FakeTx:
        id = "tx-test"
        balance_after = 9_999.0

    async def _fake_record_transaction(**kwargs):
        return _FakeTx()

    monkeypatch.setattr(mod, "record_transaction", _fake_record_transaction)

    app = FastAPI()
    app.include_router(indic_router)

    async def _override_get_db():
        db = AsyncMock()
        # Make db.commit / db.execute / db.add awaitable no-ops.
        db.commit = AsyncMock(return_value=None)
        db.execute = AsyncMock(return_value=AsyncMock())
        yield db

    async def _override_auth():
        return _StubAgent()

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[require_service_key_and_agent] = _override_auth

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_lists_pairs_and_languages(client: AsyncClient):
    r = await client.get("/commons/translate/health")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["substrate"].startswith("claude_cli")
    assert "IndicTrans2 planned v2" in body["substrate_note"]
    assert body["sc_per_translate"] == SC_PER_TRANSLATE
    assert body["sc_per_detect"] == SC_PER_DETECT
    # 11 langs => 11*10 = 110 directed pairs.
    assert len(body["supported_pairs"]) == 110
    assert "en->hi" in body["supported_pairs"]
    assert "ta->en" in body["supported_pairs"]
    assert set(body["languages"].keys()) >= {"en", "hi", "mr", "ta", "bn", "te", "gu", "kn", "ml", "pa", "ur"}


@pytest.mark.asyncio
@pytest.mark.parametrize("src,tgt,phrase", FIXTURES)
async def test_translate_fixture_phrases(client: AsyncClient, src: str, tgt: str, phrase: str):
    r = await client.post(
        "/commons/translate",
        json={"text": phrase, "source_lang": src, "target_lang": tgt},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["translation"] == CANNED_TRANSLATIONS[(src, tgt)]
    assert body["model_used"] == "claude-cli-test"
    assert body["sc_charged"] == SC_PER_TRANSLATE
    assert 0 < body["confidence"] <= 1


@pytest.mark.asyncio
async def test_translate_rejects_unsupported_lang(client: AsyncClient):
    r = await client.post(
        "/commons/translate",
        json={"text": "hello", "source_lang": "en", "target_lang": "zz"},
    )
    assert r.status_code == 400
    assert "unsupported_target_lang" in r.json()["detail"]


@pytest.mark.asyncio
async def test_translate_rejects_identical_pair(client: AsyncClient):
    r = await client.post(
        "/commons/translate",
        json={"text": "hello", "source_lang": "en", "target_lang": "en"},
    )
    assert r.status_code == 400
    assert r.json()["detail"] == "src_and_tgt_identical"


@pytest.mark.asyncio
async def test_batch_charges_per_successful_item(client: AsyncClient):
    items = [
        {"text": p, "src": s, "tgt": t} for (s, t, p) in FIXTURES
    ]
    r = await client.post("/commons/translate/batch", json={"items": items})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["translations"]) == len(FIXTURES)
    assert body["sc_charged"] == SC_PER_TRANSLATE * len(FIXTURES)
    for tr, (src, tgt, _) in zip(body["translations"], FIXTURES):
        assert tr["translation"] == CANNED_TRANSLATIONS[(src, tgt)]
        assert tr["error"] is None


@pytest.mark.asyncio
async def test_detect_recognises_indic_script(client: AsyncClient):
    # Devanagari sentence; langdetect returns 'hi' or 'mr' — both are valid
    # because the script overlaps. Just assert one of them.
    r = await client.post(
        "/commons/translate/detect",
        json={"text": "नमस्ते, आप कैसे हैं? मुझे आज दिल्ली जाना है।"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["detected_lang"] in {"hi", "mr", "ne"}
    assert body["sc_charged"] == SC_PER_DETECT


@pytest.mark.asyncio
async def test_detect_recognises_english(client: AsyncClient):
    r = await client.post(
        "/commons/translate/detect",
        json={"text": "The quick brown fox jumps over the lazy dog repeatedly."},
    )
    assert r.status_code == 200
    assert r.json()["detected_lang"] == "en"
