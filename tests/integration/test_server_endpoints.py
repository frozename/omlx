# SPDX-License-Identifier: Apache-2.0
"""
Integration tests for oMLX server endpoints.

Tests the FastAPI endpoints using TestClient with mocked EnginePool and Engine
to verify request/response formats without loading actual models.
"""

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

from fastapi.testclient import TestClient

from omlx.api.responses_utils import ResponseStore
from omlx.engine.base import BaseEngine
from omlx.engine.embedding import EmbeddingEngine
from omlx.engine.reranker import RerankerEngine
from omlx.mcp.types import MCPToolResult
from omlx.settings import GlobalSettings


@dataclass
class MockEmbeddingOutput:
    """Mock embedding output for testing."""

    embeddings: List[List[float]] = field(
        default_factory=lambda: [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    )
    total_tokens: int = 10
    dimensions: int = 3


@dataclass
class MockRerankOutput:
    """Mock rerank output for testing."""

    scores: List[float] = field(default_factory=lambda: [0.9, 0.5, 0.3])
    indices: List[int] = field(default_factory=lambda: [0, 1, 2])
    total_tokens: int = 50


@dataclass
class MockGenerationOutput:
    """Mock generation output for testing."""

    text: str = "Hello, I am a helpful assistant."
    tokens: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    prompt_tokens: int = 10
    completion_tokens: int = 5
    finish_reason: str = "stop"
    new_text: str = ""
    finished: bool = True
    tool_calls: Optional[List[Dict[str, Any]]] = None
    cached_tokens: int = 0


class MockEmbeddingEngineImpl(EmbeddingEngine):
    """Mock embedding engine for testing that inherits from EmbeddingEngine."""

    def __init__(self, model_name: str = "test-embedding-model"):
        # Don't call super().__init__ to avoid loading real model
        self._model_name = model_name
        self._model = None  # Set as None but present

    @property
    def model_name(self) -> str:
        return self._model_name

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def embed(self, texts, **kwargs) -> MockEmbeddingOutput:
        return MockEmbeddingOutput(
            embeddings=[[0.1, 0.2, 0.3] for _ in texts],
            total_tokens=len(texts) * 5,
            dimensions=3,
        )

    def get_stats(self) -> Dict[str, Any]:
        return {"model_name": self._model_name, "loaded": True}


class MockRerankerEngineImpl(RerankerEngine):
    """Mock reranker engine for testing that inherits from RerankerEngine."""

    def __init__(self, model_name: str = "test-reranker-model"):
        # Don't call super().__init__ to avoid loading real model
        self._model_name = model_name
        self._model = None  # Set as None but present

    @property
    def model_name(self) -> str:
        return self._model_name

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def rerank(
        self, query: str, documents: List[str], top_n: Optional[int] = None, **kwargs
    ) -> MockRerankOutput:
        n_docs = len(documents)
        scores = [0.9 - i * 0.2 for i in range(n_docs)]
        indices = list(range(n_docs))
        if top_n:
            indices = indices[:top_n]
        return MockRerankOutput(
            scores=scores,
            indices=indices,
            total_tokens=n_docs * 20,
        )

    def get_stats(self) -> Dict[str, Any]:
        return {"model_name": self._model_name, "loaded": True}


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self.eos_token_id = 2

    def encode(self, text: str) -> List[int]:
        # Simple simulation: split by words
        return [100 + i for i, _ in enumerate(text.split())]

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        return f"<decoded:{len(tokens)} tokens>"

    def apply_chat_template(
        self, messages: List[Dict], tokenize: bool = False, **kwargs
    ) -> str:
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)


class MockBaseEngine(BaseEngine):
    """Mock LLM engine for testing."""

    def __init__(self, model_name: str = "test-llm-model"):
        self._model_name = model_name
        self._tokenizer = MockTokenizer()
        self._model_type = "llama"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model_type(self) -> Optional[str]:
        return self._model_type

    @property
    def prefix_cache_enabled(self) -> bool:
        return False

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def generate(self, prompt: str, **kwargs) -> MockGenerationOutput:
        return MockGenerationOutput(text="Generated response.")

    async def stream_generate(self, prompt: str, **kwargs):
        yield MockGenerationOutput(
            text="Hello",
            new_text="Hello",
            finished=False,
        )
        yield MockGenerationOutput(
            text="Hello world",
            new_text=" world",
            finished=True,
            finish_reason="stop",
        )

    def count_chat_tokens(self, messages: List[Dict], tools=None, chat_template_kwargs=None, **kwargs) -> int:
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False)
        return len(self._tokenizer.encode(prompt))

    async def chat(self, messages: List[Dict], **kwargs) -> MockGenerationOutput:
        return MockGenerationOutput(text="Chat response.")

    async def stream_chat(self, messages: List[Dict], **kwargs):
        yield MockGenerationOutput(
            text="Hello",
            new_text="Hello",
            finished=False,
        )
        yield MockGenerationOutput(
            text="Hello from chat",
            new_text=" from chat",
            finished=True,
            finish_reason="stop",
        )

    def get_stats(self) -> Dict[str, Any]:
        return {}

    def get_cache_stats(self):
        return None


class RecordingResponsesEngine(MockBaseEngine):
    """Mock engine that records request messages across /v1/responses calls."""

    def __init__(self, outputs: Optional[List[MockGenerationOutput]] = None):
        super().__init__()
        self._outputs = list(outputs or [])
        self.recorded_messages: List[List[Dict[str, Any]]] = []
        self._model_type = "gpt_oss"

    async def chat(self, messages: List[Dict], **kwargs) -> MockGenerationOutput:
        self.recorded_messages.append(messages)
        if self._outputs:
            return self._outputs.pop(0)
        return MockGenerationOutput(text="Chat response.")


class MockEnginePool:
    """Mock engine pool for testing."""

    def __init__(
        self,
        llm_engine: Optional[MockBaseEngine] = None,
        embedding_engine: Optional[MockEmbeddingEngineImpl] = None,
        reranker_engine: Optional[MockRerankerEngineImpl] = None,
    ):
        self._llm_engine = llm_engine or MockBaseEngine()
        self._embedding_engine = embedding_engine
        self._reranker_engine = reranker_engine
        self._models = [
            {"id": "test-model", "loaded": True, "pinned": False, "size": 1000000}
        ]

    @property
    def model_count(self) -> int:
        return len(self._models)

    @property
    def loaded_model_count(self) -> int:
        return sum(1 for m in self._models if m["loaded"])

    @property
    def max_model_memory(self) -> int:
        return 32 * 1024 * 1024 * 1024  # 32GB

    @property
    def current_model_memory(self) -> int:
        return 1000000

    def get_entry(self, model_id: str):
        return None

    def resolve_model_id(self, model_id_or_alias, settings_manager=None):
        return model_id_or_alias

    def get_model_ids(self) -> List[str]:
        return [m["id"] for m in self._models]

    def get_status(self) -> Dict[str, Any]:
        return {
            "models": self._models,
            "loaded_count": self.loaded_model_count,
            "max_model_memory": self.max_model_memory,
        }

    async def get_engine(self, model_id: str):
        # Return appropriate engine based on model name pattern
        if "embed" in model_id.lower():
            if self._embedding_engine:
                return self._embedding_engine
            raise ValueError(f"No embedding engine for {model_id}")
        elif "rerank" in model_id.lower():
            if self._reranker_engine:
                return self._reranker_engine
            raise ValueError(f"No reranker engine for {model_id}")
        return self._llm_engine


@pytest.fixture
def mock_llm_engine():
    """Create a mock LLM engine."""
    return MockBaseEngine()


@pytest.fixture
def mock_embedding_engine():
    """Create a mock embedding engine."""
    return MockEmbeddingEngineImpl()


@pytest.fixture
def mock_reranker_engine():
    """Create a mock reranker engine."""
    return MockRerankerEngineImpl()


@pytest.fixture
def mock_engine_pool(mock_llm_engine, mock_embedding_engine, mock_reranker_engine):
    """Create a mock engine pool."""
    return MockEnginePool(
        llm_engine=mock_llm_engine,
        embedding_engine=mock_embedding_engine,
        reranker_engine=mock_reranker_engine,
    )


@pytest.fixture
def client(mock_engine_pool):
    """Create a test client with mocked server state."""
    from omlx.server import app, _server_state

    # Store original state
    original_pool = _server_state.engine_pool
    original_default = _server_state.default_model

    # Set mock state
    _server_state.engine_pool = mock_engine_pool
    _server_state.default_model = "test-model"

    yield TestClient(app)

    # Restore original state
    _server_state.engine_pool = original_pool
    _server_state.default_model = original_default


class TestSlotSaveEndpoint:
    def _configure_slot_runtime(self, server_state, slot_dir, pool, tmp_path) -> None:
        model_dir = tmp_path / "model-artifacts"
        model_dir.mkdir(exist_ok=True)
        (model_dir / "config.json").write_text('{"model":"test-model"}', encoding="utf-8")

        class _Entry:
            def __init__(self, path):
                self.model_path = str(path)
                self.engine = None

        entry = _Entry(model_dir)
        pool.get_entry = lambda model_id: entry

        settings = GlobalSettings()
        settings.slot_save_path = str(slot_dir)
        settings.scheduler.max_concurrent_requests = 1
        server_state.global_settings = settings
        server_state.engine_pool = pool
        server_state.default_model = "test-model"
        server_state.api_key = None

    def _patch_minimal_slot_payload(self, monkeypatch, payload_size: int = 8):
        import omlx.server as server_module

        def fake_serialize_slot_payload(*args, **kwargs):
            return b"x" * payload_size, {
                "n_tokens": 123,
                "tensors": [{"name": "layer_0", "dtype": "f16", "shape": [1, 2]}],
                "cache_class": "paged_ssd",
            }

        monkeypatch.setattr(
            server_module,
            "_serialize_slot_payload",
            fake_serialize_slot_payload,
        )

    def test_save_rejects_absolute_path(self, tmp_path, mock_engine_pool, monkeypatch):
        from omlx.server import app, _server_state

        slot_dir = tmp_path / "slots"
        slot_dir.mkdir()

        original_pool = _server_state.engine_pool
        original_default = _server_state.default_model
        original_settings = _server_state.global_settings
        original_api_key = _server_state.api_key
        original_slot_store = getattr(_server_state, "slot_store", None)
        try:
            self._configure_slot_runtime(_server_state, slot_dir, mock_engine_pool, tmp_path)
            self._patch_minimal_slot_payload(monkeypatch)
            client = TestClient(app)

            response = client.post(
                "/slots/0?action=save",
                json={"filename": "/etc/passwd", "model": "test-model"},
            )
            assert response.status_code == 400
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store

    def test_save_rejects_dotdot_traversal(self, tmp_path, mock_engine_pool, monkeypatch):
        from omlx.server import app, _server_state

        slot_dir = tmp_path / "slots"
        slot_dir.mkdir()

        original_pool = _server_state.engine_pool
        original_default = _server_state.default_model
        original_settings = _server_state.global_settings
        original_api_key = _server_state.api_key
        original_slot_store = getattr(_server_state, "slot_store", None)
        try:
            self._configure_slot_runtime(_server_state, slot_dir, mock_engine_pool, tmp_path)
            self._patch_minimal_slot_payload(monkeypatch)
            client = TestClient(app)

            response = client.post(
                "/slots/0?action=save",
                json={"filename": "../escape.kvslot", "model": "test-model"},
            )
            assert response.status_code == 400
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store

    def test_save_rejects_missing_model(self, tmp_path, mock_engine_pool, monkeypatch):
        from omlx.server import app, _server_state

        slot_dir = tmp_path / "slots"
        slot_dir.mkdir()

        original_pool = _server_state.engine_pool
        original_default = _server_state.default_model
        original_settings = _server_state.global_settings
        original_api_key = _server_state.api_key
        original_slot_store = getattr(_server_state, "slot_store", None)
        try:
            self._configure_slot_runtime(_server_state, slot_dir, mock_engine_pool, tmp_path)
            self._patch_minimal_slot_payload(monkeypatch)
            client = TestClient(app)

            response = client.post(
                "/slots/0?action=save",
                json={"filename": "slot.kvslot"},
            )
            assert response.status_code == 400
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store

    def test_save_returns_structured_500_on_serialize_failure(
        self, tmp_path, mock_engine_pool, monkeypatch
    ):
        import omlx.server as server_module
        from omlx.server import app, _server_state

        slot_dir = tmp_path / "slots"
        slot_dir.mkdir()

        original_pool = _server_state.engine_pool
        original_default = _server_state.default_model
        original_settings = _server_state.global_settings
        original_api_key = _server_state.api_key
        original_slot_store = getattr(_server_state, "slot_store", None)
        try:
            self._configure_slot_runtime(_server_state, slot_dir, mock_engine_pool, tmp_path)

            def fail_serialize(*args, **kwargs):
                raise RuntimeError("serialize boom")

            monkeypatch.setattr(server_module, "_serialize_slot_payload", fail_serialize)
            client = TestClient(app)

            response = client.post(
                "/slots/0?action=save",
                json={"filename": "slot.kvslot", "model": "test-model"},
            )
            assert response.status_code == 500
            body = response.json()
            assert body["error"]["code"] == "slot_serialize_failed"
            assert body["error"]["details"]["slot_id"] == 0
            assert body["error"]["details"]["filename"] == "slot.kvslot"
            assert body["error"]["details"]["model"] == "test-model"
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store

    def test_save_busy_returns_409_when_generating(
        self, tmp_path, mock_engine_pool, monkeypatch
    ):
        from omlx.server import app, _server_state
        from omlx.slot_store import SlotState

        slot_dir = tmp_path / "slots"
        slot_dir.mkdir()

        original_pool = _server_state.engine_pool
        original_default = _server_state.default_model
        original_settings = _server_state.global_settings
        original_api_key = _server_state.api_key
        original_slot_store = getattr(_server_state, "slot_store", None)
        try:
            self._configure_slot_runtime(_server_state, slot_dir, mock_engine_pool, tmp_path)
            self._patch_minimal_slot_payload(monkeypatch)
            from omlx.server import _get_slot_store
            _get_slot_store()
            client = TestClient(app)

            # Force runtime slot state into GENERATING before save request.
            _server_state.slot_store._states[0] = SlotState.GENERATING  # noqa: SLF001

            response = client.post(
                "/slots/0?action=save",
                json={"filename": "slot.kvslot", "model": "test-model"},
            )
            assert response.status_code == 409
            detail = response.json()["detail"]
            assert detail["error"]["code"] == "slot_busy"
            assert detail["error"]["state"] == "GENERATING"
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store

    def test_save_busy_returns_409_when_already_saving(
        self, tmp_path, mock_engine_pool, monkeypatch
    ):
        import omlx.server as server_module
        from omlx.server import app, _server_state

        slot_dir = tmp_path / "slots"
        slot_dir.mkdir()

        original_pool = _server_state.engine_pool
        original_default = _server_state.default_model
        original_settings = _server_state.global_settings
        original_api_key = _server_state.api_key
        original_slot_store = getattr(_server_state, "slot_store", None)
        try:
            self._configure_slot_runtime(_server_state, slot_dir, mock_engine_pool, tmp_path)

            def slow_serialize_slot_payload(*args, **kwargs):
                time.sleep(0.2)
                return b"payload", {
                    "n_tokens": 42,
                    "tensors": [{"name": "layer_0", "dtype": "f16", "shape": [1]}],
                    "cache_class": "paged_ssd",
                }

            monkeypatch.setattr(
                server_module,
                "_serialize_slot_payload",
                slow_serialize_slot_payload,
            )

            client = TestClient(app)
            first_done = threading.Event()
            first_resp = {}

            def first_save():
                first_resp["response"] = client.post(
                    "/slots/0?action=save",
                    json={"filename": "slot-a.kvslot", "model": "test-model"},
                )
                first_done.set()

            t = threading.Thread(target=first_save, daemon=True)
            t.start()
            time.sleep(0.05)

            second = client.post(
                "/slots/0?action=save",
                json={"filename": "slot-b.kvslot", "model": "test-model"},
            )
            assert second.status_code == 409
            assert second.json()["detail"]["error"]["code"] == "slot_busy"

            assert first_done.wait(timeout=1.0)
            assert first_resp["response"].status_code == 200
            t.join(timeout=1.0)
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store

    @pytest.mark.asyncio
    async def test_save_event_loop_not_blocked(self, tmp_path, mock_engine_pool, monkeypatch):
        httpx = pytest.importorskip("httpx")
        import omlx.server as server_module
        from omlx.server import app, _server_state

        slot_dir = tmp_path / "slots"
        slot_dir.mkdir()

        original_pool = _server_state.engine_pool
        original_default = _server_state.default_model
        original_settings = _server_state.global_settings
        original_api_key = _server_state.api_key
        original_slot_store = getattr(_server_state, "slot_store", None)
        try:
            self._configure_slot_runtime(_server_state, slot_dir, mock_engine_pool, tmp_path)

            def slow_large_payload(*args, **kwargs):
                time.sleep(0.3)
                return b"x" * (100 * 1024 * 1024), {
                    "n_tokens": 100_000,
                    "tensors": [{"name": "layer_0", "dtype": "f16", "shape": [100000]}],
                    "cache_class": "paged_ssd",
                }

            monkeypatch.setattr(
                server_module,
                "_serialize_slot_payload",
                slow_large_payload,
            )

            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                save_task = asyncio.create_task(
                    ac.post(
                        "/slots/0?action=save",
                        json={"filename": "slot-large.kvslot", "model": "test-model"},
                    )
                )

                await asyncio.sleep(0.02)
                started = time.perf_counter()
                health = await ac.get("/health")
                elapsed = time.perf_counter() - started

                assert health.status_code == 200
                assert elapsed < 0.05

                save_resp = await save_task
                assert save_resp.status_code == 200
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store

    def test_save_returns_n_saved_token_count(self, tmp_path, mock_engine_pool, monkeypatch):
        from omlx.server import app, _server_state

        slot_dir = tmp_path / "slots"
        slot_dir.mkdir()

        original_pool = _server_state.engine_pool
        original_default = _server_state.default_model
        original_settings = _server_state.global_settings
        original_api_key = _server_state.api_key
        original_slot_store = getattr(_server_state, "slot_store", None)
        try:
            self._configure_slot_runtime(_server_state, slot_dir, mock_engine_pool, tmp_path)
            self._patch_minimal_slot_payload(monkeypatch)
            client = TestClient(app)

            response = client.post(
                "/slots/0?action=save",
                json={"filename": "slot-return.kvslot", "model": "test-model"},
            )
            assert response.status_code == 200
            body = response.json()
            assert body["id_slot"] == 0
            assert body["model"] == "test-model"
            assert body["filename"] == "slot-return.kvslot"
            assert body["n_saved"] == 123
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store

    def test_save_busy_returns_409_not_423_when_state_is_restoring(
        self, tmp_path, mock_engine_pool, monkeypatch
    ):
        from omlx.server import _get_slot_store, _server_state, app
        from omlx.slot_store import SlotState

        slot_dir = tmp_path / "slots"
        slot_dir.mkdir()

        original_pool = _server_state.engine_pool
        original_default = _server_state.default_model
        original_settings = _server_state.global_settings
        original_api_key = _server_state.api_key
        original_slot_store = getattr(_server_state, "slot_store", None)
        try:
            self._configure_slot_runtime(_server_state, slot_dir, mock_engine_pool, tmp_path)
            self._patch_minimal_slot_payload(monkeypatch)
            _get_slot_store()
            _server_state.slot_store._states[0] = SlotState.RESTORING  # noqa: SLF001
            client = TestClient(app)

            response = client.post(
                "/slots/0?action=save",
                json={"filename": "slot-while-restoring.kvslot", "model": "test-model"},
            )
            assert response.status_code == 409
            assert response.json()["detail"]["error"]["code"] == "slot_busy"
            assert response.json()["detail"]["error"]["state"] == "RESTORING"
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store


class TestSlotRestoreEndpoint:
    def _configure_slot_runtime(self, server_state, slot_dir, pool, tmp_path) -> None:
        model_dir = tmp_path / "model-artifacts"
        model_dir.mkdir(exist_ok=True)
        (model_dir / "config.json").write_text('{"model":"test-model"}', encoding="utf-8")

        class _Entry:
            def __init__(self, path):
                self.model_path = str(path)
                self.engine = None

        entry = _Entry(model_dir)
        pool.get_entry = lambda model_id: entry

        settings = GlobalSettings()
        settings.slot_save_path = str(slot_dir)
        settings.scheduler.max_concurrent_requests = 1
        server_state.global_settings = settings
        server_state.engine_pool = pool
        server_state.default_model = "test-model"
        server_state.api_key = None

    def _write_restore_files(self, slot_dir, filename, payload, manifest_dict):
        (slot_dir / filename).write_bytes(payload)
        (slot_dir / f"{filename}.manifest.json").write_text(
            json.dumps(manifest_dict, separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )

    def _manifest_dict(self, fingerprint: str, ctx_size: int, n_tokens: int = 17):
        return {
            "slot_format_version": 1,
            "model_fingerprint": fingerprint,
            "model_id": "test-model",
            "ctx_size": ctx_size,
            "n_tokens": n_tokens,
            "tensors": [{"name": "layer_0", "dtype": "f16", "shape": [1, 2]}],
            "cache_class": "paged_ssd",
            "producer": {"mlx_version": "0.0.0", "omlx_cache_format_version": "v1"},
        }

    def test_restore_rejects_absolute_path(self, tmp_path, mock_engine_pool):
        from omlx.server import _server_state, app

        slot_dir = tmp_path / "slots"
        slot_dir.mkdir()

        original_pool = _server_state.engine_pool
        original_default = _server_state.default_model
        original_settings = _server_state.global_settings
        original_api_key = _server_state.api_key
        original_slot_store = getattr(_server_state, "slot_store", None)
        try:
            self._configure_slot_runtime(_server_state, slot_dir, mock_engine_pool, tmp_path)
            client = TestClient(app)

            response = client.post(
                "/slots/0?action=restore",
                json={"filename": "/etc/passwd", "model": "test-model"},
            )
            assert response.status_code == 400
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store

    def test_restore_rejects_dotdot_traversal(self, tmp_path, mock_engine_pool):
        from omlx.server import _server_state, app

        slot_dir = tmp_path / "slots"
        slot_dir.mkdir()

        original_pool = _server_state.engine_pool
        original_default = _server_state.default_model
        original_settings = _server_state.global_settings
        original_api_key = _server_state.api_key
        original_slot_store = getattr(_server_state, "slot_store", None)
        try:
            self._configure_slot_runtime(_server_state, slot_dir, mock_engine_pool, tmp_path)
            client = TestClient(app)

            response = client.post(
                "/slots/0?action=restore",
                json={"filename": "../escape.kvslot", "model": "test-model"},
            )
            assert response.status_code == 400
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store

    def test_restore_returns_404_when_file_missing(self, tmp_path, mock_engine_pool):
        from omlx.server import _server_state, app

        slot_dir = tmp_path / "slots"
        slot_dir.mkdir()

        original_pool = _server_state.engine_pool
        original_default = _server_state.default_model
        original_settings = _server_state.global_settings
        original_api_key = _server_state.api_key
        original_slot_store = getattr(_server_state, "slot_store", None)
        try:
            self._configure_slot_runtime(_server_state, slot_dir, mock_engine_pool, tmp_path)
            client = TestClient(app)

            response = client.post(
                "/slots/0?action=restore",
                json={"filename": "missing.kvslot", "model": "test-model"},
            )
            assert response.status_code == 404
            assert response.json()["detail"]["error"]["code"] == "slot_file_not_found"
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store

    def test_restore_busy_returns_423_when_generating(self, tmp_path, mock_engine_pool):
        from omlx.server import _get_slot_store, _server_state, app
        from omlx.slot_store import SlotState

        slot_dir = tmp_path / "slots"
        slot_dir.mkdir()

        original_pool = _server_state.engine_pool
        original_default = _server_state.default_model
        original_settings = _server_state.global_settings
        original_api_key = _server_state.api_key
        original_slot_store = getattr(_server_state, "slot_store", None)
        try:
            self._configure_slot_runtime(_server_state, slot_dir, mock_engine_pool, tmp_path)
            _get_slot_store()
            _server_state.slot_store._states[0] = SlotState.GENERATING  # noqa: SLF001
            client = TestClient(app)

            response = client.post(
                "/slots/0?action=restore",
                json={"filename": "slot.kvslot", "model": "test-model"},
            )
            assert response.status_code == 423
            detail = response.json()["detail"]
            assert detail["error"]["code"] == "slot_busy_restore"
            assert detail["error"]["state"] == "GENERATING"
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store

    def test_restore_busy_returns_423_when_already_restoring(self, tmp_path, mock_engine_pool):
        from omlx.server import _get_slot_store, _server_state, app
        from omlx.slot_store import SlotState

        slot_dir = tmp_path / "slots"
        slot_dir.mkdir()

        original_pool = _server_state.engine_pool
        original_default = _server_state.default_model
        original_settings = _server_state.global_settings
        original_api_key = _server_state.api_key
        original_slot_store = getattr(_server_state, "slot_store", None)
        try:
            self._configure_slot_runtime(_server_state, slot_dir, mock_engine_pool, tmp_path)
            _get_slot_store()
            _server_state.slot_store._states[0] = SlotState.RESTORING  # noqa: SLF001
            client = TestClient(app)

            response = client.post(
                "/slots/0?action=restore",
                json={"filename": "slot.kvslot", "model": "test-model"},
            )
            assert response.status_code == 423
            detail = response.json()["detail"]
            assert detail["error"]["code"] == "slot_busy_restore"
            assert detail["error"]["state"] == "RESTORING"
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store

    def test_restore_fingerprint_mismatch_returns_409_with_field(
        self, tmp_path, mock_engine_pool, monkeypatch
    ):
        import omlx.server as server_module
        from omlx.server import _server_state, app
        from omlx.slot_store import compute_model_fingerprint

        slot_dir = tmp_path / "slots"
        slot_dir.mkdir()

        original_pool = _server_state.engine_pool
        original_default = _server_state.default_model
        original_settings = _server_state.global_settings
        original_api_key = _server_state.api_key
        original_slot_store = getattr(_server_state, "slot_store", None)
        try:
            self._configure_slot_runtime(_server_state, slot_dir, mock_engine_pool, tmp_path)
            model_dir = tmp_path / "model-artifacts"
            fingerprint = compute_model_fingerprint(model_dir)
            manifest = self._manifest_dict(fingerprint="different", ctx_size=32768, n_tokens=5)
            self._write_restore_files(slot_dir, "slot.kvslot", b"{}", manifest)
            monkeypatch.setattr(server_module, "_apply_slot_restore_payload", lambda *args, **kwargs: 5)
            client = TestClient(app)

            response = client.post(
                "/slots/0?action=restore",
                json={"filename": "slot.kvslot", "model": "test-model"},
            )
            assert response.status_code == 409
            detail = response.json()["detail"]["error"]
            assert detail["code"] == "slot_guard_mismatch"
            assert detail["details"]["field"] == "model_fingerprint"
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store

    def test_restore_ctx_size_mismatch_returns_409(
        self, tmp_path, mock_engine_pool, monkeypatch
    ):
        import omlx.server as server_module
        from omlx.server import _server_state, app
        from omlx.slot_store import compute_model_fingerprint

        slot_dir = tmp_path / "slots"
        slot_dir.mkdir()

        original_pool = _server_state.engine_pool
        original_default = _server_state.default_model
        original_settings = _server_state.global_settings
        original_api_key = _server_state.api_key
        original_slot_store = getattr(_server_state, "slot_store", None)
        try:
            self._configure_slot_runtime(_server_state, slot_dir, mock_engine_pool, tmp_path)
            model_dir = tmp_path / "model-artifacts"
            fingerprint = compute_model_fingerprint(model_dir)
            manifest = self._manifest_dict(fingerprint=fingerprint, ctx_size=999, n_tokens=5)
            self._write_restore_files(slot_dir, "slot.kvslot", b"{}", manifest)
            monkeypatch.setattr(server_module, "_apply_slot_restore_payload", lambda *args, **kwargs: 5)
            client = TestClient(app)

            response = client.post(
                "/slots/0?action=restore",
                json={"filename": "slot.kvslot", "model": "test-model"},
            )
            assert response.status_code == 409
            detail = response.json()["detail"]["error"]
            assert detail["code"] == "slot_guard_mismatch"
            assert detail["details"]["field"] == "ctx_size"
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store

    @pytest.mark.asyncio
    async def test_restore_event_loop_not_blocked(
        self, tmp_path, mock_engine_pool, monkeypatch
    ):
        httpx = pytest.importorskip("httpx")
        import omlx.server as server_module
        from omlx.server import _server_state, app
        from omlx.slot_store import compute_model_fingerprint

        slot_dir = tmp_path / "slots"
        slot_dir.mkdir()

        original_pool = _server_state.engine_pool
        original_default = _server_state.default_model
        original_settings = _server_state.global_settings
        original_api_key = _server_state.api_key
        original_slot_store = getattr(_server_state, "slot_store", None)
        try:
            self._configure_slot_runtime(_server_state, slot_dir, mock_engine_pool, tmp_path)
            model_dir = tmp_path / "model-artifacts"
            fingerprint = compute_model_fingerprint(model_dir)
            manifest = self._manifest_dict(fingerprint=fingerprint, ctx_size=32768, n_tokens=5)
            self._write_restore_files(
                slot_dir,
                "slot-large.kvslot",
                b"x" * (100 * 1024 * 1024),
                manifest,
            )

            def slow_apply(*args, **kwargs):
                time.sleep(0.3)
                return 5

            monkeypatch.setattr(server_module, "_apply_slot_restore_payload", slow_apply)
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                restore_task = asyncio.create_task(
                    ac.post(
                        "/slots/0?action=restore",
                        json={"filename": "slot-large.kvslot", "model": "test-model"},
                    )
                )

                await asyncio.sleep(0.02)
                started = time.perf_counter()
                health = await ac.get("/health")
                elapsed = time.perf_counter() - started

                assert health.status_code == 200
                assert elapsed < 0.05

                restore_resp = await restore_task
                assert restore_resp.status_code == 200
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store

    def test_restore_returns_n_restored_token_count(
        self, tmp_path, mock_engine_pool, monkeypatch
    ):
        import omlx.server as server_module
        from omlx.server import _server_state, app
        from omlx.slot_store import compute_model_fingerprint

        slot_dir = tmp_path / "slots"
        slot_dir.mkdir()

        original_pool = _server_state.engine_pool
        original_default = _server_state.default_model
        original_settings = _server_state.global_settings
        original_api_key = _server_state.api_key
        original_slot_store = getattr(_server_state, "slot_store", None)
        try:
            self._configure_slot_runtime(_server_state, slot_dir, mock_engine_pool, tmp_path)
            model_dir = tmp_path / "model-artifacts"
            fingerprint = compute_model_fingerprint(model_dir)
            manifest = self._manifest_dict(fingerprint=fingerprint, ctx_size=32768, n_tokens=123)
            self._write_restore_files(slot_dir, "slot-roundtrip.kvslot", b"{}", manifest)
            monkeypatch.setattr(server_module, "_apply_slot_restore_payload", lambda *args, **kwargs: 123)
            client = TestClient(app)

            response = client.post(
                "/slots/0?action=restore",
                json={"filename": "slot-roundtrip.kvslot", "model": "test-model"},
            )
            assert response.status_code == 200
            body = response.json()
            assert body["id_slot"] == 0
            assert body["model"] == "test-model"
            assert body["filename"] == "slot-roundtrip.kvslot"
            assert body["n_restored"] == 123
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store

    def test_v2a_save_restore_endpoint_round_trip_via_v1_alias(
        self, tmp_path, mock_engine_pool, monkeypatch
    ):
        import mlx.core as mx
        import omlx.server as server_module
        from mlx_lm.models.cache import KVCache, load_prompt_cache
        from omlx.server import _server_state, app

        slot_dir = tmp_path / "slots"
        slot_dir.mkdir()
        original_pool = _server_state.engine_pool
        original_default = _server_state.default_model
        original_settings = _server_state.global_settings
        original_api_key = _server_state.api_key
        original_slot_store = getattr(_server_state, "slot_store", None)
        original_scratch = getattr(_server_state, "_slot_v2a_last_loaded", None)
        try:
            self._configure_slot_runtime(_server_state, slot_dir, mock_engine_pool, tmp_path)
            _server_state._slot_v2a_last_loaded = None

            def fake_extract_slot_request_payload(_entry):
                cache_layers = []
                for i in range(2):
                    cache = KVCache()
                    keys = mx.full((1, 1, 3, 2), i + 1, dtype=mx.float16)
                    values = mx.full((1, 1, 3, 2), i + 2, dtype=mx.float16)
                    cache.update_and_fetch(keys, values)
                    cache_layers.append(cache)
                return cache_layers, 11, [4]

            monkeypatch.setattr(
                server_module,
                "_extract_slot_request_payload",
                fake_extract_slot_request_payload,
            )

            client = TestClient(app)
            save_response = client.post(
                "/slots/0?action=save",
                json={"filename": "slot-roundtrip.safetensors", "model": "test-model"},
            )
            assert save_response.status_code == 200, save_response.text
            n_saved = save_response.json()["n_saved"]

            restore_response = client.post(
                "/slots/0?action=restore",
                json={"filename": "slot-roundtrip.safetensors", "model": "test-model"},
            )
            assert restore_response.status_code == 200
            assert restore_response.json()["n_restored"] == n_saved

            saved_path = slot_dir / "slot-roundtrip.safetensors"
            loaded_cache, file_metadata = load_prompt_cache(
                str(saved_path),
                return_metadata=True,
            )
            assert len(loaded_cache) == 2
            assert file_metadata.get("cached_tokens") == str(n_saved)
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store
            _server_state._slot_v2a_last_loaded = original_scratch

class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_healthy_status(self, client):
        """Test that health endpoint returns healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_contains_required_fields(self, client):
        """Test that health response contains required fields."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "default_model" in data
        assert "engine_pool" in data

    def test_health_engine_pool_info(self, client):
        """Test that health response contains engine pool info."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        pool_info = data["engine_pool"]
        assert "model_count" in pool_info
        assert "loaded_count" in pool_info
        assert "max_model_memory" in pool_info
        assert "current_model_memory" in pool_info


class TestModelsEndpoint:
    """Tests for the /v1/models endpoint."""

    def test_models_returns_list(self, client):
        """Test that models endpoint returns a list."""
        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert "data" in data

    def test_models_format(self, client):
        """Test that model entries have correct format."""
        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        if data["data"]:
            model = data["data"][0]
            assert "id" in model
            assert "object" in model


class TestResponsesEndpoint:
    def test_response_endpoint_recovers_tool_call_from_thinking(self, tmp_path):
        from omlx.server import app, _server_state

        state_dir = tmp_path / "response-state"
        engine = RecordingResponsesEngine(outputs=[
            MockGenerationOutput(
                text=(
                    "<think>Need to inspect first."
                    '<tool_call>{"name":"exec_command","arguments":{"cmd":"ls"}}</tool_call>'
                    "Then continue.</think>"
                ),
                finish_reason="stop",
            ),
        ])
        pool = MockEnginePool(llm_engine=engine)

        original_pool = _server_state.engine_pool
        original_default = _server_state.default_model
        original_store = _server_state.responses_store
        try:
            _server_state.engine_pool = pool
            _server_state.default_model = "test-model"
            _server_state.responses_store = ResponseStore(state_dir=state_dir)
            client = TestClient(app)

            response = client.post(
                "/v1/responses",
                json={
                    "model": "test-model",
                    "input": "Explore the code",
                    "tools": [{
                        "type": "function",
                        "name": "exec_command",
                        "description": "Run a shell command",
                        "parameters": {
                            "type": "object",
                            "properties": {"cmd": {"type": "string"}},
                            "required": ["cmd"],
                        },
                    }],
                },
            )
            assert response.status_code == 200

            output_items = response.json()["output"]
            message_items = [item for item in output_items if item["type"] == "message"]
            function_items = [item for item in output_items if item["type"] == "function_call"]

            assert len(message_items) == 1
            assert message_items[0]["content"][0]["text"] == ""
            assert "<tool_call>" not in message_items[0]["content"][0]["text"]
            assert len(function_items) == 1
            assert function_items[0]["name"] == "exec_command"
            assert function_items[0]["arguments"] == '{"cmd": "ls"}'
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.responses_store = original_store

    def test_previous_response_id_persists_across_store_restart(self, tmp_path):
        from omlx.server import app, _server_state

        state_dir = tmp_path / "response-state"
        engine = RecordingResponsesEngine(outputs=[
            MockGenerationOutput(
                text="",
                finish_reason="tool_calls",
                tool_calls=[{
                    "id": "call_123",
                    "name": "exec_command",
                    "arguments": '{"cmd":"ls"}',
                }],
            ),
            MockGenerationOutput(text="Done.", finish_reason="stop"),
        ])
        pool = MockEnginePool(llm_engine=engine)

        original_pool = _server_state.engine_pool
        original_default = _server_state.default_model
        original_store = _server_state.responses_store
        try:
            _server_state.engine_pool = pool
            _server_state.default_model = "test-model"
            _server_state.responses_store = ResponseStore(state_dir=state_dir)
            client = TestClient(app)

            first = client.post(
                "/v1/responses",
                json={"model": "test-model", "input": "Explore the code"},
            )
            assert first.status_code == 200
            first_id = first.json()["id"]

            # Simulate a restart by rebuilding the store from disk.
            _server_state.responses_store = ResponseStore(state_dir=state_dir)

            second = client.post(
                "/v1/responses",
                json={
                    "model": "test-model",
                    "previous_response_id": first_id,
                    "input": [
                        {
                            "type": "function_call_output",
                            "call_id": "call_123",
                            "output": "file1.txt\nfile2.txt",
                        }
                    ],
                },
            )
            assert second.status_code == 200

            replayed = engine.recorded_messages[1]
            assert replayed[0] == {"role": "user", "content": "Explore the code"}
            assert replayed[1]["role"] == "assistant"
            assert replayed[1]["tool_calls"][0]["id"] == "call_123"
            assert replayed[2] == {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": "file1.txt\nfile2.txt",
            }
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.responses_store = original_store

    def test_missing_previous_response_id_returns_404(self, tmp_path):
        from omlx.server import app, _server_state

        engine = RecordingResponsesEngine(outputs=[MockGenerationOutput(text="Done.")])
        pool = MockEnginePool(llm_engine=engine)

        original_pool = _server_state.engine_pool
        original_default = _server_state.default_model
        original_store = _server_state.responses_store
        try:
            _server_state.engine_pool = pool
            _server_state.default_model = "test-model"
            _server_state.responses_store = ResponseStore(
                state_dir=tmp_path / "response-state"
            )
            client = TestClient(app)

            response = client.post(
                "/v1/responses",
                json={
                    "model": "test-model",
                    "previous_response_id": "resp_missing",
                    "input": "Continue",
                },
            )
            assert response.status_code == 404
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.responses_store = original_store


class TestModelsStatusEndpoint:
    """Tests for the /v1/models/status endpoint."""

    def test_models_status_returns_details(self, client):
        """Test that models status returns detailed info."""
        response = client.get("/v1/models/status")

        assert response.status_code == 200
        data = response.json()
        assert "models" in data


class TestCompletionEndpoint:
    """Tests for the /v1/completions endpoint."""

    def test_completion_basic_request(self, client):
        """Test basic completion request."""
        response = client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "Hello, world!",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "text" in data["choices"][0]

    def test_completion_response_format(self, client):
        """Test completion response has correct format."""
        response = client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "Test prompt",
                "max_tokens": 100,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "text_completion"
        assert "model" in data
        assert "choices" in data
        assert "usage" in data

    def test_completion_with_list_prompt(self, client):
        """Test completion with list of prompts."""
        response = client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": ["First prompt", "Second prompt"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "choices" in data

    def test_completion_includes_cached_tokens_on_cache_hit(self, client, mock_llm_engine):
        """Non-streaming completion responses should expose cached token counts."""
        mock_llm_engine.generate = AsyncMock(return_value=MockGenerationOutput(
            text="Generated response.",
            prompt_tokens=2215,
            completion_tokens=5,
            cached_tokens=2048,
        ))

        response = client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "Cache hit prompt",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["usage"]["prompt_tokens_details"]["cached_tokens"] == 2048


class TestChatCompletionEndpoint:
    """Tests for the /v1/chat/completions endpoint."""

    def test_chat_completion_basic(self, client):
        """Test basic chat completion request."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0

    def test_chat_completion_response_format(self, client):
        """Test chat completion response format."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hi!"},
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert "model" in data
        assert "choices" in data
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "usage" in data

    def test_chat_completion_with_parameters(self, client):
        """Test chat completion with sampling parameters."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Test"}],
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 256,
            },
        )

        assert response.status_code == 200

    def test_chat_completion_includes_cached_tokens_on_cache_hit(self, client, mock_llm_engine):
        """Non-streaming chat responses should expose cached token counts."""
        mock_llm_engine.chat = AsyncMock(return_value=MockGenerationOutput(
            text="Chat response.",
            prompt_tokens=2215,
            completion_tokens=5,
            cached_tokens=2048,
            finish_reason="stop",
            finished=True,
        ))

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Cache hit prompt"}],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["usage"]["prompt_tokens_details"]["cached_tokens"] == 2048

    def test_chat_completion_sanitizes_reasoning_tool_call_markup(self, client, mock_llm_engine):
        """Thinking-only tool calls should become structured tool_calls without leaked markup."""
        mock_llm_engine.chat = AsyncMock(return_value=MockGenerationOutput(
            text=(
                "<think>Need to inspect first."
                '<tool_call>{"name":"get_weather","arguments":{"city":"SF"}}</tool_call>'
                "Then continue.</think>"
            ),
            prompt_tokens=10,
            completion_tokens=5,
            finish_reason="stop",
            finished=True,
        ))

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }],
            },
        )

        assert response.status_code == 200
        data = response.json()
        message = data["choices"][0]["message"]

        assert message["reasoning_content"] == "Need to inspect first.Then continue."
        assert "<tool_call>" not in message["reasoning_content"]
        assert len(message["tool_calls"]) == 1
        assert message["tool_calls"][0]["function"]["name"] == "get_weather"
        assert message["tool_calls"][0]["function"]["arguments"] == '{"city": "SF"}'
        assert data["choices"][0]["finish_reason"] == "tool_calls"


class TestAnthropicMessagesEndpoint:
    """Tests for the /v1/messages endpoint (Anthropic format)."""

    def test_anthropic_messages_basic(self, client):
        """Test basic Anthropic messages request."""
        response = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"

    def test_anthropic_messages_response_format(self, client):
        """Test Anthropic messages response format."""
        response = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hi there!"}],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "content" in data
        assert "usage" in data
        assert "input_tokens" in data["usage"]
        assert "output_tokens" in data["usage"]

    def test_anthropic_messages_with_system(self, client):
        """Test Anthropic messages with system prompt."""
        response = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 1024,
                "system": "You are a helpful assistant.",
                "messages": [{"role": "user", "content": "Hello!"}],
            },
        )

        assert response.status_code == 200

    def test_anthropic_messages_sanitize_thinking_tool_call_markup(self, client, mock_llm_engine):
        """Anthropic thinking blocks should not expose raw tool-call markup."""
        mock_llm_engine.chat = AsyncMock(return_value=MockGenerationOutput(
            text=(
                "<think>Need to inspect first."
                '<tool_call>{"name":"get_weather","arguments":{"city":"SF"}}</tool_call>'
                "Then continue.</think>"
            ),
            prompt_tokens=10,
            completion_tokens=5,
            finish_reason="stop",
            finished=True,
        ))

        response = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hi"}],
                "tools": [{
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }],
            },
        )

        assert response.status_code == 200
        data = response.json()
        thinking_blocks = [block for block in data["content"] if block["type"] == "thinking"]
        tool_use_blocks = [block for block in data["content"] if block["type"] == "tool_use"]

        assert len(thinking_blocks) == 1
        assert thinking_blocks[0]["thinking"] == "Need to inspect first.Then continue."
        assert "<tool_call>" not in thinking_blocks[0]["thinking"]
        assert len(tool_use_blocks) == 1
        assert tool_use_blocks[0]["name"] == "get_weather"
        assert tool_use_blocks[0]["input"] == {"city": "SF"}
        assert data["stop_reason"] == "tool_use"


class TestEmbeddingsEndpoint:
    """Tests for the /v1/embeddings endpoint."""

    def test_embeddings_single_input(self, client, mock_engine_pool):
        """Test embeddings with single input."""
        mock_engine_pool._models.append(
            {"id": "test-embed-model", "loaded": True, "pinned": False, "size": 500000}
        )

        response = client.post(
            "/v1/embeddings",
            json={
                "model": "test-embed-model",
                "input": "Hello, world!",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert "data" in data
        assert len(data["data"]) == 1
        assert data["data"][0]["object"] == "embedding"

    def test_embeddings_multiple_inputs(self, client, mock_engine_pool):
        """Test embeddings with multiple inputs."""
        mock_engine_pool._models.append(
            {"id": "test-embed-model", "loaded": True, "pinned": False, "size": 500000}
        )

        response = client.post(
            "/v1/embeddings",
            json={
                "model": "test-embed-model",
                "input": ["First text", "Second text"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 2

    def test_embeddings_response_format(self, client, mock_engine_pool):
        """Test embeddings response format."""
        mock_engine_pool._models.append(
            {"id": "test-embed-model", "loaded": True, "pinned": False, "size": 500000}
        )

        response = client.post(
            "/v1/embeddings",
            json={
                "model": "test-embed-model",
                "input": "Test text",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "model" in data
        assert "usage" in data
        assert "prompt_tokens" in data["usage"]
        assert "total_tokens" in data["usage"]
        assert "embedding" in data["data"][0]
        assert isinstance(data["data"][0]["embedding"], list)

    def test_embeddings_structured_items_input(self, client, mock_engine_pool):
        """Test embeddings with structured multimodal items."""
        mock_engine_pool._models.append(
            {"id": "test-embed-model", "loaded": True, "pinned": False, "size": 500000}
        )

        response = client.post(
            "/v1/embeddings",
            json={
                "model": "test-embed-model",
                "items": [
                    {"text": "hello"},
                    {"image": "https://example.com/image.jpg"},
                    {
                        "text": "hello",
                        "image": "https://example.com/image.jpg",
                    },
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 3

    def test_embeddings_rejects_mixed_input_sources(self, client, mock_engine_pool):
        """Test embeddings rejects input and items together."""
        mock_engine_pool._models.append(
            {"id": "test-embed-model", "loaded": True, "pinned": False, "size": 500000}
        )

        response = client.post(
            "/v1/embeddings",
            json={
                "model": "test-embed-model",
                "input": "hello",
                "items": [{"text": "hello"}],
            },
        )

        assert response.status_code == 422


class TestRerankEndpoint:
    """Tests for the /v1/rerank endpoint."""

    def test_rerank_basic(self, client, mock_engine_pool):
        """Test basic rerank request."""
        mock_engine_pool._models.append(
            {
                "id": "test-rerank-model",
                "loaded": True,
                "pinned": False,
                "size": 500000,
            }
        )

        response = client.post(
            "/v1/rerank",
            json={
                "model": "test-rerank-model",
                "query": "What is machine learning?",
                "documents": [
                    "ML is a subset of AI.",
                    "The weather is nice today.",
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 2

    def test_rerank_with_top_n(self, client, mock_engine_pool):
        """Test rerank with top_n parameter."""
        mock_engine_pool._models.append(
            {
                "id": "test-rerank-model",
                "loaded": True,
                "pinned": False,
                "size": 500000,
            }
        )

        response = client.post(
            "/v1/rerank",
            json={
                "model": "test-rerank-model",
                "query": "Test query",
                "documents": ["Doc 1", "Doc 2", "Doc 3"],
                "top_n": 2,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2

    def test_rerank_response_format(self, client, mock_engine_pool):
        """Test rerank response format."""
        mock_engine_pool._models.append(
            {
                "id": "test-rerank-model",
                "loaded": True,
                "pinned": False,
                "size": 500000,
            }
        )

        response = client.post(
            "/v1/rerank",
            json={
                "model": "test-rerank-model",
                "query": "Test",
                "documents": ["Document 1"],
                "return_documents": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "model" in data
        assert "results" in data
        result = data["results"][0]
        assert "index" in result
        assert "relevance_score" in result
        assert "document" in result


class TestTokenCountEndpoint:
    """Tests for the /v1/messages/count_tokens endpoint."""

    def test_token_count_basic(self, client):
        """Test basic token counting."""
        response = client.post(
            "/v1/messages/count_tokens",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello world"}],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "input_tokens" in data
        assert isinstance(data["input_tokens"], int)

    def test_token_count_with_system(self, client):
        """Test token counting with system prompt."""
        response = client.post(
            "/v1/messages/count_tokens",
            json={
                "model": "test-model",
                "system": "You are helpful.",
                "messages": [{"role": "user", "content": "Hi!"}],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "input_tokens" in data


class TestMCPEndpoints:
    """Tests for MCP-related endpoints."""

    def test_mcp_tools_empty(self, client):
        """Test MCP tools endpoint when no MCP configured."""
        response = client.get("/v1/mcp/tools")

        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert "count" in data
        assert data["count"] == 0

    def test_mcp_servers_empty(self, client):
        """Test MCP servers endpoint when no MCP configured."""
        response = client.get("/v1/mcp/servers")

        assert response.status_code == 200
        data = response.json()
        assert "servers" in data

    def test_mcp_execute_no_config(self, client):
        """Test MCP execute fails when not configured."""
        response = client.post(
            "/v1/mcp/execute",
            json={
                "tool_name": "test_tool",
                "arguments": {},
            },
        )

        # Should return 503 when MCP not configured
        assert response.status_code == 503

    def test_mcp_execute_accepts_tool_alias(self, client):
        """Test MCP execute accepts tool as an alias for tool_name."""
        from omlx.server import _server_state

        original_mcp_manager = _server_state.mcp_manager
        manager = AsyncMock()
        manager.execute_tool.return_value = MCPToolResult(
            tool_name="test_tool",
            content={"ok": True},
        )

        try:
            _server_state.mcp_manager = manager

            response = client.post(
                "/v1/mcp/execute",
                json={
                    "tool": "test_tool",
                    "arguments": {"query": "hello"},
                },
            )
        finally:
            _server_state.mcp_manager = original_mcp_manager

        assert response.status_code == 200
        assert response.json() == {
            "tool_name": "test_tool",
            "content": {"ok": True},
            "is_error": False,
            "error_message": None,
        }
        manager.execute_tool.assert_awaited_once_with(
            "test_tool",
            {"query": "hello"},
        )

    def test_mcp_execute_tool_name_field(self, client):
        """Test MCP execute happy path with tool_name field."""
        from omlx.server import _server_state

        original_mcp_manager = _server_state.mcp_manager
        manager = AsyncMock()
        manager.execute_tool.return_value = MCPToolResult(
            tool_name="my_tool",
            content="ok",
        )

        try:
            _server_state.mcp_manager = manager

            response = client.post(
                "/v1/mcp/execute",
                json={
                    "tool_name": "my_tool",
                    "arguments": {"q": "x"},
                },
            )
        finally:
            _server_state.mcp_manager = original_mcp_manager

        assert response.status_code == 200
        manager.execute_tool.assert_awaited_once_with("my_tool", {"q": "x"})

    def test_mcp_execute_tool_name_wins_over_tool(self, client):
        """Test tool_name takes precedence when both fields are present."""
        from omlx.server import _server_state

        original_mcp_manager = _server_state.mcp_manager
        manager = AsyncMock()
        manager.execute_tool.return_value = MCPToolResult(
            tool_name="canonical",
            content="ok",
        )

        try:
            _server_state.mcp_manager = manager

            response = client.post(
                "/v1/mcp/execute",
                json={
                    "tool_name": "canonical",
                    "tool": "alias_should_lose",
                    "arguments": {},
                },
            )
        finally:
            _server_state.mcp_manager = original_mcp_manager

        assert response.status_code == 200
        manager.execute_tool.assert_awaited_once_with("canonical", {})

    def test_mcp_execute_rejects_missing_tool(self, client):
        """Test MCP execute returns 422 when neither tool nor tool_name is present."""
        response = client.post(
            "/v1/mcp/execute",
            json={"arguments": {"q": "x"}},
        )

        assert response.status_code == 422


class TestErrorHandling:
    """Tests for error handling in endpoints."""

    def test_missing_model(self, client):
        """Test error when model is not specified."""
        # For Anthropic endpoint, missing model should raise validation error
        response = client.post(
            "/v1/messages",
            json={
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 422  # Validation error

    def test_empty_messages(self, client):
        """Test error when messages is empty."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [],
            },
        )

        # Empty messages may be allowed or raise error depending on implementation
        # Just verify we get a response
        assert response.status_code in [200, 400, 422]

    def test_invalid_request_format(self, client):
        """Test error for invalid request format."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "invalid_field": "test",
            },
        )

        assert response.status_code == 422


class TestJsonOutputParsing:
    """Tests for parse_json_output in non-streaming endpoints."""

    def test_chat_completion_parses_markdown_json(self, client, mock_llm_engine):
        """Markdown-wrapped JSON should be parsed when response_format=json_object."""
        import json

        mock_llm_engine.chat = AsyncMock(return_value=MockGenerationOutput(
            text='```json\n{"name": "test", "age": 25}\n```',
            prompt_tokens=10,
            completion_tokens=8,
            finish_reason="stop",
            finished=True,
        ))

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Return JSON"}],
                "response_format": {"type": "json_object"},
            },
        )

        assert response.status_code == 200
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        assert parsed == {"name": "test", "age": 25}

    def test_chat_completion_clean_json_unchanged(self, client, mock_llm_engine):
        """Already-clean JSON should pass through without corruption."""
        import json

        mock_llm_engine.chat = AsyncMock(return_value=MockGenerationOutput(
            text='{"key": "value"}',
            prompt_tokens=10,
            completion_tokens=5,
            finish_reason="stop",
            finished=True,
        ))

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Return JSON"}],
                "response_format": {"type": "json_object"},
            },
        )

        assert response.status_code == 200
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        assert parsed == {"key": "value"}

    def test_responses_parses_markdown_json(self, client, mock_llm_engine):
        """Responses API should parse markdown-wrapped JSON with text.format."""
        import json

        mock_llm_engine.chat = AsyncMock(return_value=MockGenerationOutput(
            text='```json\n{"city": "Seoul", "temp": 15}\n```',
            prompt_tokens=10,
            completion_tokens=8,
            finish_reason="stop",
            finished=True,
        ))

        response = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "Return weather JSON",
                "text": {
                    "format": {"type": "json_object"},
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        output_text = data["output"][0]["content"][0]["text"]
        parsed = json.loads(output_text)
        assert parsed == {"city": "Seoul", "temp": 15}

    def test_responses_without_format_unchanged(self, client, mock_llm_engine):
        """Responses API without text.format should return raw text."""
        mock_llm_engine.chat = AsyncMock(return_value=MockGenerationOutput(
            text="Hello, how can I help?",
            prompt_tokens=10,
            completion_tokens=5,
            finish_reason="stop",
            finished=True,
        ))

        response = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "Hi",
            },
        )

        assert response.status_code == 200
        data = response.json()
        output_text = data["output"][0]["content"][0]["text"]
        assert "Hello" in output_text
