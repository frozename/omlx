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
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from fastapi.testclient import TestClient

from omlx.api.responses_utils import ResponseStore
from omlx.engine.base import BaseEngine
from omlx.engine.embedding import EmbeddingEngine
from omlx.engine.reranker import RerankerEngine
from omlx.mcp.types import MCPToolResult
from omlx.settings import GlobalSettings
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler
from omlx.slot_store import (
    OneShotBind,
    OneShotBindTable,
    SlotManifest,
    hash_prompt_token_prefix,
)


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


def _build_prefix_cache_backed_entry(
    model_dir: Any,
    *,
    expected_tokens: list[int],
    cached_tokens: int,
):
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    class _BlockTable:
        def __init__(self, n_tokens: int) -> None:
            self.num_tokens = n_tokens
            self.block_ids = [1]

    class _PagedCache:
        def __init__(self) -> None:
            self.deleted_request_ids: list[str] = []

        def delete_block_table(self, request_id: str) -> None:
            self.deleted_request_ids.append(request_id)

    class _PrefixCache:
        def __init__(self) -> None:
            self.fetch_calls: list[tuple[str, list[int]]] = []
            self.paged_cache = _PagedCache()

            cache = KVCache()
            keys = mx.full((1, 1, max(cached_tokens, 1), 2), 1, dtype=mx.float16)
            values = mx.full((1, 1, max(cached_tokens, 1), 2), 2, dtype=mx.float16)
            cache.update_and_fetch(keys, values)
            self._cache_layers = [cache]

        def fetch_cache(self, request_id: str, tokens: list[int]):
            self.fetch_calls.append((request_id, list(tokens)))
            if list(tokens) != list(expected_tokens):
                return None, tokens
            return _BlockTable(cached_tokens), tokens[cached_tokens:]

        def reconstruct_cache(self, block_table):
            del block_table
            return self._cache_layers

    class _Scheduler:
        def __init__(self) -> None:
            self.block_aware_cache = _PrefixCache()
            self.requests = {}

        def snapshot_for_admin(self):
            return {"running_by_id": {}, "waiting": []}

    class _InnerEngine:
        def __init__(self) -> None:
            self.scheduler = _Scheduler()

    class _EngineCore:
        def __init__(self) -> None:
            self.engine = _InnerEngine()

    class _Engine:
        def __init__(self) -> None:
            self._engine = _EngineCore()

    class _Entry:
        def __init__(self) -> None:
            self.model_path = str(model_dir)
            self.engine = _Engine()
            self.preserve_thinking_default = False
            # Upstream (#1809) reads entry.model_context_length in
            # get_max_context_window(); None keeps this mock on the
            # server-default fallback these tests were written against.
            self.model_context_length = None

    entry = _Entry()
    prefix_cache = entry.engine._engine.engine.scheduler.block_aware_cache
    return entry, prefix_cache


class MockEmbeddingEngineImpl(EmbeddingEngine):
    """Mock embedding engine for testing that inherits from EmbeddingEngine."""

    def __init__(self, model_name: str = "test-embedding-model"):
        # Don't call super().__init__ to avoid loading real model
        self._model_name = model_name
        self._model = None  # Set as None but present
        self.calls: List[Dict[str, Any]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def embed(self, texts, **kwargs) -> MockEmbeddingOutput:
        self.calls.append({"texts": list(texts), "kwargs": dict(kwargs)})
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

    def count_chat_tokens(
        self, messages: List[Dict], tools=None, chat_template_kwargs=None, **kwargs
    ) -> int:
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
        self._entries: Dict[str, Any] = {}
        self.get_engine_calls: List[Dict[str, Any]] = []
        self.release_calls: List[str] = []
        self.abort_requested_models: set[str] = set()

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
        return self._entries.get(model_id)

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

    async def get_engine(
        self,
        model_id: str,
        _lease: bool = False,
        runtime_settings=None,
    ):
        # _lease mirrors the real EnginePool's acquire-vs-use lease (#1667);
        # the mock has no eviction so it just accepts the flag.
        # runtime_settings mirrors exposed-profile variant loads.
        self.get_engine_calls.append(
            {
                "model_id": model_id,
                "_lease": _lease,
                "runtime_settings": runtime_settings,
            }
        )
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

    async def release_engine(self, model_id: str) -> None:
        # No-op release counterpart of the in-use lease (#1667).
        self.release_calls.append(model_id)
        return None

    def is_abort_requested(self, model_id: str) -> bool:
        return model_id in self.abort_requested_models


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
                self.preserve_thinking_default = False
                # Upstream (#1809) reads entry.model_context_length in
                # get_max_context_window(); None keeps this mock on the
                # server-default fallback these tests were written against.
                self.model_context_length = None

        entry = _Entry(model_dir)
        pool.get_entry = lambda model_id: entry

        settings = GlobalSettings()
        settings.slot_save_path = str(slot_dir)
        settings.scheduler.max_concurrent_requests = 1
        server_state.global_settings = settings
        server_state.engine_pool = pool
        server_state.default_model = "test-model"
        server_state.api_key = None
        server_state.one_shot_bind_table = OneShotBindTable()

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
            assert body["error"]["message"] == "slot save failed"
            assert body["error"]["details"]["exception_type"] == "RuntimeError"
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

    def test_v2_5b_save_with_explicit_prompt_tokens_uses_prefix_cache(
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
            prompt_tokens = [11, 22, 33, 44, 55]
            entry, prefix_cache = _build_prefix_cache_backed_entry(
                tmp_path / "model-artifacts",
                expected_tokens=prompt_tokens,
                cached_tokens=3,
            )
            mock_engine_pool.get_entry = lambda model_id: entry
            monkeypatch.setattr(
                server_module,
                "_serialize_slot_payload",
                lambda entry_arg, model_arg, prompt_token_ids=None: (
                    b"payload",
                    {
                        "n_tokens": 3,
                        "tensors": [{"name": "layer_0", "dtype": "f16", "shape": [1, 2]}],
                        "cache_class": "paged_ssd",
                        "prompt_prefix_sha256": (
                            hash_prompt_token_prefix(prompt_token_ids[:3])
                            if prompt_token_ids is not None
                            else None
                        ),
                    },
                ),
            )
            client = TestClient(app)

            response = client.post(
                "/slots/0?action=save",
                json={
                    "model": "test-model",
                    "request_handle": "prefix-a",
                    "prompt_tokens": prompt_tokens,
                },
            )
            assert response.status_code == 200, response.text
            assert response.json()["n_saved"] == 3
            # Ensure prompt_tokens path went through extraction by invoking it directly.
            _, extracted_n_tokens, extracted_prefix = server_module._extract_slot_request_payload(
                entry,
                model_id="test-model",
                prompt_token_ids=prompt_tokens,
            )
            assert extracted_n_tokens == 3
            assert extracted_prefix == prompt_tokens[:3]
            assert len(prefix_cache.fetch_calls) >= 1
            assert prefix_cache.fetch_calls[0][1] == prompt_tokens
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store

    def test_v2_5b_save_writes_slot_format_version_2_when_prompt_tokens_provided(
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
            prompt_tokens = [101, 102, 103, 104]
            entry, _prefix_cache = _build_prefix_cache_backed_entry(
                tmp_path / "model-artifacts",
                expected_tokens=prompt_tokens,
                cached_tokens=3,
            )
            mock_engine_pool.get_entry = lambda model_id: entry
            monkeypatch.setattr(
                server_module,
                "_serialize_slot_payload",
                lambda entry_arg, model_arg, prompt_token_ids=None: (
                    b"payload",
                    {
                        "n_tokens": 3,
                        "tensors": [{"name": "layer_0", "dtype": "f16", "shape": [1, 2]}],
                        "cache_class": "paged_ssd",
                        "prompt_prefix_sha256": hash_prompt_token_prefix(
                            prompt_token_ids[:3]
                        ),
                    },
                ),
            )
            client = TestClient(app)

            response = client.post(
                "/slots/0?action=save",
                json={
                    "model": "test-model",
                    "request_handle": "prefix-b",
                    "prompt_tokens": prompt_tokens,
                },
            )
            assert response.status_code == 200, response.text
            manifest = json.loads(
                (slot_dir / "prefix-b.kvslot.manifest.json").read_text(encoding="utf-8")
            )
            assert manifest["slot_format_version"] == 2
            assert manifest["prompt_prefix_sha256"] == hash_prompt_token_prefix(
                prompt_tokens[:3]
            )
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store

    def test_v2_5b_save_legacy_path_without_prompt_tokens_writes_format_version_1(
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
            self._patch_minimal_slot_payload(monkeypatch)
            client = TestClient(app)

            response = client.post(
                "/slots/0?action=save",
                json={"model": "test-model", "request_handle": "legacy-b"},
            )
            assert response.status_code == 200, response.text
            manifest = json.loads(
                (slot_dir / "legacy-b.kvslot.manifest.json").read_text(encoding="utf-8")
            )
            assert manifest["slot_format_version"] == 1
            assert manifest.get("prompt_prefix_sha256") is None
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store

    def test_v2_phase1a_save_accepts_request_handle(
        self, tmp_path, mock_engine_pool, monkeypatch
    ):
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
                json={"model": "test-model", "request_handle": "my_handle"},
            )
            assert response.status_code == 200
            body = response.json()
            assert body["request_handle"] == "my_handle"
            assert body["filename"] == "my_handle.kvslot"
            assert (slot_dir / "my_handle.kvslot").exists()
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store

    def test_v2_phase1a_save_rejects_invalid_request_handle_charset(
        self, tmp_path, mock_engine_pool, monkeypatch
    ):
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
                json={"model": "test-model", "request_handle": "../escape"},
            )
            assert response.status_code == 400
            assert response.json()["detail"] == "invalid_request_handle"
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store

    def test_v2_phase1a_save_rejects_request_handle_too_long(
        self, tmp_path, mock_engine_pool, monkeypatch
    ):
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
                json={"model": "test-model", "request_handle": "a" * 129},
            )
            assert response.status_code == 400
            assert response.json()["detail"] == "invalid_request_handle"
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store

    def test_v2_phase1a_save_rejects_request_handle_empty(
        self, tmp_path, mock_engine_pool, monkeypatch
    ):
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
                json={"model": "test-model", "request_handle": ""},
            )
            assert response.status_code == 400
            assert response.json()["detail"] == "invalid_request_handle"
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store

    def test_v2_phase1a_save_rejects_request_handle_with_kvslot_suffix(
        self, tmp_path, mock_engine_pool, monkeypatch
    ):
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
                json={"model": "test-model", "request_handle": "abc.kvslot"},
            )
            assert response.status_code == 400
            detail = response.json()["detail"]["error"]
            assert detail["code"] == "invalid_request_handle"
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store

    def test_v2_phase1a_save_rejects_inconsistent_handle_filename(
        self, tmp_path, mock_engine_pool, monkeypatch
    ):
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
                json={"model": "test-model", "filename": "a.kvslot", "request_handle": "b"},
            )
            assert response.status_code == 400
            assert response.json()["detail"] == "inconsistent_handle_filename"
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store

    def test_v2_5a_save_requires_explicit_handle_when_no_filename(
        self, tmp_path, mock_engine_pool, monkeypatch
    ):
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

            response = client.post("/slots/0?action=save", json={"model": "test-model"})
            assert response.status_code == 400
            detail = response.json()["detail"]["error"]
            assert detail["code"] == "request_handle_required"
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store

    def test_v2_5a_save_still_works_with_filename_alone(
        self, tmp_path, mock_engine_pool, monkeypatch
    ):
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
                json={"filename": "x.kvslot", "model": "test-model"},
            )
            assert response.status_code == 200
            body = response.json()
            assert body["request_handle"] == "x"
            assert body["filename"] == "x.kvslot"
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store

    def test_v2_5a_save_rejects_filename_without_kvslot_suffix(
        self, tmp_path, mock_engine_pool, monkeypatch
    ):
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
                json={"filename": "x.safetensors", "model": "test-model"},
            )
            assert response.status_code == 400
            detail = response.json()["detail"]["error"]
            assert detail["code"] == "invalid_slot_filename"
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store

    def test_v2_5a_save_works_with_explicit_handle(
        self, tmp_path, mock_engine_pool, monkeypatch
    ):
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
                json={"model": "test-model", "request_handle": "abc"},
            )
            assert response.status_code == 200
            body = response.json()
            assert body["request_handle"] == "abc"
            assert body["filename"] == "abc.kvslot"
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store

    def test_v2_phase1a_capability_bit_present_when_slot_api_enabled(
        self, tmp_path, mock_engine_pool, monkeypatch
    ):
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
            client = TestClient(app)

            response = client.get("/v1/slots/capabilities")
            assert response.status_code == 200
            body = response.json()
            assert body["slot_api_version"] == "0.1.0"
            assert body["slot_save_path_configured"] is True
            assert body["slots"]["api_version"] == 2
            assert body["slots"]["supports_request_handle"] is True
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store

    def test_v2_phase1a_capability_bit_absent_when_slot_api_disabled(
        self, tmp_path, mock_engine_pool
    ):
        from omlx.server import app, _server_state

        original_pool = _server_state.engine_pool
        original_default = _server_state.default_model
        original_settings = _server_state.global_settings
        original_api_key = _server_state.api_key
        original_slot_store = getattr(_server_state, "slot_store", None)
        try:
            _server_state.engine_pool = mock_engine_pool
            _server_state.default_model = "test-model"
            _server_state.global_settings = GlobalSettings()
            _server_state.api_key = None
            _server_state.slot_store = None
            client = TestClient(app)

            response = client.get("/v1/slots/capabilities")
            assert response.status_code == 200
            body = response.json()
            assert body["slot_save_path_configured"] is False
            assert body["slots"]["api_version"] == 0
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store

    def test_v2_phase1a_v1_alias_still_works(
        self, tmp_path, mock_engine_pool, monkeypatch
    ):
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
                json={"filename": "legacy.kvslot", "model": "test-model"},
            )
            assert response.status_code == 200
            body = response.json()
            assert body["filename"] == "legacy.kvslot"
            assert body["request_handle"] == "legacy"
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
                self.preserve_thinking_default = False
                # Upstream (#1809) reads entry.model_context_length in
                # get_max_context_window(); None keeps this mock on the
                # server-default fallback these tests were written against.
                self.model_context_length = None

        entry = _Entry(model_dir)
        pool.get_entry = lambda model_id: entry

        settings = GlobalSettings()
        settings.slot_save_path = str(slot_dir)
        settings.scheduler.max_concurrent_requests = 1
        server_state.global_settings = settings
        server_state.engine_pool = pool
        server_state.default_model = "test-model"
        server_state.api_key = None
        server_state.one_shot_bind_table = OneShotBindTable()

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

    def test_v2_5a_restore_requires_explicit_handle_when_no_filename(
        self, tmp_path, mock_engine_pool
    ):
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
                json={"model": "test-model"},
            )
            assert response.status_code == 400
            detail = response.json()["detail"]["error"]
            assert detail["code"] == "request_handle_required"
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store

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
        original_bind_table = getattr(_server_state, "one_shot_bind_table", None)
        try:
            self._configure_slot_runtime(_server_state, slot_dir, mock_engine_pool, tmp_path)
            _server_state.one_shot_bind_table = OneShotBindTable()

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
                json={"filename": "slot-roundtrip.kvslot", "model": "test-model"},
            )
            assert save_response.status_code == 200, save_response.text
            n_saved = save_response.json()["n_saved"]

            restore_response = client.post(
                "/slots/0?action=restore",
                json={"filename": "slot-roundtrip.kvslot", "model": "test-model"},
            )
            assert restore_response.status_code == 200
            assert restore_response.json()["n_restored"] == n_saved

            saved_path = slot_dir / "slot-roundtrip.kvslot"
            safetensors_path = slot_dir / "slot-roundtrip.safetensors"
            safetensors_path.write_bytes(saved_path.read_bytes())
            loaded_cache, file_metadata = load_prompt_cache(
                str(safetensors_path),
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
            _server_state.one_shot_bind_table = original_bind_table

    def test_v2_phase4a_restore_emits_slot_restore_parked_event(
        self, tmp_path, mock_engine_pool, monkeypatch, caplog
    ):
        import omlx.server as server_module
        from omlx.server import _server_state, app

        slot_dir = tmp_path / "slots"
        slot_dir.mkdir()
        original_pool = _server_state.engine_pool
        original_default = _server_state.default_model
        original_settings = _server_state.global_settings
        original_api_key = _server_state.api_key
        original_slot_store = getattr(_server_state, "slot_store", None)
        original_bind_table = getattr(_server_state, "one_shot_bind_table", None)
        try:
            self._configure_slot_runtime(_server_state, slot_dir, mock_engine_pool, tmp_path)
            monkeypatch.setattr(
                server_module,
                "_serialize_slot_payload",
                lambda *args, **kwargs: (
                    b"x",
                    {
                        "n_tokens": 123,
                        "tensors": [{"name": "layer_0", "dtype": "f16", "shape": [1, 2]}],
                        "cache_class": "paged_ssd",
                    },
                ),
            )
            monkeypatch.setattr(server_module, "_apply_slot_restore_payload", lambda *args, **kwargs: 123)
            client = TestClient(app)
            save_response = client.post(
                "/slots/0?action=save",
                json={"filename": "slot-roundtrip.kvslot", "model": "test-model"},
            )
            assert save_response.status_code == 200, save_response.text
            with caplog.at_level("INFO"):
                response = client.post(
                    "/slots/0?action=restore",
                    json={"filename": "slot-roundtrip.kvslot", "model": "test-model"},
                )
            assert response.status_code == 200, response.text
            assert any("slot_restore_parked" in record.message for record in caplog.records)
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store
            _server_state.one_shot_bind_table = original_bind_table

    def test_v2_phase4a_save_emits_slot_save_completed_event(
        self, tmp_path, mock_engine_pool, monkeypatch, caplog
    ):
        import omlx.server as server_module
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
            monkeypatch.setattr(
                server_module,
                "_serialize_slot_payload",
                lambda *args, **kwargs: (
                    b"x",
                    {
                        "n_tokens": 123,
                        "tensors": [{"name": "layer_0", "dtype": "f16", "shape": [1, 2]}],
                        "cache_class": "paged_ssd",
                    },
                ),
            )
            client = TestClient(app)
            with caplog.at_level("INFO"):
                response = client.post(
                    "/slots/0?action=save",
                    json={"filename": "slot-roundtrip.kvslot", "model": "test-model"},
                )
            assert response.status_code == 200, response.text
            assert any("slot_save_completed" in record.message for record in caplog.records)
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store

    def test_v2_phase1a_restore_accepts_request_handle(
        self, tmp_path, mock_engine_pool, monkeypatch
    ):
        import omlx.server as server_module
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
            monkeypatch.setattr(server_module, "_serialize_slot_payload", lambda *args, **kwargs: (b"payload", {"n_tokens": 123, "tensors": [{"name": "layer_0", "dtype": "f16", "shape": [1, 2]}], "cache_class": "paged_ssd"}))
            monkeypatch.setattr(server_module, "_apply_slot_restore_payload", lambda *args, **kwargs: 123)
            client = TestClient(app)

            save_response = client.post(
                "/slots/0?action=save",
                json={"model": "test-model", "request_handle": "roundtrip"},
            )
            assert save_response.status_code == 200

            restore_response = client.post(
                "/slots/0?action=restore",
                json={"model": "test-model", "request_handle": "roundtrip"},
            )
            assert restore_response.status_code == 200
            body = restore_response.json()
            assert body["request_handle"] == "roundtrip"
            assert body["filename"] == "roundtrip.kvslot"
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store

    def test_v2_phase2_restore_returns_restore_epoch(
        self, tmp_path, mock_engine_pool, monkeypatch
    ):
        import omlx.server as server_module
        from omlx.server import _server_state, app

        slot_dir = tmp_path / "slots"
        slot_dir.mkdir()

        original_pool = _server_state.engine_pool
        original_default = _server_state.default_model
        original_settings = _server_state.global_settings
        original_api_key = _server_state.api_key
        original_slot_store = getattr(_server_state, "slot_store", None)
        original_bind_table = getattr(_server_state, "one_shot_bind_table", None)
        try:
            self._configure_slot_runtime(_server_state, slot_dir, mock_engine_pool, tmp_path)
            monkeypatch.setattr(server_module, "_serialize_slot_payload", lambda *args, **kwargs: (b"payload", {"n_tokens": 7, "tensors": [{"name": "layer_0", "dtype": "f16", "shape": [1, 2]}], "cache_class": "paged_ssd"}))
            monkeypatch.setattr(server_module, "_apply_slot_restore_payload", lambda *args, **kwargs: 7)
            client = TestClient(app)

            save_response = client.post(
                "/slots/0?action=save",
                json={"model": "test-model", "request_handle": "phase2-handle"},
            )
            assert save_response.status_code == 200

            restore_response = client.post(
                "/slots/0?action=restore",
                json={"model": "test-model", "request_handle": "phase2-handle"},
            )
            assert restore_response.status_code == 200
            body = restore_response.json()
            assert body["restore_epoch"]
            assert isinstance(body["restore_epoch"], str)
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store
            _server_state.one_shot_bind_table = original_bind_table

    def test_v2_phase2_full_round_trip_apply(
        self, tmp_path, mock_engine_pool, mock_llm_engine, monkeypatch
    ):
        import omlx.server as server_module
        from omlx.server import _server_state, app

        slot_dir = tmp_path / "slots"
        slot_dir.mkdir()

        original_pool = _server_state.engine_pool
        original_default = _server_state.default_model
        original_settings = _server_state.global_settings
        original_api_key = _server_state.api_key
        original_slot_store = getattr(_server_state, "slot_store", None)
        original_bind_table = getattr(_server_state, "one_shot_bind_table", None)
        try:
            self._configure_slot_runtime(_server_state, slot_dir, mock_engine_pool, tmp_path)
            monkeypatch.setattr(server_module, "_serialize_slot_payload", lambda *args, **kwargs: (b"payload", {"n_tokens": 6, "tensors": [{"name": "layer_0", "dtype": "f16", "shape": [1, 2]}], "cache_class": "paged_ssd"}))
            monkeypatch.setattr(server_module, "_apply_slot_restore_payload", lambda *args, **kwargs: 6)

            scheduler = Scheduler(model=MagicMock(), tokenizer=MockTokenizer())

            async def fake_chat(messages, **kwargs):
                del messages
                req = Request(
                    request_id="req-e2e",
                    prompt=[1, 2, 3, 4, 5, 6, 7],
                    sampling_params=SamplingParams(max_tokens=16),
                    x_omlx_request_handle=kwargs.get("x_omlx_request_handle"),
                    x_omlx_restore_epoch=kwargs.get("x_omlx_restore_epoch"),
                    x_omlx_model_id=kwargs.get("x_omlx_model_id"),
                )
                req.prompt_token_ids = [1, 2, 3, 4, 5, 6, 7]
                table = _server_state.one_shot_bind_table
                bind = await table.peek_any("test-model", req.x_omlx_request_handle)
                assert bind is not None
                scheduler._deserialize_one_shot_bind_payload = lambda _payload: ["cache-ok"]  # type: ignore[method-assign]
                scheduler._current_slot_restore_guards = lambda _model_id: (bind.manifest.model_fingerprint, bind.manifest.ctx_size)  # type: ignore[method-assign]
                applied = scheduler.try_apply_one_shot_bind(req)
                return MockGenerationOutput(
                    text="Chat response.",
                    prompt_tokens=7,
                    completion_tokens=2,
                    finish_reason="stop",
                    finished=True,
                    cached_tokens=req.cached_tokens if applied else 0,
                )

            mock_llm_engine.chat = AsyncMock(side_effect=fake_chat)

            client = TestClient(app)
            save_response = client.post(
                "/slots/0?action=save",
                json={"model": "test-model", "request_handle": "roundtrip"},
            )
            assert save_response.status_code == 200

            restore_response = client.post(
                "/slots/0?action=restore",
                json={"model": "test-model", "request_handle": "roundtrip"},
            )
            assert restore_response.status_code == 200
            restore_body = restore_response.json()
            assert restore_body["restore_epoch"]

            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "x_omlx_request_handle": "roundtrip",
                    "x_omlx_restore_epoch": restore_body["restore_epoch"],
                },
            )
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store
            _server_state.one_shot_bind_table = original_bind_table

    def test_v2_6a_real_impl_roundtrip_and_runtime_apply_error_envelope(
        self, tmp_path, mock_engine_pool, mock_llm_engine, caplog
    ):
        import asyncio
        import os
        import tempfile

        import mlx.core as mx
        from mlx_lm.models.cache import KVCache, save_prompt_cache
        from omlx.scheduler import SchedulerConfig
        from omlx.server import _server_state, app
        from omlx.slot_store import (
            OneShotBind,
            SlotManifest,
            compute_model_fingerprint,
            hash_prompt_token_prefix,
        )

        slot_dir = tmp_path / "slots"
        slot_dir.mkdir()
        paged_cache_dir = tmp_path / "paged-cache"
        paged_cache_dir.mkdir()
        model_dir = tmp_path / "model-artifacts"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"model":"test-model"}', encoding="utf-8")

        class _Model:
            layers = [object()]

        scheduler = Scheduler(
            model=_Model(),
            tokenizer=MockTokenizer(),
            config=SchedulerConfig(
                paged_ssd_cache_dir=str(paged_cache_dir),
                paged_cache_block_size=2,
                initial_cache_blocks=8,
                max_cache_blocks=32,
                model_name="test-model",
            ),
        )
        assert scheduler.block_aware_cache is not None

        prompt_tokens = [11, 12, 13, 14]
        cache = KVCache()
        keys = mx.full((1, 1, len(prompt_tokens), 2), 1, dtype=mx.float16)
        values = mx.full((1, 1, len(prompt_tokens), 2), 2, dtype=mx.float16)
        cache.update_and_fetch(keys, values)

        class _Entry:
            def __init__(self, path):
                self.model_path = str(path)
                self.engine = None
                self.preserve_thinking_default = False
                # Upstream (#1809) reads entry.model_context_length in
                # get_max_context_window(); None keeps this mock on the
                # server-default fallback these tests were written against.
                self.model_context_length = None

        entry = _Entry(model_dir)
        table = OneShotBindTable()

        original_pool = _server_state.engine_pool
        original_default = _server_state.default_model
        original_settings = _server_state.global_settings
        original_api_key = _server_state.api_key
        original_slot_store = getattr(_server_state, "slot_store", None)
        original_bind_table = getattr(_server_state, "one_shot_bind_table", None)
        try:
            self._configure_slot_runtime(_server_state, slot_dir, mock_engine_pool, tmp_path)
            mock_engine_pool.get_entry = lambda _model_id: entry
            mock_engine_pool.resolve_model_id = (
                lambda model_id_or_alias, settings_manager=None: model_id_or_alias
            )
            with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
                payload_path = tmp.name
            try:
                save_prompt_cache(
                    payload_path,
                    [cache],
                    metadata={
                        "model_id": "test-model",
                        "cached_tokens": str(len(prompt_tokens)),
                    },
                )
                with open(payload_path, "rb") as f:
                    payload_bytes = f.read()
            finally:
                try:
                    os.unlink(payload_path)
                except OSError:
                    pass
            metadata = {
                "n_tokens": len(prompt_tokens),
                "tensors": [{"name": "layer_0", "dtype": "float16", "shape": [1, 1, 4, 2]}],
                "cache_class": "KVCache",
                "prompt_prefix_sha256": hash_prompt_token_prefix(prompt_tokens),
            }
            manifest = SlotManifest(
                slot_format_version=2,
                model_fingerprint=compute_model_fingerprint(model_dir),
                model_id="test-model",
                ctx_size=_server_state.sampling.max_context_window,
                n_tokens=int(metadata["n_tokens"]),
                tensors=list(metadata["tensors"]),
                cache_class=str(metadata["cache_class"]),
                producer={"mlx_version": "0.0.0", "omlx_cache_format_version": "v1"},
                prompt_prefix_sha256=metadata.get("prompt_prefix_sha256"),
            )
            _server_state.one_shot_bind_table = table

            async def fake_chat(messages, **kwargs):
                del messages
                req = Request(
                    request_id="req-v26a",
                    prompt=prompt_tokens + [99],
                    sampling_params=SamplingParams(max_tokens=16),
                    x_omlx_request_handle=kwargs.get("x_omlx_request_handle"),
                    x_omlx_restore_epoch=kwargs.get("x_omlx_restore_epoch"),
                    x_omlx_model_id=kwargs.get("x_omlx_model_id"),
                )
                req.prompt_token_ids = list(prompt_tokens + [99])
                if req.x_omlx_restore_epoch == "epoch-bad":
                    scheduler._deserialize_one_shot_bind_payload = lambda _payload: (_ for _ in ()).throw(  # type: ignore[method-assign]
                        RuntimeError("runtime-apply-inner-detail")
                    )
                applied = scheduler.try_apply_one_shot_bind(req)
                return MockGenerationOutput(
                    text="Chat response.",
                    prompt_tokens=len(req.prompt_token_ids),
                    completion_tokens=2,
                    finish_reason="stop",
                    finished=True,
                    cached_tokens=req.cached_tokens if applied else 0,
                )

            mock_llm_engine.chat = AsyncMock(side_effect=fake_chat)
            client = TestClient(app)

            asyncio.run(
                table.put(
                    OneShotBind(
                        model_id="test-model",
                        request_handle="roundtrip",
                        payload_bytes=payload_bytes,
                        manifest=manifest,
                        restore_epoch="epoch-ok",
                    )
                )
            )
            ok_response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "x_omlx_request_handle": "roundtrip",
                    "x_omlx_restore_epoch": "epoch-ok",
                },
            )
            assert ok_response.status_code == 200, ok_response.text
            assert (
                ok_response.json()["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
            )

            asyncio.run(
                table.put(
                    OneShotBind(
                        model_id="test-model",
                        request_handle="roundtrip",
                        payload_bytes=payload_bytes[:32],
                        manifest=manifest,
                        restore_epoch="epoch-bad",
                    )
                )
            )
            with caplog.at_level("WARNING"):
                bad_response = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "hello"}],
                        "x_omlx_request_handle": "roundtrip",
                        "x_omlx_restore_epoch": "epoch-bad",
                    },
                )
            assert bad_response.status_code == 500, bad_response.text
            payload = bad_response.json()
            detail = None
            if isinstance(payload.get("detail"), dict):
                detail_payload = payload["detail"]
                if isinstance(detail_payload.get("error"), dict):
                    detail = detail_payload["error"]
            if detail is None and isinstance(payload.get("error"), dict):
                detail = payload["error"]
                nested = detail.get("message")
                if isinstance(nested, dict) and isinstance(nested.get("error"), dict):
                    detail = nested["error"]
            assert isinstance(detail, dict)
            assert detail["code"] == "slot_apply_runtime_error"
            assert detail["message"] == "slot apply failed"
            assert detail["details"]["message"] == "slot apply failed"
            assert detail["details"]["exception_type"] == "SlotApplyRuntimeError"
            assert detail["details"]["reason"] == "deserialize_failed"
            assert "runtime-apply-inner-detail" not in bad_response.text
            assert "runtime-apply-inner-detail" in caplog.text
            assert "correlation_id=roundtrip:epoch-bad" in caplog.text
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store
            _server_state.one_shot_bind_table = original_bind_table

    def test_v2_5b_full_round_trip_with_prefix_guard(
        self, tmp_path, mock_engine_pool, mock_llm_engine, monkeypatch
    ):
        import omlx.server as server_module
        from omlx.server import _server_state, app

        slot_dir = tmp_path / "slots"
        slot_dir.mkdir()

        original_pool = _server_state.engine_pool
        original_default = _server_state.default_model
        original_settings = _server_state.global_settings
        original_api_key = _server_state.api_key
        original_slot_store = getattr(_server_state, "slot_store", None)
        original_bind_table = getattr(_server_state, "one_shot_bind_table", None)
        try:
            self._configure_slot_runtime(_server_state, slot_dir, mock_engine_pool, tmp_path)
            matching_prompt = [1, 2, 3, 4, 5, 6]
            mismatched_prompt = [9, 9, 9, 4, 5, 6]
            entry, _prefix_cache = _build_prefix_cache_backed_entry(
                tmp_path / "model-artifacts",
                expected_tokens=matching_prompt,
                cached_tokens=3,
            )
            mock_engine_pool.get_entry = lambda model_id: entry
            monkeypatch.setattr(
                server_module,
                "_serialize_slot_payload",
                lambda entry_arg, model_arg, prompt_token_ids=None: (
                    b"payload",
                    {
                        "n_tokens": 3,
                        "tensors": [{"name": "layer_0", "dtype": "f16", "shape": [1, 2]}],
                        "cache_class": "paged_ssd",
                        "prompt_prefix_sha256": hash_prompt_token_prefix(
                            prompt_token_ids[:3]
                        ),
                    },
                ),
            )
            monkeypatch.setattr(server_module, "_apply_slot_restore_payload", lambda *args, **kwargs: 3)

            scheduler = Scheduler(model=MagicMock(), tokenizer=MagicMock())

            async def fake_chat(messages, **kwargs):
                prompt = matching_prompt
                if messages and "mismatch" in str(messages[0].get("content", "")):
                    prompt = mismatched_prompt
                req = Request(
                    request_id="req-v25b",
                    prompt=prompt,
                    sampling_params=SamplingParams(max_tokens=16),
                    x_omlx_request_handle=kwargs.get("x_omlx_request_handle"),
                    x_omlx_restore_epoch=kwargs.get("x_omlx_restore_epoch"),
                    x_omlx_model_id=kwargs.get("x_omlx_model_id"),
                )
                req.prompt_token_ids = list(prompt)
                bind = await _server_state.one_shot_bind_table.peek_any(
                    "test-model", "roundtrip"
                )
                assert bind is not None
                scheduler._deserialize_one_shot_bind_payload = lambda _payload: ["cache-ok"]  # type: ignore[method-assign]
                scheduler._current_slot_restore_guards = lambda _model_id: (bind.manifest.model_fingerprint, bind.manifest.ctx_size)  # type: ignore[method-assign]
                applied = scheduler.try_apply_one_shot_bind(req)
                return MockGenerationOutput(
                    text="Chat response.",
                    prompt_tokens=len(prompt),
                    completion_tokens=2,
                    finish_reason="stop",
                    finished=True,
                    cached_tokens=req.cached_tokens if applied else 0,
                )

            mock_llm_engine.chat = AsyncMock(side_effect=fake_chat)
            client = TestClient(app)

            save_response = client.post(
                "/slots/0?action=save",
                json={
                    "model": "test-model",
                    "request_handle": "roundtrip",
                    "prompt_tokens": matching_prompt,
                },
            )
            assert save_response.status_code == 200, save_response.text

            restore_response = client.post(
                "/slots/0?action=restore",
                json={"model": "test-model", "request_handle": "roundtrip"},
            )
            assert restore_response.status_code == 200
            restore_body = restore_response.json()

            matching_response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "x_omlx_request_handle": "roundtrip",
                    "x_omlx_restore_epoch": restore_body["restore_epoch"],
                },
            )
            assert matching_response.status_code == 200, matching_response.text
            assert (
                matching_response.json()["usage"]["prompt_tokens_details"]["cached_tokens"]
                > 0
            )

            restore_response_2 = client.post(
                "/slots/0?action=restore",
                json={"model": "test-model", "request_handle": "roundtrip"},
            )
            assert restore_response_2.status_code == 200
            restore_body_2 = restore_response_2.json()

            mismatch_response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "mismatch"}],
                    "x_omlx_request_handle": "roundtrip",
                    "x_omlx_restore_epoch": restore_body_2["restore_epoch"],
                },
            )
            assert mismatch_response.status_code == 409
            detail = mismatch_response.json()["error"]
            nested = detail.get("message") if isinstance(detail, dict) else None
            if isinstance(nested, dict) and isinstance(nested.get("error"), dict):
                detail = nested["error"]
            assert detail["code"] == "slot_apply_guard_mismatch"
            assert detail["details"]["field"] == "prompt_prefix"
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store
            _server_state.one_shot_bind_table = original_bind_table

    def test_v2_phase2_request_with_handle_but_wrong_epoch_returns_409(
        self, tmp_path, mock_engine_pool, mock_llm_engine, monkeypatch
    ):
        import omlx.server as server_module
        from omlx.server import _server_state, app

        slot_dir = tmp_path / "slots"
        slot_dir.mkdir()

        original_pool = _server_state.engine_pool
        original_default = _server_state.default_model
        original_settings = _server_state.global_settings
        original_api_key = _server_state.api_key
        original_slot_store = getattr(_server_state, "slot_store", None)
        original_bind_table = getattr(_server_state, "one_shot_bind_table", None)
        try:
            self._configure_slot_runtime(_server_state, slot_dir, mock_engine_pool, tmp_path)
            monkeypatch.setattr(server_module, "_serialize_slot_payload", lambda *args, **kwargs: (b"payload", {"n_tokens": 6, "tensors": [{"name": "layer_0", "dtype": "f16", "shape": [1, 2]}], "cache_class": "paged_ssd"}))
            monkeypatch.setattr(server_module, "_apply_slot_restore_payload", lambda *args, **kwargs: 6)

            scheduler = Scheduler(model=MagicMock(), tokenizer=MockTokenizer())

            async def fake_chat(messages, **kwargs):
                del messages
                req = Request(
                    request_id="req-wrong-epoch",
                    prompt=[1, 2, 3],
                    sampling_params=SamplingParams(max_tokens=16),
                    x_omlx_request_handle=kwargs.get("x_omlx_request_handle"),
                    x_omlx_restore_epoch=kwargs.get("x_omlx_restore_epoch"),
                    x_omlx_model_id=kwargs.get("x_omlx_model_id"),
                )
                req.prompt_token_ids = [1, 2, 3]
                scheduler._deserialize_one_shot_bind_payload = lambda _payload: ["cache-ok"]  # type: ignore[method-assign]
                bind = await _server_state.one_shot_bind_table.peek_any(
                    "test-model", "roundtrip"
                )
                assert bind is not None
                scheduler._current_slot_restore_guards = lambda _model_id: (bind.manifest.model_fingerprint, bind.manifest.ctx_size)  # type: ignore[method-assign]
                scheduler.try_apply_one_shot_bind(req)
                return MockGenerationOutput(text="unused")

            mock_llm_engine.chat = AsyncMock(side_effect=fake_chat)
            client = TestClient(app)

            assert client.post(
                "/slots/0?action=save",
                json={"model": "test-model", "request_handle": "roundtrip"},
            ).status_code == 200
            restore_response = client.post(
                "/slots/0?action=restore",
                json={"model": "test-model", "request_handle": "roundtrip"},
            )
            assert restore_response.status_code == 200

            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "x_omlx_request_handle": "roundtrip",
                    "x_omlx_restore_epoch": "wrong-epoch",
                },
            )
            assert response.status_code == 409
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store
            _server_state.one_shot_bind_table = original_bind_table

    def test_v2_phase2_request_with_handle_no_parked_entry_returns_409(
        self, tmp_path, mock_engine_pool, mock_llm_engine
    ):
        from omlx.server import _server_state, app

        slot_dir = tmp_path / "slots"
        slot_dir.mkdir()

        original_pool = _server_state.engine_pool
        original_default = _server_state.default_model
        original_settings = _server_state.global_settings
        original_api_key = _server_state.api_key
        original_slot_store = getattr(_server_state, "slot_store", None)
        original_bind_table = getattr(_server_state, "one_shot_bind_table", None)
        try:
            self._configure_slot_runtime(_server_state, slot_dir, mock_engine_pool, tmp_path)
            _server_state.one_shot_bind_table = OneShotBindTable()
            scheduler = Scheduler(model=MagicMock(), tokenizer=MockTokenizer())

            async def fake_chat(messages, **kwargs):
                del messages
                req = Request(
                    request_id="req-no-entry",
                    prompt=[1, 2, 3],
                    sampling_params=SamplingParams(max_tokens=16),
                    x_omlx_request_handle=kwargs.get("x_omlx_request_handle"),
                    x_omlx_restore_epoch=kwargs.get("x_omlx_restore_epoch"),
                    x_omlx_model_id=kwargs.get("x_omlx_model_id"),
                )
                req.prompt_token_ids = [1, 2, 3]
                scheduler.try_apply_one_shot_bind(req)
                return MockGenerationOutput(text="unused")

            mock_llm_engine.chat = AsyncMock(side_effect=fake_chat)
            client = TestClient(app)
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "x_omlx_request_handle": "missing",
                    "x_omlx_restore_epoch": "epoch-a",
                },
            )
            assert response.status_code == 409
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store
            _server_state.one_shot_bind_table = original_bind_table

    def test_v2_5a_chat_completion_with_bogus_handle_returns_409(
        self, tmp_path, mock_engine_pool
    ):
        from omlx.server import _server_state, app

        slot_dir = tmp_path / "slots"
        slot_dir.mkdir()

        original_pool = _server_state.engine_pool
        original_default = _server_state.default_model
        original_settings = _server_state.global_settings
        original_api_key = _server_state.api_key
        original_slot_store = getattr(_server_state, "slot_store", None)
        original_bind_table = getattr(_server_state, "one_shot_bind_table", None)
        try:
            self._configure_slot_runtime(_server_state, slot_dir, mock_engine_pool, tmp_path)
            _server_state.one_shot_bind_table = OneShotBindTable()
            client = TestClient(app)
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "x_omlx_request_handle": "bogus",
                    "x_omlx_restore_epoch": "epoch-a",
                },
            )
            assert response.status_code == 409
            detail = response.json()["error"]
            assert detail["code"] == "slot_handle_not_found"
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store
            _server_state.one_shot_bind_table = original_bind_table

    def test_v2_5a_chat_completion_with_wrong_epoch_returns_409(
        self, tmp_path, mock_engine_pool, monkeypatch
    ):
        import omlx.server as server_module
        from omlx.server import _server_state, app

        slot_dir = tmp_path / "slots"
        slot_dir.mkdir()

        original_pool = _server_state.engine_pool
        original_default = _server_state.default_model
        original_settings = _server_state.global_settings
        original_api_key = _server_state.api_key
        original_slot_store = getattr(_server_state, "slot_store", None)
        original_bind_table = getattr(_server_state, "one_shot_bind_table", None)
        try:
            self._configure_slot_runtime(_server_state, slot_dir, mock_engine_pool, tmp_path)
            monkeypatch.setattr(
                server_module,
                "_serialize_slot_payload",
                lambda *args, **kwargs: (
                    b"payload",
                    {
                        "n_tokens": 6,
                        "tensors": [{"name": "layer_0", "dtype": "f16", "shape": [1, 2]}],
                        "cache_class": "paged_ssd",
                    },
                ),
            )
            monkeypatch.setattr(server_module, "_apply_slot_restore_payload", lambda *args, **kwargs: 6)
            client = TestClient(app)

            assert client.post(
                "/slots/0?action=save",
                json={"model": "test-model", "request_handle": "roundtrip"},
            ).status_code == 200
            restore_response = client.post(
                "/slots/0?action=restore",
                json={"model": "test-model", "request_handle": "roundtrip"},
            )
            assert restore_response.status_code == 200

            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "x_omlx_request_handle": "roundtrip",
                    "x_omlx_restore_epoch": "wrong-epoch",
                },
            )
            assert response.status_code == 409
            detail = response.json()["error"]
            assert detail["code"] == "slot_apply_epoch_mismatch"
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store
            _server_state.one_shot_bind_table = original_bind_table

    def test_v2_5a_streaming_chat_completion_with_bogus_handle_returns_409_BEFORE_stream_starts(
        self, tmp_path, mock_engine_pool
    ):
        from omlx.server import _server_state, app

        slot_dir = tmp_path / "slots"
        slot_dir.mkdir()

        original_pool = _server_state.engine_pool
        original_default = _server_state.default_model
        original_settings = _server_state.global_settings
        original_api_key = _server_state.api_key
        original_slot_store = getattr(_server_state, "slot_store", None)
        original_bind_table = getattr(_server_state, "one_shot_bind_table", None)
        try:
            self._configure_slot_runtime(_server_state, slot_dir, mock_engine_pool, tmp_path)
            _server_state.one_shot_bind_table = OneShotBindTable()
            client = TestClient(app)
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hello"}],
                    "x_omlx_request_handle": "bogus",
                    "x_omlx_restore_epoch": "epoch-a",
                },
            )
            assert response.status_code == 409
            assert "data:" not in response.text
            detail = response.json()["error"]
            assert detail["code"] == "slot_handle_not_found"
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store
            _server_state.one_shot_bind_table = original_bind_table

    def test_v2_5a_chat_completion_without_slot_fields_unchanged(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert response.status_code == 200

    def test_v2_phase2_n_saved_equals_cached_tokens(
        self, tmp_path, mock_engine_pool, monkeypatch
    ):
        import mlx.core as mx
        import omlx.server as server_module
        from mlx_lm.models.cache import KVCache
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

            def fake_extract_slot_request_payload(_entry):
                cache = KVCache()
                keys = mx.full((1, 1, 3, 2), 1, dtype=mx.float16)
                values = mx.full((1, 1, 3, 2), 2, dtype=mx.float16)
                cache.update_and_fetch(keys, values)
                return [cache], 11, [99]

            monkeypatch.setattr(
                server_module,
                "_extract_slot_request_payload",
                fake_extract_slot_request_payload,
            )
            client = TestClient(app)
            save_response = client.post(
                "/slots/0?action=save",
                json={"model": "test-model", "request_handle": "nsaved"},
            )
            assert save_response.status_code == 200
            assert save_response.json()["n_saved"] == 11
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.global_settings = original_settings
            _server_state.api_key = original_api_key
            _server_state.slot_store = original_slot_store


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
        assert "final_ceiling" in pool_info
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
    def test_responses_uses_llm_lease(self, client, mock_engine_pool):
        response = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "Hello"},
        )

        assert response.status_code == 200
        assert mock_engine_pool.get_engine_calls[-1]["_lease"] is True
        assert mock_engine_pool.release_calls == ["test-model"]

    def test_response_endpoint_includes_reasoning_item_for_think_blocks(
        self, client, mock_llm_engine
    ):
        mock_llm_engine.chat = AsyncMock(
            return_value=MockGenerationOutput(
                text="<think>Need to reason.</think>Hello!",
                prompt_tokens=3,
                completion_tokens=6,
                finish_reason="stop",
                finished=True,
            )
        )

        response = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "Hello"},
        )

        assert response.status_code == 200
        data = response.json()
        assert [item["type"] for item in data["output"]] == ["reasoning", "message"]
        assert data["output"][0]["summary"][0]["text"] == "Need to reason."
        assert data["output"][1]["content"][0]["text"] == "Hello!"
        assert data["usage"]["output_tokens_details"]["reasoning_tokens"] == 3

    def test_response_endpoint_marks_length_as_incomplete(
        self, client, mock_llm_engine
    ):
        mock_llm_engine.chat = AsyncMock(
            return_value=MockGenerationOutput(
                text="Partial response",
                prompt_tokens=2,
                completion_tokens=3,
                finish_reason="length",
                finished=True,
            )
        )

        response = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "Hello",
                "max_output_tokens": 3,
                "store": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "incomplete"
        assert data["incomplete_details"] == {"reason": "max_output_tokens"}

    def test_responses_forwards_request_chat_template_kwargs(
        self, client, mock_llm_engine
    ):
        recorded_chat_kwargs = []

        async def chat(messages, **kwargs):
            recorded_chat_kwargs.append(kwargs)
            return MockGenerationOutput(
                text="pong",
                prompt_tokens=1,
                completion_tokens=1,
                finish_reason="stop",
            )

        mock_llm_engine.chat = chat

        response = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "Say pong",
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

        assert response.status_code == 200
        assert recorded_chat_kwargs
        ct_kwargs = recorded_chat_kwargs[0].get("chat_template_kwargs") or {}
        assert ct_kwargs["enable_thinking"] is False

    def test_response_stream_includes_reasoning_item_for_think_blocks(
        self, client, mock_llm_engine
    ):
        async def stream_chat(messages, **kwargs):
            yield MockGenerationOutput(
                text="<think>Need to reason.</think>Hello!",
                new_text="<think>Need to reason.</think>Hello!",
                prompt_tokens=3,
                completion_tokens=6,
                finish_reason="stop",
                finished=True,
            )

        mock_llm_engine.stream_chat = stream_chat

        response = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "Hello", "stream": True},
        )

        assert response.status_code == 200
        events = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        reasoning_deltas = [
            event["delta"]
            for event in events
            if event.get("type") == "response.reasoning_summary_text.delta"
        ]
        assert "".join(reasoning_deltas) == "Need to reason."

        added_items = [
            event
            for event in events
            if event.get("type") == "response.output_item.added"
        ]
        assert added_items[0]["item"]["type"] == "reasoning"
        assert added_items[0]["output_index"] == 0
        assert added_items[1]["item"]["type"] == "message"
        assert added_items[1]["output_index"] == 1

        completed = next(
            event for event in events if event.get("type") == "response.completed"
        )
        output = completed["response"]["output"]
        assert [item["type"] for item in output] == ["reasoning", "message"]
        assert output[0]["summary"][0]["text"] == "Need to reason."
        assert output[1]["content"][0]["text"] == "Hello!"
        usage = completed["response"]["usage"]
        assert usage["output_tokens_details"]["reasoning_tokens"] == 3

    def test_response_stream_emits_incomplete_event_on_length(
        self, client, mock_llm_engine
    ):
        async def stream_chat(messages, **kwargs):
            yield MockGenerationOutput(
                text="Partial response",
                new_text="Partial response",
                prompt_tokens=2,
                completion_tokens=3,
                finish_reason="length",
                finished=True,
            )

        mock_llm_engine.stream_chat = stream_chat

        response = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "Hello",
                "max_output_tokens": 3,
                "stream": True,
                "store": False,
            },
        )

        assert response.status_code == 200
        terminal_block = response.text.strip().split("\n\n")[-1]
        assert terminal_block.startswith("event: response.incomplete\n")
        data_line = next(
            line for line in terminal_block.splitlines() if line.startswith("data: ")
        )
        event = json.loads(data_line.removeprefix("data: "))
        assert event["type"] == "response.incomplete"
        assert event["response"]["status"] == "incomplete"
        assert event["response"]["incomplete_details"] == {
            "reason": "max_output_tokens"
        }

    def test_response_stream_summary_log_names_model(self, client, caplog):
        with caplog.at_level("INFO", logger="omlx.server"):
            response = client.post(
                "/v1/responses",
                json={"model": "test-model", "input": "Hello", "stream": True},
            )

        assert response.status_code == 200
        summaries = [
            record.message
            for record in caplog.records
            if "Responses API: " in record.message
        ]
        assert summaries, "no Responses API summary was logged"
        assert "model=test-model" in summaries[-1]

    def test_response_endpoint_recovers_tool_call_from_thinking(self, tmp_path):
        from omlx.server import app, _server_state

        state_dir = tmp_path / "response-state"
        engine = RecordingResponsesEngine(
            outputs=[
                MockGenerationOutput(
                    text=(
                        "<think>Need to inspect first."
                        '<tool_call>{"name":"exec_command","arguments":{"cmd":"ls"}}</tool_call>'
                        "Then continue.</think>"
                    ),
                    finish_reason="stop",
                ),
            ]
        )
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
                    "tools": [
                        {
                            "type": "function",
                            "name": "exec_command",
                            "description": "Run a shell command",
                            "parameters": {
                                "type": "object",
                                "properties": {"cmd": {"type": "string"}},
                                "required": ["cmd"],
                            },
                        }
                    ],
                },
            )
            assert response.status_code == 200

            output_items = response.json()["output"]
            message_items = [item for item in output_items if item["type"] == "message"]
            reasoning_items = [
                item for item in output_items if item["type"] == "reasoning"
            ]
            function_items = [
                item for item in output_items if item["type"] == "function_call"
            ]

            assert len(reasoning_items) == 1
            assert reasoning_items[0]["summary"][0]["text"] == (
                "Need to inspect first.Then continue."
            )
            assert "<tool_call>" not in reasoning_items[0]["summary"][0]["text"]
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
        engine = RecordingResponsesEngine(
            outputs=[
                MockGenerationOutput(
                    text="",
                    finish_reason="tool_calls",
                    tool_calls=[
                        {
                            "id": "call_123",
                            "name": "exec_command",
                            "arguments": '{"cmd":"ls"}',
                        }
                    ],
                ),
                MockGenerationOutput(text="Done.", finish_reason="stop"),
            ]
        )
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

    def test_models_status_includes_model_alias(self, client):
        """Model aliases should be available to clients that join status metadata."""
        from omlx.server import _server_state

        class Settings:
            model_alias = "gpt-4o"
            max_context_window = 32768
            max_tokens = 8192
            is_favorite = False
            is_hidden = False

        class SettingsManager:
            def get_settings(self, model_id):
                return Settings()

            def get_settings_for_request(self, model_id, resolved_model_id=None):
                return Settings()

        original_settings_manager = _server_state.settings_manager
        try:
            _server_state.settings_manager = SettingsManager()
            response = client.get("/v1/models/status")
        finally:
            _server_state.settings_manager = original_settings_manager

        assert response.status_code == 200
        model = response.json()["models"][0]
        assert model["id"] == "test-model"
        assert model["model_alias"] == "gpt-4o"
        assert model["max_context_window"] == 32768
        assert model["max_tokens"] == 8192


class TestCompletionEndpoint:
    """Tests for the /v1/completions endpoint."""

    def test_completion_uses_llm_lease(self, client, mock_engine_pool):
        """LLM completion keeps a pool lease until the response body finishes."""
        response = client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "Hello, world!",
            },
        )

        assert response.status_code == 200
        assert mock_engine_pool.get_engine_calls[-1]["_lease"] is True
        assert mock_engine_pool.release_calls == ["test-model"]

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

    def test_completion_includes_cached_tokens_on_cache_hit(
        self, client, mock_llm_engine
    ):
        """Non-streaming completion responses should expose cached token counts."""
        mock_llm_engine.generate = AsyncMock(
            return_value=MockGenerationOutput(
                text="Generated response.",
                prompt_tokens=2215,
                completion_tokens=5,
                cached_tokens=2048,
            )
        )

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

    def test_completion_forwards_thinking_budget(self, client, mock_llm_engine):
        """Non-streaming /v1/completions must forward thinking_budget to the
        engine. Regression for #1825: the field was absent from
        CompletionRequest, so Pydantic dropped it and it never reached
        generate()."""
        mock_llm_engine.generate = AsyncMock(
            return_value=MockGenerationOutput(text="Generated response.")
        )

        response = client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "Explain why the sky is blue.",
                "thinking_budget": 300,
            },
        )

        assert response.status_code == 200
        assert mock_llm_engine.generate.call_args.kwargs["thinking_budget"] == 300

    def test_completion_streaming_forwards_thinking_budget(
        self, client, mock_llm_engine
    ):
        """Streaming /v1/completions must forward thinking_budget to the
        engine (companion to the non-streaming path; see #1825)."""
        captured = {}

        async def recording_stream_generate(prompt, **kwargs):
            captured.update(kwargs)
            yield MockGenerationOutput(text="Hi", new_text="Hi", finished=False)
            yield MockGenerationOutput(
                text="Hi there",
                new_text=" there",
                finished=True,
                finish_reason="stop",
            )

        mock_llm_engine.stream_generate = recording_stream_generate

        response = client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "Explain why the sky is blue.",
                "stream": True,
                "thinking_budget": 300,
            },
        )

        assert response.status_code == 200
        assert captured.get("thinking_budget") == 300

    def test_completion_thinking_budget_from_model_settings(
        self, client, mock_llm_engine
    ):
        """A model-level thinking budget (admin settings) applies to
        /v1/completions even when the request omits the parameter, matching
        /v1/chat/completions."""
        from omlx.model_settings import ModelSettings
        from omlx.server import _server_state

        class StubSettingsManager:
            def get_settings(self, model_id):
                return ModelSettings(
                    thinking_budget_enabled=True,
                    thinking_budget_tokens=256,
                )

        mock_llm_engine.generate = AsyncMock(
            return_value=MockGenerationOutput(text="Generated response.")
        )

        original_settings_manager = _server_state.settings_manager
        _server_state.settings_manager = StubSettingsManager()
        try:
            response = client.post(
                "/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": "Explain why the sky is blue.",
                },
            )
        finally:
            _server_state.settings_manager = original_settings_manager

        assert response.status_code == 200
        assert mock_llm_engine.generate.call_args.kwargs["thinking_budget"] == 256


class TestChatCompletionEndpoint:
    """Tests for the /v1/chat/completions endpoint."""

    def test_chat_completion_uses_llm_lease(self, client, mock_engine_pool):
        """Chat completion keeps a pool lease until the response body finishes."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 200
        assert mock_engine_pool.get_engine_calls[-1]["_lease"] is True
        assert mock_engine_pool.release_calls == ["test-model"]

    def test_chat_completion_summary_log_names_the_model(self, client, caplog):
        """The per-request summary must identify the serving model.

        With several models loaded, interleaved summaries are otherwise
        unattributable.
        """
        with caplog.at_level("INFO", logger="omlx.server"):
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

        assert response.status_code == 200
        summaries = [
            r.message for r in caplog.records if "Chat completion: " in r.message
        ]
        assert summaries, "no chat completion summary was logged"
        assert "model=test-model" in summaries[-1]

    def test_chat_completion_summary_uses_fallback_model(
        self,
        client,
        caplog,
        mock_engine_pool,
        monkeypatch,
    ):
        from omlx.exceptions import ModelNotFoundError
        from omlx.server import _server_state
        from omlx.settings import GlobalSettings

        settings = GlobalSettings()
        settings.model.model_fallback = True
        monkeypatch.setattr(_server_state, "global_settings", settings)

        original_get_engine = mock_engine_pool.get_engine

        async def get_engine(model_id, _lease=False, runtime_settings=None):
            if model_id == "missing-model":
                raise ModelNotFoundError(model_id, ["test-model"])
            return await original_get_engine(
                model_id,
                _lease=_lease,
                runtime_settings=runtime_settings,
            )

        monkeypatch.setattr(mock_engine_pool, "get_engine", get_engine)

        with caplog.at_level("INFO", logger="omlx.server"):
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "missing-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

        assert response.status_code == 200
        summaries = [
            record.message
            for record in caplog.records
            if "Chat completion: " in record.message
        ]
        assert summaries, "no chat completion summary was logged"
        assert "model=test-model" in summaries[-1]
        assert "model=missing-model" not in summaries[-1]
        assert mock_engine_pool.release_calls == ["test-model"]

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

    def test_chat_completion_includes_cached_tokens_on_cache_hit(
        self, client, mock_llm_engine
    ):
        """Non-streaming chat responses should expose cached token counts."""
        mock_llm_engine.chat = AsyncMock(
            return_value=MockGenerationOutput(
                text="Chat response.",
                prompt_tokens=2215,
                completion_tokens=5,
                cached_tokens=2048,
                finish_reason="stop",
                finished=True,
            )
        )

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

    def test_chat_completion_sanitizes_reasoning_tool_call_markup(
        self, client, mock_llm_engine
    ):
        """Thinking-only tool calls should become structured tool_calls without leaked markup."""
        mock_llm_engine.chat = AsyncMock(
            return_value=MockGenerationOutput(
                text=(
                    "<think>Need to inspect first."
                    '<tool_call>{"name":"get_weather","arguments":{"city":"SF"}}</tool_call>'
                    "Then continue.</think>"
                ),
                prompt_tokens=10,
                completion_tokens=5,
                finish_reason="stop",
                finished=True,
            )
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "tools": [
                    {
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
                    }
                ],
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

    @pytest.mark.parametrize(
        ("tool_choice", "request_tools", "expect_tools"),
        [
            (None, None, True),
            ("auto", None, True),
            ("none", None, False),
            (
                "none",
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "user_search",
                            "description": "Search from request tools",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                },
                            },
                        },
                    }
                ],
                False,
            ),
        ],
    )
    def test_chat_completion_tool_choice_controls_mcp_tools(
        self,
        client,
        mock_llm_engine,
        tool_choice,
        request_tools,
        expect_tools,
    ):
        """tool_choice='none' should suppress request and globally configured tools."""
        from omlx.server import _server_state

        class RecordingMCPManager:
            def __init__(self):
                self.calls = []

            def get_merged_tools(self, user_tools=None):
                self.calls.append(user_tools)
                return [
                    {
                        "type": "function",
                        "function": {
                            "name": "mcp_search",
                            "description": "Search via MCP",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                },
                            },
                        },
                    }
                ]

        recorded_count_tools = []
        recorded_chat_kwargs = []

        def count_chat_tokens(messages, tools=None, **kwargs):
            recorded_count_tools.append(tools)
            return 1

        async def chat(messages, **kwargs):
            recorded_chat_kwargs.append(kwargs)
            return MockGenerationOutput(
                text="Plain response.",
                prompt_tokens=1,
                completion_tokens=1,
                finish_reason="stop",
            )

        original_mcp_manager = _server_state.mcp_manager
        manager = RecordingMCPManager()
        mock_llm_engine.count_chat_tokens = count_chat_tokens
        mock_llm_engine.chat = chat

        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if request_tools is not None:
            payload["tools"] = request_tools

        try:
            _server_state.mcp_manager = manager
            response = client.post("/v1/chat/completions", json=payload)
        finally:
            _server_state.mcp_manager = original_mcp_manager

        assert response.status_code == 200
        assert recorded_chat_kwargs

        if expect_tools:
            assert manager.calls == [request_tools]
            assert recorded_count_tools[0] is not None
            assert "tools" in recorded_chat_kwargs[0]
        else:
            assert manager.calls == []
            assert recorded_count_tools == [None]
            assert "tools" not in recorded_chat_kwargs[0]


class TestAnthropicMessagesEndpoint:
    """Tests for the /v1/messages endpoint (Anthropic format)."""

    def test_anthropic_messages_uses_llm_lease(self, client, mock_engine_pool):
        response = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 200
        assert mock_engine_pool.get_engine_calls[-1]["_lease"] is True
        assert mock_engine_pool.release_calls == ["test-model"]

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

    def test_anthropic_stream_summary_log_names_model(self, client, caplog):
        with caplog.at_level("INFO", logger="omlx.server"):
            response = client.post(
                "/v1/messages",
                json={
                    "model": "test-model",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": True,
                },
            )

        assert response.status_code == 200
        summaries = [
            record.message
            for record in caplog.records
            if "Anthropic message: " in record.message
        ]
        assert summaries, "no Anthropic message summary was logged"
        assert "model=test-model" in summaries[-1]

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

    def test_anthropic_messages_sanitize_thinking_tool_call_markup(
        self, client, mock_llm_engine
    ):
        """Anthropic thinking blocks should not expose raw tool-call markup."""
        mock_llm_engine.chat = AsyncMock(
            return_value=MockGenerationOutput(
                text=(
                    "<think>Need to inspect first."
                    '<tool_call>{"name":"get_weather","arguments":{"city":"SF"}}</tool_call>'
                    "Then continue.</think>"
                ),
                prompt_tokens=10,
                completion_tokens=5,
                finish_reason="stop",
                finished=True,
            )
        )

        response = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hi"}],
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get weather",
                        "input_schema": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    }
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()
        thinking_blocks = [
            block for block in data["content"] if block["type"] == "thinking"
        ]
        tool_use_blocks = [
            block for block in data["content"] if block["type"] == "tool_use"
        ]

        assert len(thinking_blocks) == 1
        assert thinking_blocks[0]["thinking"] == "Need to inspect first.Then continue."
        assert "<tool_call>" not in thinking_blocks[0]["thinking"]
        assert len(tool_use_blocks) == 1
        assert tool_use_blocks[0]["name"] == "get_weather"
        assert tool_use_blocks[0]["input"] == {"city": "SF"}
        assert data["stop_reason"] == "tool_use"

    def test_anthropic_messages_with_exposed_profile_model(
        self, client, mock_llm_engine, mock_engine_pool, tmp_path
    ):
        """A model:profile id resolves and overlays profile settings on /v1/messages."""
        from omlx.model_settings import ModelSettings, ModelSettingsManager
        from omlx.server import _server_state

        manager = ModelSettingsManager(tmp_path)
        manager.set_settings("test-model", ModelSettings(temperature=0.1))
        manager.save_profile(
            model_id="test-model",
            name="thinking",
            display_name="Thinking",
            description=None,
            settings={"temperature": 0.9, "enable_thinking": True},
            expose_as_model=True,
        )

        def resolve_model_id(model_id, settings_manager=None):
            if settings_manager is not None:
                source = settings_manager.get_exposed_profile_source_model_id(model_id)
                if source:
                    return source
            return model_id

        recorded_chat_kwargs = []

        async def chat(messages, **kwargs):
            recorded_chat_kwargs.append(kwargs)
            return MockGenerationOutput(
                text="Hello from the profile.",
                prompt_tokens=1,
                completion_tokens=1,
                finish_reason="stop",
            )

        mock_llm_engine.chat = chat
        original_resolve = mock_engine_pool.resolve_model_id
        original_settings_manager = _server_state.settings_manager
        mock_engine_pool.resolve_model_id = resolve_model_id
        try:
            _server_state.settings_manager = manager
            response = client.post(
                "/v1/messages",
                json={
                    "model": "test-model:thinking",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
        finally:
            _server_state.settings_manager = original_settings_manager
            mock_engine_pool.resolve_model_id = original_resolve

        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"

        # The profile's settings — not the base model's — reached the engine.
        assert recorded_chat_kwargs
        assert recorded_chat_kwargs[0]["temperature"] == 0.9
        ct_kwargs = recorded_chat_kwargs[0].get("chat_template_kwargs") or {}
        assert ct_kwargs.get("enable_thinking") is True


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

    def test_embeddings_use_discovered_context_length(self, client, mock_engine_pool):
        """Embedding requests should not fall back to mlx-embeddings' 512 default."""
        mock_engine_pool._models.append(
            {"id": "test-embed-model", "loaded": True, "pinned": False, "size": 500000}
        )
        mock_engine_pool._entries["test-embed-model"] = SimpleNamespace(
            model_context_length=40960
        )

        response = client.post(
            "/v1/embeddings",
            json={
                "model": "test-embed-model",
                "input": "hello",
            },
        )

        assert response.status_code == 200
        kwargs = mock_engine_pool._embedding_engine.calls[-1]["kwargs"]
        assert kwargs["max_length"] == 40960
        assert kwargs["truncation"] is True

    def test_embeddings_request_max_length_overrides_default(
        self, client, mock_engine_pool
    ):
        """Explicit max_length should be forwarded to the embedding engine."""
        mock_engine_pool._models.append(
            {"id": "test-embed-model", "loaded": True, "pinned": False, "size": 500000}
        )
        mock_engine_pool._entries["test-embed-model"] = SimpleNamespace(
            model_context_length=40960
        )

        response = client.post(
            "/v1/embeddings",
            json={
                "model": "test-embed-model",
                "input": "hello",
                "max_length": 1024,
                "truncation": False,
            },
        )

        assert response.status_code == 200
        kwargs = mock_engine_pool._embedding_engine.calls[-1]["kwargs"]
        assert kwargs["max_length"] == 1024
        assert kwargs["truncation"] is False

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
        image_data_uri = (
            "data:image/png;base64,"
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/"
            "x8AAwMCAO+/p9sAAAAASUVORK5CYII="
        )

        response = client.post(
            "/v1/embeddings",
            json={
                "model": "test-embed-model",
                "items": [
                    {"text": "hello"},
                    {"image": image_data_uri},
                    {
                        "text": "hello",
                        "image": image_data_uri,
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

    def test_token_count_uses_llm_lease(self, client, mock_engine_pool):
        response = client.post(
            "/v1/messages/count_tokens",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 200
        assert mock_engine_pool.get_engine_calls[-1]["_lease"] is True
        assert mock_engine_pool.release_calls == ["test-model"]

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


class TestTokenizeEndpoint:
    """Tests for the /v1/tokenize endpoint."""

    def test_tokenize_messages_applies_chat_template(self, client):
        response = client.post(
            "/v1/tokenize",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello world"}],
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["model"] == "test-model"
        assert body["applied_chat_template"] is True
        assert isinstance(body["token_ids"], list)
        assert body["n_tokens"] == len(body["token_ids"])
        assert body["n_tokens"] > 0

    def test_tokenize_prompt_raw(self, client):
        response = client.post(
            "/v1/tokenize",
            json={
                "model": "test-model",
                "prompt": "raw prompt tokens here",
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["model"] == "test-model"
        assert body["applied_chat_template"] is False
        assert body["n_tokens"] == len(body["token_ids"])
        assert body["n_tokens"] > 0

    def test_tokenize_rejects_both_messages_and_prompt(self, client):
        response = client.post(
            "/v1/tokenize",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "prompt": "raw prompt",
            },
        )

        assert response.status_code == 400
        body = response.json()
        assert (
            body["error"]["message"]["error"]["code"] == "invalid_tokenize_payload"
        )


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


class _RecordingMCPManager:
    """MCP manager stand-in that records merge requests."""

    def __init__(self):
        self.calls = []

    def get_merged_tools(self, user_tools=None):
        self.calls.append(user_tools)
        mcp_tools = [
            {
                "type": "function",
                "function": {
                    "name": "mcp_search",
                    "description": "Search via MCP",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                        },
                    },
                },
            }
        ]
        return mcp_tools + (user_tools or [])

    def get_all_tools_openai(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "mcp_search",
                    "description": "Search via MCP",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                        },
                    },
                },
            }
        ]


class TestMCPExposeToolsToggle:
    """Settings > Global Settings > MCP: "Expose backend MCP tools to clients".

    When the toggle is OFF, backend MCP tools must not be merged into client
    requests, while tools the client itself sends still pass through.
    """

    USER_SEARCH_TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "user_search",
                "description": "Search from request tools",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                },
            },
        }
    ]

    USER_RESPONSE_TOOLS = [
        {
            "type": "function",
            "name": "user_search",
            "description": "Search from request tools",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
            },
        }
    ]

    def _install(self, expose_tools, manager):
        """Install manager + toggle into server state; returns restore fn."""
        from omlx.server import _server_state
        from omlx.settings import GlobalSettings, MCPSettings

        original_manager = _server_state.mcp_manager
        original_settings = _server_state.global_settings
        _server_state.mcp_manager = manager
        _server_state.global_settings = GlobalSettings(
            mcp=MCPSettings(
                config_path="/mcp.json",
                expose_tools=expose_tools,
            )
        )
        return lambda: (
            setattr(_server_state, "mcp_manager", original_manager),
            setattr(_server_state, "global_settings", original_settings),
        )

    def _tool_names(self, recorded_chat_kwargs):
        tools = recorded_chat_kwargs[0].get("tools") or []
        return [t.get("function", {}).get("name") for t in tools]

    @pytest.mark.parametrize(
        ("expose_tools", "request_tools", "expect_mcp_merge"),
        [
            (True, None, True),
            (False, None, False),
            (False, "user_tools", False),
        ],
    )
    def test_chat_completion_expose_tools_controls_mcp_merge(
        self,
        client,
        mock_llm_engine,
        expose_tools,
        request_tools,
        expect_mcp_merge,
    ):
        """OpenAI-compatible /v1/chat/completions honours the toggle."""
        recorded_chat_kwargs = []

        async def chat(messages, **kwargs):
            recorded_chat_kwargs.append(kwargs)
            return MockGenerationOutput(
                text="Plain response.",
                prompt_tokens=1,
                completion_tokens=1,
                finish_reason="stop",
            )

        manager = _RecordingMCPManager()
        restore = self._install(expose_tools, manager)
        mock_llm_engine.chat = chat

        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        if request_tools == "user_tools":
            payload["tools"] = self.USER_SEARCH_TOOLS

        try:
            response = client.post("/v1/chat/completions", json=payload)
        finally:
            restore()

        assert response.status_code == 200
        assert recorded_chat_kwargs
        names = self._tool_names(recorded_chat_kwargs)

        if expect_mcp_merge:
            assert manager.calls == [payload.get("tools")]
            assert "mcp_search" in names
        else:
            assert manager.calls == []
            assert "mcp_search" not in names
            if request_tools == "user_tools":
                assert "user_search" in names
            else:
                assert "tools" not in recorded_chat_kwargs[0]

    @pytest.mark.parametrize(
        ("expose_tools", "expect_mcp_merge"),
        [
            (True, True),
            (False, False),
        ],
    )
    def test_anthropic_messages_expose_tools_controls_mcp_merge(
        self,
        client,
        mock_llm_engine,
        expose_tools,
        expect_mcp_merge,
    ):
        """Anthropic-compatible /v1/messages honours the toggle."""
        recorded_chat_kwargs = []

        async def chat(messages, **kwargs):
            recorded_chat_kwargs.append(kwargs)
            return MockGenerationOutput(
                text="Plain response.",
                prompt_tokens=1,
                completion_tokens=1,
                finish_reason="stop",
            )

        manager = _RecordingMCPManager()
        restore = self._install(expose_tools, manager)
        mock_llm_engine.chat = chat

        payload = {
            "model": "test-model",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            ],
        }

        try:
            response = client.post("/v1/messages", json=payload)
        finally:
            restore()

        assert response.status_code == 200
        assert recorded_chat_kwargs
        names = self._tool_names(recorded_chat_kwargs)

        if expect_mcp_merge:
            assert "mcp_search" in names
            assert "get_weather" in names
        else:
            assert "mcp_search" not in names
            assert "get_weather" in names

    @pytest.mark.parametrize(
        ("expose_tools", "expect_mcp_merge"),
        [
            (True, True),
            (False, False),
        ],
    )
    def test_responses_expose_tools_controls_mcp_merge(
        self,
        client,
        mock_llm_engine,
        expose_tools,
        expect_mcp_merge,
    ):
        """OpenAI-compatible /v1/responses honours the toggle."""
        recorded_chat_kwargs = []

        async def chat(messages, **kwargs):
            recorded_chat_kwargs.append(kwargs)
            return MockGenerationOutput(
                text="Plain response.",
                prompt_tokens=1,
                completion_tokens=1,
                finish_reason="stop",
            )

        manager = _RecordingMCPManager()
        restore = self._install(expose_tools, manager)
        mock_llm_engine.chat = chat

        payload = {
            "model": "test-model",
            "input": "Hello",
            "tools": self.USER_RESPONSE_TOOLS,
            "store": False,
        }

        try:
            response = client.post("/v1/responses", json=payload)
        finally:
            restore()

        assert response.status_code == 200
        assert recorded_chat_kwargs
        names = self._tool_names(recorded_chat_kwargs)
        assert "user_search" in names

        if expect_mcp_merge:
            assert manager.calls
            assert "mcp_search" in names
        else:
            assert manager.calls == []
            assert "mcp_search" not in names


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

        mock_llm_engine.chat = AsyncMock(
            return_value=MockGenerationOutput(
                text='```json\n{"name": "test", "age": 25}\n```',
                prompt_tokens=10,
                completion_tokens=8,
                finish_reason="stop",
                finished=True,
            )
        )

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

        mock_llm_engine.chat = AsyncMock(
            return_value=MockGenerationOutput(
                text='{"key": "value"}',
                prompt_tokens=10,
                completion_tokens=5,
                finish_reason="stop",
                finished=True,
            )
        )

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

        mock_llm_engine.chat = AsyncMock(
            return_value=MockGenerationOutput(
                text='```json\n{"city": "Seoul", "temp": 15}\n```',
                prompt_tokens=10,
                completion_tokens=8,
                finish_reason="stop",
                finished=True,
            )
        )

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
        mock_llm_engine.chat = AsyncMock(
            return_value=MockGenerationOutput(
                text="Hello, how can I help?",
                prompt_tokens=10,
                completion_tokens=5,
                finish_reason="stop",
                finished=True,
            )
        )

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
