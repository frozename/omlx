# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests that the prefill memory guard is wired up.

Until 2026-05-15 the guard was dead code: ``Scheduler.memory_monitor`` was
left as ``None`` and ``_set_model_info_for_monitor`` had zero callers, so
``_preflight_memory_check`` short-circuited at the ``memory_monitor is None``
gate even when ``_prefill_memory_guard`` was flipped on by the enforcer.

These tests pin the wiring so a future refactor cannot silently revert it.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

from omlx.exceptions import PrefillMemoryExceededError
from omlx.memory_monitor import MemoryMonitor
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler, SchedulerConfig


class _ModelConfig:
    """Minimal config object exposing the fields the estimator reads."""

    def __init__(
        self,
        num_hidden_layers: int | None = 32,
        num_key_value_heads: int = 8,
        num_attention_heads: int = 32,
        head_dim: int = 192,  # > 128 → high-head-dim tiled SDPA scratch
    ) -> None:
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim


def _make_scheduler() -> Scheduler:
    model = MagicMock()
    model.layers = []
    model.config = _ModelConfig()
    # Strip make_cache so the KVCache-counting branch in
    # _set_model_info_for_monitor doesn't try to iterate a MagicMock.
    del model.make_cache

    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2

    config = SchedulerConfig(
        max_num_seqs=8,
        prefill_step_size=2048,
        paged_cache_block_size=0,
    )
    return Scheduler(model=model, tokenizer=tokenizer, config=config)


def _make_request(prompt_tokens: int = 65536) -> Request:
    req = Request(
        request_id="req-large",
        prompt=list(range(prompt_tokens)),
        sampling_params=SamplingParams(max_tokens=8),
    )
    req.prompt_token_ids = list(range(prompt_tokens))
    req.num_prompt_tokens = prompt_tokens
    return req


def test_scheduler_init_instantiates_memory_monitor():
    scheduler = _make_scheduler()
    assert isinstance(scheduler.memory_monitor, MemoryMonitor)


def test_scheduler_init_populates_estimator_dims():
    scheduler = _make_scheduler()
    monitor = scheduler.memory_monitor
    assert monitor is not None
    assert monitor._num_attention_heads == 32
    assert monitor._head_dim == 192
    assert monitor._num_layers == 32
    assert monitor._num_kv_heads == 8


def test_estimator_produces_nonzero_peak_after_init():
    scheduler = _make_scheduler()
    assert scheduler.memory_monitor is not None
    peak = scheduler.memory_monitor.estimate_prefill_peak_bytes(65536, 2048)
    assert peak > 0


def test_preflight_positive_control_passes_normal_request():
    """Positive-control: a normal prompt under a generous limit must NOT
    be rejected. Defends against an accidental sign-flip on the
    threshold comparison in _preflight_memory_check.
    """
    scheduler = _make_scheduler()
    scheduler._prefill_memory_guard = True
    # Huge limit — even a multi-GB peak fits comfortably.
    scheduler._memory_hard_limit_bytes = 10**18
    with (
        patch("omlx.scheduler.mx.get_active_memory", return_value=0),
        patch("omlx.scheduler.get_phys_footprint", return_value=0),
    ):
        assert scheduler._preflight_memory_check(_make_request(32768)) is None


def test_preflight_rejects_when_estimated_peak_exceeds_hard_limit():
    scheduler = _make_scheduler()
    scheduler._prefill_memory_guard = True
    scheduler._memory_hard_limit_bytes = 1  # any allocation exceeds

    with (
        patch("omlx.scheduler.mx.get_active_memory", return_value=0),
        patch("omlx.scheduler.get_phys_footprint", return_value=0),
    ):
        rejection = scheduler._preflight_memory_check(_make_request(65536))

    assert rejection is not None
    assert "Prefill would require" in rejection.message
    assert "KV+SDPA" in rejection.message
    assert rejection.estimated_bytes > 0
    assert rejection.limit_bytes == 1


def test_route_preflight_requests_eviction_before_safety_cap_rejection(monkeypatch):
    scheduler = _make_scheduler()
    scheduler._prefill_memory_guard = True
    scheduler._memory_hard_limit_bytes = 1_000
    scheduler._memory_abort_limit_bytes = 100
    scheduler._prefill_min_chunk_tokens = 4
    scheduler.memory_monitor.estimate_prefill_peak_bytes = MagicMock(return_value=10)
    scheduler.memory_monitor.estimate_prompt_kv_bytes = MagicMock(return_value=20)
    scheduler._predicted_chunk_transient = MagicMock(return_value=30)

    import omlx.scheduler as scheduler_mod

    monkeypatch.setattr(scheduler_mod.mx, "get_active_memory", lambda: 0)
    monkeypatch.setattr(scheduler_mod, "get_phys_footprint", lambda: 60)

    eviction = scheduler.preflight_eviction_request(
        num_prompt_tokens=128,
        request_id="req-safety",
    )

    assert eviction is not None
    assert eviction.reason == "prefill_safety_cap"
    assert eviction.request_id == "req-safety"
    assert eviction.current_bytes == 60
    assert eviction.predicted_transient_bytes == 50
    assert eviction.target_cap_bytes == 90
    assert eviction.requested_tokens == 4

    with pytest.raises(PrefillMemoryExceededError) as exc:
        scheduler.preflight_or_raise(
            num_prompt_tokens=128,
            request_id="req-safety",
        )

    assert "preflight safety guard" in str(exc.value)
    assert exc.value.request_id == "req-safety"
    assert exc.value.estimated_bytes == 110
    assert exc.value.limit_bytes == 90


def test_current_usage_subtracts_shared_hot_cache_bytes_from_phys_side():
    scheduler = _make_scheduler()
    scheduler.config.hot_cache_budget = SimpleNamespace(total_bytes=3 * 1024**3)

    with (
        patch("omlx.scheduler.mx.get_active_memory", return_value=4 * 1024**3),
        patch("omlx.scheduler.get_phys_footprint", return_value=10 * 1024**3),
    ):
        assert scheduler._current_usage_bytes() == 7 * 1024**3


def test_current_usage_keeps_mlx_active_as_floor_after_hot_cache_subtract():
    scheduler = _make_scheduler()
    scheduler.config.hot_cache_budget = SimpleNamespace(total_bytes=9 * 1024**3)

    with (
        patch("omlx.scheduler.mx.get_active_memory", return_value=6 * 1024**3),
        patch("omlx.scheduler.get_phys_footprint", return_value=10 * 1024**3),
    ):
        assert scheduler._current_usage_bytes() == 6 * 1024**3


def test_current_usage_falls_back_to_local_hot_cache_counter():
    scheduler = _make_scheduler()

    class _LocalHotCacheManager:
        _hot_cache_total_bytes = 2 * 1024**3

        def get_stats(self):
            raise RuntimeError("stats unavailable")

    scheduler.paged_ssd_cache_manager = _LocalHotCacheManager()

    with (
        patch("omlx.scheduler.mx.get_active_memory", return_value=1 * 1024**3),
        patch("omlx.scheduler.get_phys_footprint", return_value=8 * 1024**3),
    ):
        assert scheduler._current_usage_bytes() == 6 * 1024**3


def test_preflight_returns_none_when_guard_disabled():
    scheduler = _make_scheduler()
    scheduler._prefill_memory_guard = False
    scheduler._memory_hard_limit_bytes = 1
    assert scheduler._preflight_memory_check(_make_request(65536)) is None


def test_preflight_admits_fully_cached_request_by_passing_check():
    """A true full-cache hit (cached_tokens == num_prompt_tokens) must
    still be admitted — by PASSING the check, not by skipping it.

    With a generous limit the 1-token floored estimate fits, so the guard
    returns None (admitted). This replaces the old fail-open behaviour
    where the guard returned None unconditionally for a full cache hit
    regardless of the limit.

    ``None`` from ``_preflight_memory_check`` means "admitted" both when
    the guard PASSED and when it was SKIPPED, so this test also asserts
    the estimate is actually computed (not None) to distinguish the two.
    """
    scheduler = _make_scheduler()
    scheduler._prefill_memory_guard = True
    scheduler._memory_hard_limit_bytes = 10**18
    req = _make_request(1000)
    req.cached_tokens = 1000
    with (
        patch("omlx.scheduler.mx.get_active_memory", return_value=0),
        patch("omlx.scheduler.get_phys_footprint", return_value=0),
    ):
        est = scheduler._admission_estimate(
            num_prompt_tokens=1000,
            cached_tokens=1000,
            current=0,
        )
        assert est is not None, (
            "estimate must be computed (not skipped) for a full-cache hit"
        )
        assert est.estimated > 0
        assert scheduler._preflight_memory_check(req) is None


def test_preflight_rejects_fully_cached_request_under_pressure():
    """The credit-driven fail-open: when cached_tokens == num_prompt_tokens
    (exact over-credit) and memory is exhausted, the guard must REJECT
    rather than skip. Previously _admission_estimate returned None from
    the new_tokens == 0 early return and the guard was skipped entirely.
    """
    scheduler = _make_scheduler()
    scheduler._prefill_memory_guard = True
    scheduler._memory_hard_limit_bytes = 1  # any allocation exceeds
    req = _make_request(1000)
    req.cached_tokens = 1000
    with (
        patch("omlx.scheduler.mx.get_active_memory", return_value=0),
        patch("omlx.scheduler.get_phys_footprint", return_value=0),
    ):
        rejection = scheduler._preflight_memory_check(req)
    assert rejection is not None, (
        "guard must reject an exact over-credit under pressure, not skip"
    )
    assert rejection.estimated_bytes > 0
    assert rejection.limit_bytes == 1


def test_preflight_rejects_near_exact_over_credit_under_pressure():
    """EARLY RETURN 2 case: cached_tokens == num_prompt_tokens - 1 makes
    new_tokens == 1, which made prefill_tokens == 0 and triggered the
    second early return. The guard must still reject under pressure.
    """
    scheduler = _make_scheduler()
    scheduler._prefill_memory_guard = True
    scheduler._memory_hard_limit_bytes = 1
    req = _make_request(1000)
    req.cached_tokens = 999  # new_tokens == 1 -> prefill_tokens floored to 1
    with (
        patch("omlx.scheduler.mx.get_active_memory", return_value=0),
        patch("omlx.scheduler.get_phys_footprint", return_value=0),
    ):
        rejection = scheduler._preflight_memory_check(req)
    assert rejection is not None, (
        "guard must reject a near-exact over-credit (new_tokens == 1), "
        "not skip via the prefill_tokens == 0 early return"
    )
    assert rejection.estimated_bytes > 0


def test_admission_estimate_exact_over_credit_returns_real_estimate():
    """Direct unit test: cached_tokens == num_prompt_tokens must produce a
    real _AdmissionEstimate, not None.
    """
    scheduler = _make_scheduler()
    est = scheduler._admission_estimate(
        num_prompt_tokens=1000,
        cached_tokens=1000,
        current=0,
    )
    assert est is not None, (
        "exact over-credit must yield a real estimate, not None"
    )
    assert est.kv_exact > 0
    assert est.estimated > 0


def test_admission_estimate_near_exact_over_credit_returns_real_estimate():
    """Direct unit test: cached_tokens == num_prompt_tokens - 1 (the
    EARLY RETURN 2 case) must produce a real _AdmissionEstimate, not None.
    """
    scheduler = _make_scheduler()
    est = scheduler._admission_estimate(
        num_prompt_tokens=1000,
        cached_tokens=999,
        current=0,
    )
    assert est is not None, (
        "near-exact over-credit (new_tokens == 1) must yield a real "
        "estimate, not None"
    )
    assert est.kv_exact > 0
    assert est.estimated > 0


def test_admission_estimate_absurd_over_credit_returns_real_estimate():
    """Direct unit test: cached_tokens > num_prompt_tokens (absurd
    over-credit) must produce a real _AdmissionEstimate, not None, and
    must not raise.
    """
    scheduler = _make_scheduler()
    est = scheduler._admission_estimate(
        num_prompt_tokens=1000,
        cached_tokens=2000,
        current=0,
    )
    assert est is not None, (
        "absurd over-credit must yield a real estimate, not None"
    )
    assert est.kv_exact > 0
    assert est.estimated > 0


def test_admission_estimate_returns_none_when_monitor_is_none():
    """The genuinely-uninformative case: memory_monitor is None must still
    return None. This is one of two legitimate None paths; the other is
    when ``kv_exact <= 0 and transient <= 0`` (missing model dims).
    """
    scheduler = _make_scheduler()
    scheduler.memory_monitor = None
    est = scheduler._admission_estimate(
        num_prompt_tokens=1000,
        cached_tokens=1000,
        current=0,
    )
    assert est is None


def test_preflight_or_raise_rejects_exact_over_credit_under_pressure():
    """Drive the fix through preflight_or_raise (cached_kv_resident=False),
    not only via _admission_estimate directly. The route-time path gets
    cached_tokens from the peek, which can over-report; when it does and
    memory is tight, the guard must raise, not silently admit.
    """
    scheduler = _make_scheduler()
    scheduler._prefill_memory_guard = True
    scheduler._memory_hard_limit_bytes = 1
    with (
        patch("omlx.scheduler.mx.get_active_memory", return_value=0),
        patch("omlx.scheduler.get_phys_footprint", return_value=0),
        pytest.raises(PrefillMemoryExceededError) as exc,
    ):
        scheduler.preflight_or_raise(
            num_prompt_tokens=1000,
            cached_tokens=1000,
            request_id="req-over-credit",
        )
    assert exc.value.estimated_bytes > 0
    assert exc.value.limit_bytes == 1


def test_preflight_rejects_heavily_cached_long_context():
    """Regression for M3: a request whose suffix is small but whose
    *full* prompt is long must still trip the guard, because the SDPA
    fallback score matrix spans the full prompt (cached + new), not just the
    new tokens. Previously the estimator passed only new_tokens to the
    fallback formula and the heavily-cached path slipped through.
    """
    scheduler = _make_scheduler()
    scheduler._prefill_memory_guard = True
    # Tight limit so even a partial prefill against a 100k KV trips it.
    scheduler._memory_hard_limit_bytes = 100 * 1024**2  # 100 MB
    req = _make_request(100_000)
    req.cached_tokens = 99_000  # only 1k new tokens but kv_len = 100k
    with (
        patch("omlx.scheduler.mx.get_active_memory", return_value=0),
        patch("omlx.scheduler.get_phys_footprint", return_value=0),
    ):
        error = scheduler._preflight_memory_check(req)
    assert error is not None, (
        "guard must trip on heavily-cached long-context: SDPA scores "
        "still span the full prompt"
    )


def test_preflight_rejects_uncached_long_context():
    """Symmetric to test_preflight_rejects_heavily_cached_long_context:
    a request with mostly NEW tokens (no cache) at a 100k prompt must
    also trip the guard. This locks in the high-head-dim SDPA span formula
    in both directions; if a future refactor regressed the cached path
    OR the uncached path, only one of these two tests would fail.
    """
    scheduler = _make_scheduler()
    scheduler._prefill_memory_guard = True
    scheduler._memory_hard_limit_bytes = 100 * 1024**2  # 100 MB
    req = _make_request(100_000)
    req.cached_tokens = 1_000  # almost everything is new
    with (
        patch("omlx.scheduler.mx.get_active_memory", return_value=0),
        patch("omlx.scheduler.get_phys_footprint", return_value=0),
    ):
        error = scheduler._preflight_memory_check(req)
    assert error is not None, "guard must trip on uncached long-context too"


class _VLMConfig:
    """Top-level VLM config whose LM dims live under text_config (Qwen3.6-VL,
    Gemma-4 layout). The top-level surface deliberately has no num_hidden_layers,
    so this exercises the nested-config descent path."""

    def __init__(self):
        self.architectures = ["Qwen3_5MoeForConditionalGeneration"]
        self.model_type = "qwen3_5_moe"
        self.text_config = _ModelConfig(
            num_hidden_layers=40,
            num_key_value_heads=2,
            num_attention_heads=16,
            head_dim=256,  # > 128 → high-head-dim tiled SDPA scratch
        )


def _make_vlm_scheduler() -> Scheduler:
    model = MagicMock()
    model.layers = []
    model.config = _VLMConfig()
    del model.make_cache

    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2

    config = SchedulerConfig(
        max_num_seqs=8,
        prefill_step_size=2048,
        paged_cache_block_size=0,
    )
    return Scheduler(model=model, tokenizer=tokenizer, config=config)


def test_vlm_nested_config_populates_estimator_dims():
    """Regression: VLM models nest LM dims under config.text_config — the
    estimator must follow the sub-config or it stays silently dead at
    runtime (no Model info set log, peak == 0, guard short-circuits)."""
    scheduler = _make_vlm_scheduler()
    monitor = scheduler.memory_monitor
    assert monitor is not None
    assert monitor._num_layers == 40
    assert monitor._num_kv_heads == 2
    assert monitor._num_attention_heads == 16
    assert monitor._head_dim == 256


def test_vlm_estimator_produces_nonzero_peak():
    scheduler = _make_vlm_scheduler()
    assert scheduler.memory_monitor is not None
    # 90k tokens at head_dim=256 / n_q=16 should yield a multi-GiB peak:
    # KV growth plus a bounded tiled SDPA scratch term.
    peak = scheduler.memory_monitor.estimate_prefill_peak_bytes(90000, 2048)
    assert peak > 7 * 1024 * 1024 * 1024  # > 7 GiB


def test_dict_nested_config_populates_estimator_dims_and_preflight_rejects():
    """Real Qwen3.6 text-only packs can expose LM dims as a dict-valued
    ``text_config``. The guard must read that shape too; otherwise real
    servers keep the estimator dim-less and route preflight becomes a no-op.
    """
    model = MagicMock()
    model.layers = []
    model.config = {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "num_hidden_layers": 40,
            "num_key_value_heads": 2,
            "num_attention_heads": 16,
            "head_dim": 256,
        },
    }
    del model.make_cache

    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    scheduler = Scheduler(
        model=model,
        tokenizer=tokenizer,
        config=SchedulerConfig(
            max_num_seqs=8,
            prefill_step_size=2048,
            paged_cache_block_size=0,
        ),
    )

    monitor = scheduler.memory_monitor
    assert monitor is not None
    assert monitor._num_layers == 40
    assert monitor._num_kv_heads == 2
    assert monitor._num_attention_heads == 16
    assert monitor._head_dim == 256

    scheduler._prefill_memory_guard = True
    scheduler._memory_hard_limit_bytes = 2 * 1024**3
    with (
        patch("omlx.scheduler.mx.get_active_memory", return_value=0),
        patch("omlx.scheduler.get_phys_footprint", return_value=0),
        pytest.raises(PrefillMemoryExceededError),
    ):
        scheduler.preflight_or_raise(num_prompt_tokens=50_000, request_id="dict-cfg")


def test_rejection_releases_block_aware_cache_when_present():
    """Regression for the prefix-cache leak found in review: a request
    rejected by the prefill memory guard had its ref counts on every
    prefix-matched paged block (and its ``request_tables`` entry)
    incremented by ``add_request → fetch_cache``. Without releasing
    them on the rejection path, those refs pin the paged cache and
    compound the very memory pressure that triggered the rejection.
    """
    scheduler = _make_scheduler()
    block_aware_cache = MagicMock()
    paged_cache_manager = MagicMock()
    scheduler.block_aware_cache = block_aware_cache
    scheduler.paged_cache_manager = paged_cache_manager

    scheduler._release_paged_cache_for_request("req-leak")

    # When block_aware_cache is present it owns the cleanup chain
    # (release_cache → paged_cache_manager.delete_block_table).
    block_aware_cache.release_cache.assert_called_once_with("req-leak")
    paged_cache_manager.delete_block_table.assert_not_called()


def test_rejection_releases_paged_cache_when_no_prefix_cache():
    """When block_aware_cache is absent but a paged_cache_manager is
    wired up, the rejection path must call ``delete_block_table``
    directly — otherwise the request's ``request_tables`` entry and
    every block ref it holds leaks for the process lifetime.
    """
    scheduler = _make_scheduler()
    scheduler.block_aware_cache = None
    paged_cache_manager = MagicMock()
    scheduler.paged_cache_manager = paged_cache_manager

    scheduler._release_paged_cache_for_request("req-leak")

    paged_cache_manager.delete_block_table.assert_called_once_with("req-leak")


def test_rejection_releases_draft_prefix_cache_for_specprefill_requests():
    """SpecPrefill primes an independent ``_draft_prefix_cache`` in
    ``_try_specprefill_scoring`` (via its own ``fetch_cache``).
    The rejection path must release that draft cache too, symmetric
    to the target cache — otherwise a rejected SpecPrefill request
    leaks every draft-block ref and orphans its ``_request_tables``
    entry exactly like the target-cache bug this commit fixes."""
    scheduler = _make_scheduler()
    scheduler.block_aware_cache = MagicMock()
    scheduler.paged_cache_manager = MagicMock()
    draft_cache = MagicMock()
    scheduler._draft_prefix_cache = draft_cache

    scheduler._release_paged_cache_for_request("req-spec-leak")

    draft_cache.release_cache.assert_called_once_with("req-spec-leak")


def test_rejection_helper_noop_without_caches():
    """No caches wired up → helper must not raise. Embedded test
    schedulers (this file's ``_make_scheduler``) build without paged
    caches; the helper must be safe to call unconditionally on the
    rejection path."""
    scheduler = _make_scheduler()
    scheduler.block_aware_cache = None
    scheduler.paged_cache_manager = None
    # Must not raise.
    scheduler._release_paged_cache_for_request("req-leak")


def test_preflight_rejection_path_invokes_release_helper():
    """End-to-end wiring: the preflight rejection in ``_schedule_waiting``
    must invoke the cache-release helper before popping
    ``self.requests``. Pins the call-site fix for the leak — without
    this hook the helper could exist but never be called from the hot
    path.
    """
    scheduler = _make_scheduler()
    scheduler._prefill_memory_guard = True
    scheduler._memory_hard_limit_bytes = 1  # forces rejection

    req = _make_request(65536)
    scheduler.requests[req.request_id] = req
    scheduler.waiting.append(req)

    # Make the rejection branch take effect even before
    # _ensure_batch_generator runs — patch the preflight check to
    # short-circuit on entry and keep this test independent of the
    # batch-generator construction path.
    from omlx.scheduler import _PreflightRejection

    def _force_reject(_request):
        return _PreflightRejection(
            message="forced rejection for test",
            estimated_bytes=1,
            limit_bytes=1,
        )

    with (
        patch.object(scheduler, "_release_paged_cache_for_request") as release_spy,
        patch.object(scheduler, "_preflight_memory_check", side_effect=_force_reject),
        patch.object(scheduler, "_ensure_batch_generator", return_value=None),
    ):
        # Pretend a batch_generator exists so the loop continues past
        # the ``if self.batch_generator is None: break`` guard.
        scheduler.batch_generator = MagicMock()
        scheduler._schedule_waiting()

    release_spy.assert_any_call(req.request_id)
    assert req.request_id not in scheduler.requests


def test_vlm_preflight_rejects_oversize_request():
    scheduler = _make_vlm_scheduler()
    scheduler._prefill_memory_guard = True
    scheduler._memory_hard_limit_bytes = 36 * 1024 * 1024 * 1024  # 36 GiB hard limit

    with (
        patch("omlx.scheduler.mx.get_active_memory", return_value=28 * 1024**3),
        patch("omlx.scheduler.get_phys_footprint", return_value=28 * 1024**3),
    ):
        # 100k tokens at head_dim=256 should push (28 GiB baseline + KV+SDPA
        # peak) past the 36 GiB limit.
        rejection = scheduler._preflight_memory_check(_make_request(100000))

    assert rejection is not None
    assert "KV+SDPA" in rejection.message


# ---------------------------------------------------------------------------
# Config-descent edge cases (M3 in the upstream review of this commit)
# ---------------------------------------------------------------------------


class _VLMTopLevelVisionConfig:
    """Top-level config has num_hidden_layers that refers to the *vision*
    encoder. The estimator must descend into text_config rather than
    accept the top-level value, otherwise it miscalibrates the SDPA peak.
    """

    def __init__(self):
        self.architectures = ["FakeVisionLM"]
        self.model_type = "fake_vlm"
        # Vision encoder block count surfaces at top-level on some
        # HF auto-wrapped packs — accepting this would silently use
        # 27 layers / wrong heads for the LM math.
        self.num_hidden_layers = 27
        self.num_attention_heads = 16  # vision attn heads
        self.head_dim = 80  # vision head_dim (< 128, different SDPA path)
        self.text_config = _ModelConfig(
            num_hidden_layers=40,
            num_key_value_heads=2,
            num_attention_heads=16,
            head_dim=256,  # LM head_dim → SDPA-fallback path
        )


def test_vlm_descent_prefers_text_config_over_top_level_vision_field():
    """Regression: top-level num_hidden_layers can refer to the vision
    encoder; the estimator must prefer text_config when present."""
    model = MagicMock()
    model.layers = []
    model.config = _VLMTopLevelVisionConfig()
    del model.make_cache
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    cfg = SchedulerConfig(
        max_num_seqs=8,
        prefill_step_size=2048,
        paged_cache_block_size=0,
    )
    sched = Scheduler(model=model, tokenizer=tokenizer, config=cfg)

    monitor = sched.memory_monitor
    assert monitor is not None
    # Must be the LM dims from text_config, NOT vision (27 / 80).
    assert monitor._num_layers == 40
    assert monitor._head_dim == 256


class _AltSubConfigContainer:
    """Some packs name the LM sub-config ``language_config`` (or
    ``llm_config``) instead of ``text_config``."""

    def __init__(self, sub_attr_name: str):
        self.architectures = ["AltSubConfigVLM"]
        sub = _ModelConfig(
            num_hidden_layers=24,
            num_key_value_heads=4,
            num_attention_heads=24,
            head_dim=192,
        )
        setattr(self, sub_attr_name, sub)


def test_vlm_descent_handles_language_config_alias():
    model = MagicMock()
    model.layers = []
    model.config = _AltSubConfigContainer("language_config")
    del model.make_cache
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    sched = Scheduler(
        model=model,
        tokenizer=tokenizer,
        config=SchedulerConfig(
            max_num_seqs=8,
            prefill_step_size=2048,
            paged_cache_block_size=0,
        ),
    )
    assert sched.memory_monitor._num_layers == 24
    assert sched.memory_monitor._head_dim == 192


def test_vlm_descent_handles_llm_config_alias():
    model = MagicMock()
    model.layers = []
    model.config = _AltSubConfigContainer("llm_config")
    del model.make_cache
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    sched = Scheduler(
        model=model,
        tokenizer=tokenizer,
        config=SchedulerConfig(
            max_num_seqs=8,
            prefill_step_size=2048,
            paged_cache_block_size=0,
        ),
    )
    assert sched.memory_monitor._num_layers == 24


class _LegacyLMConfig:
    """GPT-style legacy config exposing ``n_layer`` / ``n_head`` / ``n_embd``
    instead of HuggingFace's ``num_hidden_layers`` etc."""

    def __init__(self):
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768  # head_dim derived as n_embd / n_head = 64


def test_legacy_n_layer_fallback_path():
    """The extractor falls back to ``n_layer`` / ``n_head`` / ``n_embd`` for
    GPT-style configs and derives head_dim when not directly present."""
    model = MagicMock()
    model.layers = []
    model.config = _LegacyLMConfig()
    del model.make_cache
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    sched = Scheduler(
        model=model,
        tokenizer=tokenizer,
        config=SchedulerConfig(
            max_num_seqs=8,
            prefill_step_size=2048,
            paged_cache_block_size=0,
        ),
    )
    monitor = sched.memory_monitor
    assert monitor is not None
    assert monitor._num_layers == 12
    assert monitor._num_kv_heads == 12  # falls back to n_head
    assert monitor._head_dim == 64  # n_embd / n_head


class _BrokenConfig:
    """A config whose attribute access raises — exercises the outer
    try/except wrap in _set_model_info_for_monitor."""

    @property
    def num_hidden_layers(self):
        raise RuntimeError("synthetic boom")


class _VLMWithNestedLegacyLayer:
    """Hypothetical VLM whose LM sub-config exposes only the legacy
    GPT-style ``n_layer`` (no ``num_hidden_layers``). The descent rule
    must accept this so the LM dims aren't shadowed by the top-level
    vision-encoder dims.
    """

    def __init__(self):
        self.architectures = ["LegacyNestedVLM"]
        # Top-level matches vision encoder dims that should be ignored.
        self.num_hidden_layers = 27
        self.num_key_value_heads = 16
        self.num_attention_heads = 16
        self.head_dim = 80
        self.text_config = _ModelConfig(
            num_hidden_layers=None,
            num_key_value_heads=8,
            num_attention_heads=32,
            head_dim=128,
        )
        # Force the sub-config to surface only n_layer, not
        # num_hidden_layers.
        self.text_config.num_hidden_layers = None
        self.text_config.n_layer = 36


def test_vlm_descent_prefers_text_config_via_legacy_n_layer():
    """Regression: the sub-config preference rule must accept legacy
    ``n_layer`` in addition to ``num_hidden_layers`` so the descent
    isn't silently skipped when only the legacy alias is present —
    otherwise the top-level (vision) dims leak into the SDPA-peak
    calculation.
    """
    model = MagicMock()
    model.layers = []
    model.config = _VLMWithNestedLegacyLayer()
    del model.make_cache
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    sched = Scheduler(
        model=model,
        tokenizer=tokenizer,
        config=SchedulerConfig(
            max_num_seqs=8,
            prefill_step_size=2048,
            paged_cache_block_size=0,
        ),
    )
    monitor = sched.memory_monitor
    assert monitor is not None
    # Must be the LM dims (n_layer=36, head_dim=128), NOT vision (27/80).
    assert monitor._num_layers == 36
    assert monitor._head_dim == 128


def test_exception_during_descent_is_swallowed():
    """The whole _set_model_info_for_monitor body is wrapped in
    try/except so a malformed config can't break Scheduler init."""
    model = MagicMock()
    model.layers = []
    model.config = _BrokenConfig()
    del model.make_cache
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    # Must not raise.
    sched = Scheduler(
        model=model,
        tokenizer=tokenizer,
        config=SchedulerConfig(
            max_num_seqs=8,
            prefill_step_size=2048,
            paged_cache_block_size=0,
        ),
    )
    # Monitor exists but dims stayed None — estimator returns 0 / guard skips.
    assert sched.memory_monitor is not None
    assert sched.memory_monitor._num_layers is None


def test_scheduler_init_populates_rotating_specs():
    """Hybrid make_cache classification reaches the monitor: full layers
    counted strictly, rotating layers grouped by window."""
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    model = MagicMock()
    model.layers = []
    model.config = _ModelConfig()
    model.make_cache = lambda: (
        [KVCache() for _ in range(5)]
        + [RotatingKVCache(max_size=1024) for _ in range(27)]
    )

    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    config = SchedulerConfig(
        max_num_seqs=8, prefill_step_size=2048, paged_cache_block_size=0
    )
    scheduler = Scheduler(model=model, tokenizer=tokenizer, config=config)

    monitor = scheduler.memory_monitor
    assert monitor is not None
    assert monitor._num_kv_cache_layers == 5
    assert monitor._rotating_layer_specs == ((27, 1024),)
    # No ArraysCache layers: the fixed-state probe stays unarmed.
    assert scheduler._fixed_state_measure_armed is False


def test_admission_estimate_is_the_single_formula():
    """Every preflight path prices current + kv_exact + transient."""
    scheduler = _make_scheduler()
    scheduler._prefill_memory_guard = True
    scheduler._memory_hard_limit_bytes = 10**18

    with (
        patch("omlx.scheduler.mx.get_active_memory", return_value=0),
        patch("omlx.scheduler.get_phys_footprint", return_value=0),
    ):
        est = scheduler._admission_estimate(
            num_prompt_tokens=32768, cached_tokens=0, current=0
        )
    assert est is not None
    floor = min(max(1, scheduler._prefill_min_chunk_tokens), 32767)
    pre_chunk_kv_len = 32767 - floor
    assert est.floor_chunk == floor
    assert est.kv_len == pre_chunk_kv_len
    assert est.kv_exact == int(
        scheduler.memory_monitor.estimate_resident_kv_bytes(
            32768, chunk_tokens=floor
        )
    )
    assert est.transient == int(
        scheduler._admission_transient_bound(floor, pre_chunk_kv_len)
    )
    assert est.estimated == est.kv_exact + est.transient


def test_route_admission_prices_cached_prefix_as_resident_kv():
    """Route-time preflight must charge the cached prefix as resident-to-be
    KV, not as already-resident.

    At HTTP route time the stored prefix has not been materialized yet, so
    ``cached_kv_resident=False`` makes ``_admission_estimate`` price the
    full prompt's KV (``num_prompt_tokens``) via
    ``estimate_resident_kv_bytes``.  The in-stream re-check keeps the
    default ``True`` and charges only ``new_tokens`` because
    ``_prepare_prefix_cache_for_request`` has already loaded the hit.

    The ordering invariant is ``e_resident < e_route < e_cold``: an
    over-credit can only cost latency (the in-stream re-check pauses),
    never memory, because the credited KV is priced as resident-to-be.
    """
    scheduler = _make_scheduler()
    scheduler._prefill_memory_guard = True
    scheduler._memory_hard_limit_bytes = 10**18

    real_fn = scheduler.memory_monitor.estimate_resident_kv_bytes
    spy = MagicMock(wraps=real_fn)
    scheduler.memory_monitor.estimate_resident_kv_bytes = spy

    with (
        patch("omlx.scheduler.mx.get_active_memory", return_value=0),
        patch("omlx.scheduler.get_phys_footprint", return_value=0),
    ):
        e_route = scheduler._admission_estimate(
            num_prompt_tokens=100,
            cached_tokens=96,
            current=0,
            cached_kv_resident=False,
        )
        route_call_tokens = spy.call_args.args[0]

        spy.reset_mock()
        e_resident = scheduler._admission_estimate(
            num_prompt_tokens=100,
            cached_tokens=96,
            current=0,
        )
        resident_call_tokens = spy.call_args.args[0]

        spy.reset_mock()
        e_cold = scheduler._admission_estimate(
            num_prompt_tokens=100,
            cached_tokens=0,
            current=0,
        )
        cold_call_tokens = spy.call_args.args[0]

    assert e_route is not None
    assert e_resident is not None
    assert e_cold is not None

    # Route-time charges the full prompt; in-stream charges only new tokens.
    assert route_call_tokens == 100
    assert resident_call_tokens == 4
    assert cold_call_tokens == 100

    # Over-credit is safe: e_route prices the credited KV as resident-to-be,
    # so it can only be tighter than the cold prefill, never looser than the
    # in-stream check.
    assert e_resident.estimated < e_route.estimated < e_cold.estimated

    # The route-time path through preflight_or_raise must also charge the
    # full prompt (num_prompt_tokens) not just new_tokens, so the cached
    # prefix is priced as resident-to-be KV at HTTP time.
    with (
        patch("omlx.scheduler.mx.get_active_memory", return_value=0),
        patch("omlx.scheduler.get_phys_footprint", return_value=0),
    ):
        spy.reset_mock()
        scheduler.preflight_or_raise(
            num_prompt_tokens=100, cached_tokens=96, request_id="req-route"
        )
        assert spy.call_args.args[0] == 100


def test_admission_charges_full_step_under_speed_priority():
    """Speed priority prices the full prefill_step_size chunk instead of the
    throttle floor, so admission only accepts what completes at full speed."""
    scheduler = _make_scheduler()
    scheduler._prefill_memory_guard = True
    scheduler._memory_hard_limit_bytes = 10**18

    patches = (
        patch("omlx.scheduler.mx.get_active_memory", return_value=0),
        patch("omlx.scheduler.get_phys_footprint", return_value=0),
    )
    with patches[0], patches[1]:
        est_context = scheduler._admission_estimate(
            num_prompt_tokens=32768, cached_tokens=0, current=0
        )

    scheduler._prefill_speed_priority = True
    with patches[0], patches[1]:
        est_speed = scheduler._admission_estimate(
            num_prompt_tokens=32768, cached_tokens=0, current=0
        )

    assert est_context is not None and est_speed is not None
    assert est_context.floor_chunk == min(
        max(1, scheduler._prefill_min_chunk_tokens), 32768
    )
    assert est_speed.floor_chunk == scheduler.config.prefill_step_size
    assert est_speed.kv_exact == int(
        scheduler.memory_monitor.estimate_resident_kv_bytes(
            32768, chunk_tokens=scheduler.config.prefill_step_size
        )
    )
    assert est_speed.transient == int(
        scheduler._admission_transient_bound(
            scheduler.config.prefill_step_size,
            32767 - scheduler.config.prefill_step_size,
        )
    )
    # The full-step charge is strictly more conservative.
    assert est_speed.estimated > est_context.estimated

    # Prompts shorter than the step are charged at their own size.
    with patches[0], patches[1]:
        est_small = scheduler._admission_estimate(
            num_prompt_tokens=1024, cached_tokens=0, current=0
        )
    assert est_small is not None
    assert est_small.floor_chunk == 1023


def test_deepseek_v4_200k_native_admission_avoids_81_gib_dense_charge(
    monkeypatch,
):
    """Issue #2521: V4's local + pooled sparse cache must not be priced as
    43 full-context K/V layers followed by a dense 200K SDPA."""
    from mlx_lm.models.cache import RotatingKVCache

    import omlx.memory_monitor as memory_monitor
    from omlx.memory_monitor import estimate_unfused_sdpa_call_bytes
    from omlx.patches.deepseek_v4 import wsdpa_attention as wsdpa

    monkeypatch.setattr(
        memory_monitor,
        "native_indexer_eligible",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(wsdpa, "_ENABLED", True)
    monkeypatch.setattr(wsdpa, "_TOPK_ENABLED", True)
    monkeypatch.setattr(wsdpa, "_broken", False)
    monkeypatch.setattr(wsdpa, "_ready", False)
    monkeypatch.setattr(wsdpa, "_topk_ready", False)

    config = _ModelConfig(
        num_hidden_layers=43,
        num_key_value_heads=1,
        num_attention_heads=64,
        head_dim=512,
    )
    config.model_type = "deepseek_v4"
    config.sliding_window = 128
    config.index_n_heads = 64
    config.index_head_dim = 128
    config.index_topk = 512
    config.compress_ratios = [0, 0] + [4, 128] * 20 + [4]

    model = MagicMock()
    model.layers = []
    model.config = config
    del model.dtype
    model.model = SimpleNamespace(
        embed_tokens=SimpleNamespace(
            weight=mx.zeros((1,), dtype=mx.bfloat16),
        )
    )
    model.make_cache = lambda: [RotatingKVCache(max_size=128) for _ in range(43)]
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    scheduler = Scheduler(
        model=model,
        tokenizer=tokenizer,
        config=SchedulerConfig(
            max_num_seqs=8,
            prefill_step_size=2048,
            paged_cache_block_size=0,
        ),
    )
    scheduler._prefill_speed_priority = True

    monitor = scheduler.memory_monitor
    assert monitor is not None
    gib = 1024**3
    current = int(156.05 * gib)
    limit = int(235.96 * gib)
    cold_admission = scheduler._admission_estimate(
        num_prompt_tokens=200_000,
        cached_tokens=0,
        current=current,
    )
    assert cold_admission is not None
    assert cold_admission.estimated < limit

    cold_fallback = monitor.estimate_chunk_transient_bytes(2048, 66_000)
    monkeypatch.setattr(wsdpa, "_ready", True)
    monkeypatch.setattr(wsdpa, "_topk_ready", True)
    active = monitor.estimate_chunk_transient_bytes(2048, 66_000)
    monkeypatch.setattr(wsdpa, "_broken", True)
    failed_fallback = monitor.estimate_chunk_transient_bytes(2048, 66_000)
    assert active < cold_fallback
    assert failed_fallback == cold_fallback
    monkeypatch.setattr(wsdpa, "_broken", False)

    est = scheduler._admission_estimate(
        num_prompt_tokens=200_000,
        cached_tokens=0,
        current=current,
    )
    assert est is not None
    assert est.estimated < limit
    assert est.kv_exact < 2 * gib
    assert est.transient < 20 * gib

    old_kv = 200_000 * 43 * 512 * 2 * 2 + 43 * (128 + 2048 - 1) * 512 * 2 * 2
    old_sdpa = estimate_unfused_sdpa_call_bytes(64, 2048, 202_047, 512, 2)
    old_chunk_kv = 2048 * 43 * 512 * 2 * 2
    old_peak = old_kv + (old_sdpa + old_chunk_kv) * 1.3
    assert old_peak / gib == pytest.approx(81.25, abs=0.02)
    assert current + old_peak > limit


def test_preflight_charges_observed_max_transient():
    """A session's observed max chunk transient converts a would-be
    mid-prefill abort into an upfront 400."""
    scheduler = _make_scheduler()
    scheduler._prefill_memory_guard = True
    # Keep the safety cap out of the way so the hard limit drives.
    scheduler._memory_abort_limit_bytes = 10**18

    patches = (
        patch("omlx.scheduler.mx.get_active_memory", return_value=0),
        patch("omlx.scheduler.get_phys_footprint", return_value=0),
    )
    with patches[0], patches[1]:
        est = scheduler._admission_estimate(
            num_prompt_tokens=32768, cached_tokens=0, current=0
        )
    assert est is not None
    scheduler._memory_hard_limit_bytes = int(est.estimated) + 1

    with patches[0], patches[1]:
        scheduler.preflight_or_raise(num_prompt_tokens=32768)  # fits

    scheduler._prefill_transient_tracker._dense_history.observed_max_bytes = (
        est.transient + 2 * 1024**3
    )
    with patches[0], patches[1], pytest.raises(PrefillMemoryExceededError):
        scheduler.preflight_or_raise(num_prompt_tokens=32768)


def test_admission_compares_against_hard_watermark():
    """The enforcer kills at the watermark, so admission must not admit
    into the watermark..hard-limit band."""
    scheduler = _make_scheduler()
    scheduler._prefill_memory_guard = True
    scheduler._memory_abort_limit_bytes = 10**18  # keep safety cap out

    patches = (
        patch("omlx.scheduler.mx.get_active_memory", return_value=0),
        patch("omlx.scheduler.get_phys_footprint", return_value=0),
    )
    with patches[0], patches[1]:
        est = scheduler._admission_estimate(
            num_prompt_tokens=32768, cached_tokens=0, current=0
        )
    assert est is not None

    # Watermark unset: falls back to the hard limit, request fits.
    scheduler._memory_hard_limit_bytes = int(est.estimated) + 1
    scheduler._memory_hard_watermark_bytes = 0
    with patches[0], patches[1]:
        scheduler.preflight_or_raise(num_prompt_tokens=32768)

    # Watermark below the estimate: the same request is now an upfront 400.
    scheduler._memory_hard_watermark_bytes = int(est.estimated) - 1
    with patches[0], patches[1], pytest.raises(PrefillMemoryExceededError) as ei:
        scheduler.preflight_or_raise(num_prompt_tokens=32768)
    assert ei.value.limit_bytes == int(est.estimated) - 1

    # Watermark above the estimate: admitted again.
    scheduler._memory_hard_watermark_bytes = int(est.estimated) + 1
    with patches[0], patches[1]:
        scheduler.preflight_or_raise(num_prompt_tokens=32768)


def _qwen4_prefill_profile():
    from omlx.memory_monitor import make_prefill_memory_profile

    return make_prefill_memory_profile(
        SimpleNamespace(
            model_type="qwen4_exp",
            num_hidden_layers=48,
            num_attention_heads=24,
            num_key_value_heads=2,
            head_dim=256,
            indexer_n_heads=4,
            indexer_head_dim=128,
            indexer_budget=2048,
            indexer_compress_ratio=4,
            full_attention_interval=4,
            layer_types=None,
        ),
        compute_dtype_size=2,
    )


def _attach_qwen4_profile(scheduler: Scheduler) -> None:
    profile = _qwen4_prefill_profile()
    scheduler.memory_monitor.set_model_info(
        num_layers=48,
        num_kv_heads=2,
        head_dim=256,
        dtype_size=2,
        num_attention_heads=24,
        compute_dtype_size=2,
        prefill_memory_profile=profile,
    )


def test_qwen4_text_admission_uses_gathered_transient():
    scheduler = _make_scheduler()
    _attach_qwen4_profile(scheduler)
    current = 147 * 1024**3
    dense = scheduler._admission_estimate(
        num_prompt_tokens=233_472,
        cached_tokens=0,
        current=current,
        text_only=False,
    )
    gathered = scheduler._admission_estimate(
        num_prompt_tokens=233_472,
        cached_tokens=0,
        current=current,
        text_only=True,
    )
    assert dense is not None and gathered is not None
    assert gathered.kv_exact == dense.kv_exact
    assert gathered.transient * 4 < dense.transient
    assert scheduler._qwen4_text_gathered_pricing(True) is True
    assert scheduler._qwen4_text_gathered_pricing(False) is False


def test_qwen4_preflight_doors_use_gathered_for_text_only():
    scheduler = _make_scheduler()
    _attach_qwen4_profile(scheduler)
    scheduler._prefill_memory_guard = True
    current = 147 * 1024**3
    dense = scheduler._admission_estimate(
        num_prompt_tokens=233_472,
        cached_tokens=0,
        current=current,
        text_only=False,
    )
    gathered = scheduler._admission_estimate(
        num_prompt_tokens=233_472,
        cached_tokens=0,
        current=current,
        text_only=True,
    )
    assert dense is not None and gathered is not None
    cap = (dense.estimated + gathered.estimated) // 2
    scheduler._memory_hard_limit_bytes = cap
    scheduler._memory_abort_limit_bytes = 10**18
    with (
        patch("omlx.scheduler.mx.get_active_memory", return_value=current),
        patch("omlx.scheduler.get_phys_footprint", return_value=current),
    ):
        with pytest.raises(PrefillMemoryExceededError):
            scheduler.preflight_or_raise(
                num_prompt_tokens=233_472, text_only=False
            )
        scheduler.preflight_or_raise(num_prompt_tokens=233_472, text_only=True)
        assert (
            scheduler.preflight_eviction_request(
                num_prompt_tokens=233_472, text_only=True
            )
            is None
        )
        assert (
            scheduler.preflight_eviction_request(
                num_prompt_tokens=233_472, text_only=False
            )
            is not None
        )


def test_qwen4_image_request_preflight_stays_dense():
    scheduler = _make_scheduler()
    _attach_qwen4_profile(scheduler)
    scheduler._prefill_memory_guard = True
    current = 147 * 1024**3
    dense = scheduler._admission_estimate(
        num_prompt_tokens=233_472,
        cached_tokens=0,
        current=current,
        text_only=False,
    )
    gathered = scheduler._admission_estimate(
        num_prompt_tokens=233_472,
        cached_tokens=0,
        current=current,
        text_only=True,
    )
    assert dense is not None and gathered is not None
    scheduler._memory_hard_limit_bytes = (dense.estimated + gathered.estimated) // 2
    scheduler._memory_abort_limit_bytes = 10**18
    request = _make_request(233_472)
    request.vlm_inputs_embeds = object()
    with (
        patch("omlx.scheduler.mx.get_active_memory", return_value=current),
        patch("omlx.scheduler.get_phys_footprint", return_value=current),
    ):
        rejection = scheduler._preflight_memory_check(request)
    assert rejection is not None


def _peek_scheduler(
    *,
    required: bool | None,
    split: bool = False,
    cached: int = 8,
    with_cache: bool = True,
) -> Scheduler:
    """Scheduler slice holding only what estimate_cached_prefix_tokens reads."""
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = SchedulerConfig(paged_cache_block_size=4)
    scheduler.block_aware_cache = (
        MagicMock(
            peek_cached_prefix_tokens=MagicMock(return_value=cached),
            peek_cached_prefix_split=MagicMock(
                return_value=(0, cached)
            ),
        )
        if with_cache
        else None
    )
    scheduler._boundary_snapshot_required = required
    scheduler._gdn_split_active = MagicMock(return_value=split)
    # Spied, never stubbed out: the real one runs make_cache() and mutates the
    # model, which a route-time estimate must not trigger.
    scheduler._detect_boundary_snapshot_need = MagicMock(return_value=True)
    return scheduler


def test_estimate_cached_prefix_applies_stateful_exact_hit_rule():
    """The stateful exact-hit rule lives here, and only here.

    A full-prompt cache hit cannot kick off generation directly: the model
    needs state at N-1. Sliceable caches get there by trimming one token,
    stateful ones cannot, and a split GDN sidecar can only be rewound a whole
    block. An unresolved answer must fail closed to the stateful behaviour
    rather than resolve itself, because resolving mutates the model.
    """
    exact = list(range(8))  # exactly the cached length
    partial = list(range(9))  # one token beyond the cached prefix

    def estimate(**kwargs) -> int:
        scheduler = _peek_scheduler(**kwargs)
        cached = scheduler.estimate_cached_prefix_tokens(exact)
        # Asserted before the value is compared: a lazy resolve that happens
        # to return the same boolean is invisible behaviourally, yet it still
        # ran make_cache() and mutated the model.
        assert not scheduler._detect_boundary_snapshot_need.called
        return cached

    # Stateful + exact hit + GDN split: re-prefill only the last block.
    assert estimate(required=True, split=True) == 4

    # Stateful + exact hit, no split: the whole prompt is recomputed.
    assert estimate(required=True, split=False) == 0

    # Unresolved fails closed to the stateful answer in both split modes.
    assert estimate(required=None, split=True) == 4
    assert estimate(required=None, split=False) == 0

    # Resolved as sliceable: the exact hit is reusable as-is.
    assert estimate(required=False, split=True) == 8
    assert estimate(required=False, split=False) == 8

    # A partial hit is never an exact hit, so every mode reports it unchanged.
    for required in (True, False, None):
        for split in (True, False):
            scheduler = _peek_scheduler(required=required, split=split)
            assert scheduler.estimate_cached_prefix_tokens(partial) == 8
            assert not scheduler._detect_boundary_snapshot_need.called

    # No prefix cache wired up at all.
    uncached = _peek_scheduler(required=True, with_cache=False)
    assert uncached.estimate_cached_prefix_tokens(exact) == 0
    assert not uncached._detect_boundary_snapshot_need.called


# ---------------------------------------------------------------------------
# Resident-KV credit at route time (this commit)
# ---------------------------------------------------------------------------


def _peek_split_scheduler(
    *,
    resident: int = 0,
    ssd_only: int = 0,
    with_cache: bool = True,
) -> Scheduler:
    """Scheduler slice holding only what estimate_cached_prefix_split reads."""
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.block_aware_cache = (
        MagicMock(
            peek_cached_prefix_split=MagicMock(
                return_value=(resident, ssd_only)
            ),
            peek_cached_prefix_tokens=MagicMock(
                return_value=resident + ssd_only
            ),
        )
        if with_cache
        else None
    )
    return scheduler


def test_estimate_cached_prefix_split_returns_raw_split():
    """estimate_cached_prefix_split returns (resident, ssd_only) without
    applying the stateful exact-hit clamp.  The clamp applies to the total
    in estimate_cached_prefix_tokens, not to the split: resident blocks
    are in ``current`` regardless of whether the stateful rule says they
    need re-prefill, and SSD-only blocks still need loading regardless.
    """
    scheduler = _peek_split_scheduler(resident=8, ssd_only=4)
    scheduler._boundary_snapshot_required = True
    scheduler._gdn_split_active = MagicMock(return_value=False)
    scheduler._detect_boundary_snapshot_need = MagicMock(return_value=True)

    # Split is raw — no clamp.
    assert scheduler.estimate_cached_prefix_split(list(range(12))) == (8, 4)

    # Total is clamped: cached=12 >= len=12, stateful, no split -> 0.
    assert scheduler.estimate_cached_prefix_tokens(list(range(12))) == 0


def test_estimate_cached_prefix_split_fail_closed_on_exception():
    """Any exception in the peek must yield (0, 0), mirroring the
    fail-closed contract of estimate_cached_prefix_tokens.
    """
    scheduler = Scheduler.__new__(Scheduler)
    cache = MagicMock()
    cache.peek_cached_prefix_split = MagicMock(side_effect=RuntimeError)
    scheduler.block_aware_cache = cache
    assert scheduler.estimate_cached_prefix_split(list(range(8))) == (0, 0)


def test_estimate_cached_prefix_split_returns_zeros_without_cache():
    """No prefix cache -> (0, 0), same as estimate_cached_prefix_tokens."""
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.block_aware_cache = None
    assert scheduler.estimate_cached_prefix_split(list(range(8))) == (0, 0)


def test_route_admission_credits_resident_kv_and_admits_under_tight_limit():
    """Headline behaviour: a prompt whose prefix is fully memory-resident
    is charged ~0 incremental KV at route time and is ADMITTED under a
    limit that the full-prompt (cold) charge would have failed.

    Resident blocks are already counted in ``current``, so charging them
    again in ``kv_exact`` double-counts.  At route time
    (``cached_kv_resident=False``) the KV charge is
    ``num_prompt_tokens - resident_kv_tokens``; with all blocks resident
    that is 0, leaving only the transient.
    """
    scheduler = _make_scheduler()
    scheduler._prefill_memory_guard = True
    scheduler._memory_abort_limit_bytes = 10**18  # keep safety cap out

    num_prompt = 10_000
    patches = (
        patch("omlx.scheduler.mx.get_active_memory", return_value=0),
        patch("omlx.scheduler.get_phys_footprint", return_value=0),
    )

    with patches[0], patches[1]:
        cold_est = scheduler._admission_estimate(
            num_prompt_tokens=num_prompt,
            cached_tokens=0,
            current=0,
            cached_kv_resident=False,
            resident_kv_tokens=0,
        )
    assert cold_est is not None
    assert cold_est.kv_exact > 0

    with patches[0], patches[1]:
        resident_est = scheduler._admission_estimate(
            num_prompt_tokens=num_prompt,
            cached_tokens=num_prompt,
            current=0,
            cached_kv_resident=False,
            resident_kv_tokens=num_prompt,
        )
    assert resident_est is not None
    assert resident_est.kv_exact == 0, (
        "fully-resident prefix must charge 0 incremental KV"
    )
    assert resident_est.estimated < cold_est.estimated

    # Tight limit: resident fits, cold does not.
    tight_limit = int(resident_est.estimated) + 1
    assert tight_limit < cold_est.estimated
    scheduler._memory_hard_limit_bytes = tight_limit

    # Resident: admitted (no raise).
    with patches[0], patches[1]:
        scheduler.preflight_or_raise(
            num_prompt_tokens=num_prompt,
            cached_tokens=num_prompt,
            resident_kv_tokens=num_prompt,
            request_id="req-resident",
        )

    # Cold: rejected.
    with patches[0], patches[1], pytest.raises(PrefillMemoryExceededError):
        scheduler.preflight_or_raise(
            num_prompt_tokens=num_prompt,
            cached_tokens=0,
            resident_kv_tokens=0,
            request_id="req-cold",
        )


def test_route_admission_charges_ssd_only_kv_and_rejects_under_tight_limit():
    """SSD-only blocks are NOT resident: they must be loaded into newly
    allocated memory, so they are still charged at route time.  Under the
    same tight limit that admits a fully-resident prompt, an SSD-only
    prompt of the same size is REJECTED.  Resident and SSD-only must be
    demonstrably different.
    """
    scheduler = _make_scheduler()
    scheduler._prefill_memory_guard = True
    scheduler._memory_abort_limit_bytes = 10**18

    num_prompt = 10_000
    patches = (
        patch("omlx.scheduler.mx.get_active_memory", return_value=0),
        patch("omlx.scheduler.get_phys_footprint", return_value=0),
    )

    with patches[0], patches[1]:
        resident_est = scheduler._admission_estimate(
            num_prompt_tokens=num_prompt,
            cached_tokens=num_prompt,
            current=0,
            cached_kv_resident=False,
            resident_kv_tokens=num_prompt,
        )
    assert resident_est is not None
    assert resident_est.kv_exact == 0

    with patches[0], patches[1]:
        ssd_est = scheduler._admission_estimate(
            num_prompt_tokens=num_prompt,
            cached_tokens=num_prompt,  # SSD blocks count as cached
            current=0,
            cached_kv_resident=False,
            resident_kv_tokens=0,  # but none are resident
        )
    assert ssd_est is not None
    assert ssd_est.kv_exact > 0, (
        "SSD-only prefix must charge full KV (not resident)"
    )
    assert ssd_est.estimated > resident_est.estimated

    # Tight limit: resident fits, SSD-only does not.
    tight_limit = int(resident_est.estimated) + 1
    assert tight_limit < ssd_est.estimated
    scheduler._memory_hard_limit_bytes = tight_limit

    # Resident: admitted.
    with patches[0], patches[1]:
        scheduler.preflight_or_raise(
            num_prompt_tokens=num_prompt,
            cached_tokens=num_prompt,
            resident_kv_tokens=num_prompt,
            request_id="req-resident",
        )

    # SSD-only: rejected.
    with patches[0], patches[1], pytest.raises(PrefillMemoryExceededError):
        scheduler.preflight_or_raise(
            num_prompt_tokens=num_prompt,
            cached_tokens=num_prompt,
            resident_kv_tokens=0,
            request_id="req-ssd-only",
        )


def test_route_admission_mixed_prefix_charges_only_non_resident():
    """A mixed prefix (some resident, some SSD-only) charges only the
    non-resident part.  The KV charge is ``num_prompt_tokens -
    resident_kv_tokens``, which includes SSD-only and uncached tokens but
    not resident tokens.
    """
    scheduler = _make_scheduler()
    scheduler._prefill_memory_guard = True
    scheduler._memory_abort_limit_bytes = 10**18

    num_prompt = 10_000
    resident = 6_000
    ssd_only = 3_000
    cached_total = resident + ssd_only  # 9_000 cached, 1_000 new

    patches = (
        patch("omlx.scheduler.mx.get_active_memory", return_value=0),
        patch("omlx.scheduler.get_phys_footprint", return_value=0),
    )

    with patches[0], patches[1]:
        mixed_est = scheduler._admission_estimate(
            num_prompt_tokens=num_prompt,
            cached_tokens=cached_total,
            current=0,
            cached_kv_resident=False,
            resident_kv_tokens=resident,
        )
    assert mixed_est is not None

    # charge_kv_tokens = num_prompt - resident = 4000
    # (3000 SSD-only + 1000 new)
    with patches[0], patches[1]:
        spy = MagicMock(
            wraps=scheduler.memory_monitor.estimate_resident_kv_bytes
        )
        scheduler.memory_monitor.estimate_resident_kv_bytes = spy
        scheduler._admission_estimate(
            num_prompt_tokens=num_prompt,
            cached_tokens=cached_total,
            current=0,
            cached_kv_resident=False,
            resident_kv_tokens=resident,
        )
        charge_kv = spy.call_args.args[0]

    assert charge_kv == num_prompt - resident  # 4000

    # Compare against fully-resident and fully-cold estimates.
    with patches[0], patches[1]:
        full_resident_est = scheduler._admission_estimate(
            num_prompt_tokens=num_prompt,
            cached_tokens=num_prompt,
            current=0,
            cached_kv_resident=False,
            resident_kv_tokens=num_prompt,
        )
    with patches[0], patches[1]:
        cold_est = scheduler._admission_estimate(
            num_prompt_tokens=num_prompt,
            cached_tokens=0,
            current=0,
            cached_kv_resident=False,
            resident_kv_tokens=0,
        )

    assert full_resident_est.kv_exact == 0
    assert cold_est.kv_exact > mixed_est.kv_exact > 0
    assert (
        full_resident_est.estimated
        < mixed_est.estimated
        < cold_est.estimated
    )


def test_route_admission_resident_credit_through_preflight_or_raise():
    """Drive the resident credit through ``preflight_or_raise`` (the
    route-time caller), not only via ``_admission_estimate`` directly.
    A prior agent shipped a mutation that did not bite precisely because
    its test called ``_admission_estimate`` directly and never exercised
    the caller's kwarg.
    """
    scheduler = _make_scheduler()
    scheduler._prefill_memory_guard = True
    scheduler._memory_abort_limit_bytes = 10**18

    num_prompt = 10_000
    patches = (
        patch("omlx.scheduler.mx.get_active_memory", return_value=0),
        patch("omlx.scheduler.get_phys_footprint", return_value=0),
    )

    # Find the tight limit from the estimates.
    with patches[0], patches[1]:
        resident_est = scheduler._admission_estimate(
            num_prompt_tokens=num_prompt,
            cached_tokens=num_prompt,
            current=0,
            cached_kv_resident=False,
            resident_kv_tokens=num_prompt,
        )
    with patches[0], patches[1]:
        cold_est = scheduler._admission_estimate(
            num_prompt_tokens=num_prompt,
            cached_tokens=0,
            current=0,
            cached_kv_resident=False,
            resident_kv_tokens=0,
        )
    assert resident_est.kv_exact == 0
    assert cold_est.kv_exact > 0

    tight_limit = int(resident_est.estimated) + 1
    scheduler._memory_hard_limit_bytes = tight_limit

    # Through preflight_or_raise: resident admitted, cold rejected.
    with patches[0], patches[1]:
        scheduler.preflight_or_raise(
            num_prompt_tokens=num_prompt,
            cached_tokens=num_prompt,
            resident_kv_tokens=num_prompt,
            request_id="req-resident-route",
        )

    with patches[0], patches[1], pytest.raises(PrefillMemoryExceededError):
        scheduler.preflight_or_raise(
            num_prompt_tokens=num_prompt,
            cached_tokens=0,
            resident_kv_tokens=0,
            request_id="req-cold-route",
        )


def test_in_stream_path_ignores_resident_kv_tokens():
    """The in-stream path (cached_kv_resident=True) must NOT change
    behaviour when resident_kv_tokens is passed: it charges new_tokens
    because _prepare_prefix_cache_for_request has already loaded the hit.
    """
    scheduler = _make_scheduler()
    scheduler._prefill_memory_guard = True
    scheduler._memory_hard_limit_bytes = 10**18

    patches = (
        patch("omlx.scheduler.mx.get_active_memory", return_value=0),
        patch("omlx.scheduler.get_phys_footprint", return_value=0),
    )

    with patches[0], patches[1]:
        est_default = scheduler._admission_estimate(
            num_prompt_tokens=1000,
            cached_tokens=900,
            current=0,
        )
    with patches[0], patches[1]:
        est_with_resident = scheduler._admission_estimate(
            num_prompt_tokens=1000,
            cached_tokens=900,
            current=0,
            resident_kv_tokens=900,
        )

    assert est_default is not None
    assert est_with_resident is not None
    # In-stream path: resident_kv_tokens is ignored, both charge new_tokens.
    assert est_default.kv_exact == est_with_resident.kv_exact
    assert est_default.estimated == est_with_resident.estimated


# ---------------------------------------------------------------------------
# Engine-chain tests (F1: verify resident_kv_tokens reaches admission through
# the real engine preflight path, not just by calling _admission_estimate).
# The prior commit shipped inert because no test drove the engine chain.
# ---------------------------------------------------------------------------

from omlx.cache.paged_cache import PagedCacheManager, compute_block_hash


def _make_scheduler_with_cache(block_size: int = 4):
    """Scheduler with a real PagedCacheManager as block_aware_cache.

    The scheduler from ``_make_scheduler`` has a real memory_monitor but no
    paged cache (block_size=0).  We attach a real PagedCacheManager so
    ``estimate_cached_prefix_for_admission`` does a genuine walk and returns
    non-zero resident_kv_tokens when blocks are inserted.
    """
    scheduler = _make_scheduler()
    scheduler._prefill_memory_guard = True
    scheduler._memory_abort_limit_bytes = 10**18
    scheduler._memory_hard_watermark_bytes = 0
    manager = PagedCacheManager(
        block_size=block_size, max_blocks=200,
        model_name="test-model", initial_blocks=200,
    )
    scheduler.block_aware_cache = manager
    scheduler.config.paged_cache_block_size = block_size
    # Non-stateful: the stateful clamp is tested separately.
    scheduler._boundary_snapshot_required = False
    scheduler._gdn_split_active = MagicMock(return_value=False)
    scheduler._detect_boundary_snapshot_need = MagicMock(return_value=True)
    return scheduler, manager


def _insert_resident_blocks(manager, token_ids, block_size):
    """Insert chained blocks covering token_ids into the paged cache hash map."""
    parent_hash = None
    num_full = len(token_ids) // block_size
    for i in range(num_full):
        start = i * block_size
        end = start + block_size
        block_hash = compute_block_hash(
            parent_hash, token_ids[start:end], model_name="test-model"
        )
        block = manager.allocate_block()
        block.block_hash = block_hash
        block.token_count = block_size
        manager.cached_block_hash_to_block.insert(block_hash, block)
        parent_hash = block_hash
    return parent_hash


def _compute_block_hashes(token_ids, block_size):
    """Compute the chained block hashes for token_ids without inserting."""
    parent_hash = None
    hashes = []
    num_full = len(token_ids) // block_size
    for i in range(num_full):
        start = i * block_size
        end = start + block_size
        block_hash = compute_block_hash(
            parent_hash, token_ids[start:end], model_name="test-model"
        )
        hashes.append(block_hash)
        parent_hash = block_hash
    return hashes


def _make_batched_engine(scheduler, token_ids):
    """Create a BatchedEngine slice wired to scheduler for preflight_chat.

    Uses ``__new__`` to bypass the full constructor; sets only the attributes
    that ``preflight_chat`` reads.  The tokenizer returns ``token_ids`` for
    any prompt, and the chat template is stubbed to a dummy string.
    """
    from omlx.engine.batched import BatchedEngine

    engine = BatchedEngine.__new__(BatchedEngine)
    engine._loaded = True
    engine._model_name = "test-model"
    engine._tokenizer = MagicMock()
    engine._tokenizer.encode = MagicMock(return_value=list(token_ids))
    engine._prefill_eviction_callback = None
    engine._engine = MagicMock()
    engine._engine.engine.scheduler = scheduler
    engine._preprocess_messages = MagicMock(side_effect=lambda m: m)
    engine._apply_chat_template = MagicMock(return_value="dummy")
    return engine


_PATCHES = (
    patch("omlx.scheduler.mx.get_active_memory", return_value=0),
    patch("omlx.scheduler.get_phys_footprint", return_value=0),
)


async def test_engine_preflight_chat_forwards_resident_kv_to_admission():
    """F1 regression: ``BatchedEngine.preflight_chat`` must forward
    ``resident_kv_tokens`` from the combined probe all the way to
    ``_admission_estimate``.  The prior commit shipped inert because every
    engine caller passed only ``cached_tokens`` and let the parameter
    default to 0.  This test drives the real engine path and asserts the
    credit arrives non-zero.
    """
    block_size = 4
    scheduler, manager = _make_scheduler_with_cache(block_size)
    scheduler._memory_hard_limit_bytes = 10**18  # admit everything

    # 20 blocks = 80 tokens, all resident.
    token_ids = list(range(1, 81))
    _insert_resident_blocks(manager, token_ids, block_size)

    engine = _make_batched_engine(scheduler, token_ids)

    # Spy on _admission_estimate to capture the kwargs.
    original = scheduler._admission_estimate
    captured = {}

    def spy(**kwargs):
        captured.update(kwargs)
        return original(**kwargs)

    scheduler._admission_estimate = spy
    try:
        with _PATCHES[0], _PATCHES[1]:
            await engine.preflight_chat(
                [{"role": "user", "content": "hello"}],
                request_id="req-engine",
            )
    finally:
        scheduler._admission_estimate = original

    assert captured.get("cached_kv_resident") is False, (
        "engine path must use the route-time path (cached_kv_resident=False)"
    )
    assert captured.get("resident_kv_tokens", 0) > 0, (
        "resident_kv_tokens must arrive non-zero at _admission_estimate "
        "through the engine chain — this is the F1 regression test"
    )
    assert captured.get("resident_kv_tokens") == 80, (
        f"expected 80 resident tokens, got {captured.get('resident_kv_tokens')}"
    )
    assert captured.get("cached_tokens") == 80


async def test_engine_preflight_resident_admitted_cold_rejected_under_tight_limit():
    """Through the engine path (not by calling the scheduler directly):
    a resident prefix is ADMITTED under a tight limit that rejects the
    same-size cold prompt.
    """
    block_size = 4
    scheduler, manager = _make_scheduler_with_cache(block_size)

    token_ids = list(range(1, 81))  # 80 tokens, 20 blocks
    _insert_resident_blocks(manager, token_ids, block_size)

    # Find the tight limit from the estimates.
    with _PATCHES[0], _PATCHES[1]:
        resident_est = scheduler._admission_estimate(
            num_prompt_tokens=80, cached_tokens=80, current=0,
            cached_kv_resident=False, resident_kv_tokens=80,
        )
        cold_est = scheduler._admission_estimate(
            num_prompt_tokens=80, cached_tokens=0, current=0,
            cached_kv_resident=False, resident_kv_tokens=0,
        )
    assert resident_est is not None and cold_est is not None
    assert resident_est.kv_exact == 0
    assert cold_est.kv_exact > 0

    tight_limit = int(resident_est.estimated) + 1
    assert tight_limit < cold_est.estimated
    scheduler._memory_hard_limit_bytes = tight_limit

    engine = _make_batched_engine(scheduler, token_ids)

    # Resident: admitted through the engine path.
    with _PATCHES[0], _PATCHES[1]:
        await engine.preflight_chat(
            [{"role": "user", "content": "hello"}],
            request_id="req-resident-engine",
        )

    # Cold: same engine, no cache -> rejected.
    cold_manager = PagedCacheManager(
        block_size=block_size, max_blocks=200,
        model_name="test-model", initial_blocks=200,
    )
    scheduler.block_aware_cache = cold_manager
    engine_cold = _make_batched_engine(scheduler, token_ids)
    with _PATCHES[0], _PATCHES[1], pytest.raises(PrefillMemoryExceededError):
        await engine_cold.preflight_chat(
            [{"role": "user", "content": "hello"}],
            request_id="req-cold-engine",
        )


async def test_engine_preflight_ssd_only_rejected_under_same_limit():
    """An SSD-only prefix (blocks on SSD but not in memory) is still
    CHARGED full KV at route time and REJECTED under the same tight limit
    that admits a resident prefix.  This goes through the engine path.
    """
    block_size = 4
    scheduler, manager = _make_scheduler_with_cache(block_size)

    token_ids = list(range(1, 81))  # 80 tokens, 20 blocks

    # Insert NO blocks in memory.  Instead, mock the SSD manager to report
    # all blocks as present on SSD.
    hashes = _compute_block_hashes(token_ids, block_size)
    hash_set = set(hashes)
    mock_ssd = MagicMock()
    mock_ssd.has_block = MagicMock(side_effect=lambda h: h in hash_set)
    manager._paged_ssd_cache_manager = mock_ssd

    # First, find the tight limit using a resident setup.
    resident_manager = PagedCacheManager(
        block_size=block_size, max_blocks=200,
        model_name="test-model", initial_blocks=200,
    )
    _insert_resident_blocks(resident_manager, token_ids, block_size)
    scheduler.block_aware_cache = resident_manager
    with _PATCHES[0], _PATCHES[1]:
        resident_est = scheduler._admission_estimate(
            num_prompt_tokens=80, cached_tokens=80, current=0,
            cached_kv_resident=False, resident_kv_tokens=80,
        )
    assert resident_est is not None and resident_est.kv_exact == 0
    tight_limit = int(resident_est.estimated) + 1
    scheduler._memory_hard_limit_bytes = tight_limit

    # Resident: admitted.
    engine_resident = _make_batched_engine(scheduler, token_ids)
    with _PATCHES[0], _PATCHES[1]:
        await engine_resident.preflight_chat(
            [{"role": "user", "content": "hello"}],
            request_id="req-resident-engine",
        )

    # SSD-only: switch to the SSD-only cache and reject.
    scheduler.block_aware_cache = manager
    engine_ssd = _make_batched_engine(scheduler, token_ids)
    with _PATCHES[0], _PATCHES[1], pytest.raises(PrefillMemoryExceededError):
        await engine_ssd.preflight_chat(
            [{"role": "user", "content": "hello"}],
            request_id="req-ssd-engine",
        )


def test_admission_estimate_raises_on_over_credit():
    """F4: ``resident_kv_tokens`` exceeding ``cached_tokens`` or
    ``num_prompt_tokens`` raises ``ValueError`` — misuse is loud, not
    silently clamped.  This prevents a large value from zeroing
    ``charge_kv_tokens`` and bypassing the memory guard.
    """
    scheduler = _make_scheduler()
    scheduler._prefill_memory_guard = True
    scheduler._memory_hard_limit_bytes = 10**18

    with _PATCHES[0], _PATCHES[1]:
        # resident_kv_tokens > cached_tokens
        with pytest.raises(ValueError, match="out of range"):
            scheduler._admission_estimate(
                num_prompt_tokens=1000, cached_tokens=500, current=0,
                cached_kv_resident=False, resident_kv_tokens=900,
            )
        # resident_kv_tokens > num_prompt_tokens
        with pytest.raises(ValueError, match="out of range"):
            scheduler._admission_estimate(
                num_prompt_tokens=1000, cached_tokens=2000, current=0,
                cached_kv_resident=False, resident_kv_tokens=1500,
            )
        # negative
        with pytest.raises(ValueError, match="out of range"):
            scheduler._admission_estimate(
                num_prompt_tokens=1000, cached_tokens=500, current=0,
                cached_kv_resident=False, resident_kv_tokens=-1,
            )


def test_admission_estimate_raises_on_over_credit_even_without_monitor():
    """F4 hoist pin: the ``resident_kv_tokens`` range validation sits ABOVE
    the ``memory_monitor is None`` early return, so the invariant holds
    whether or not a monitor exists.  Without the hoist, a missing monitor
    lets an out-of-range credit return None (fail-open) instead of raising.

    The existing ``test_admission_estimate_raises_on_over_credit`` builds its
    scheduler via ``_make_scheduler()``, which ALWAYS supplies a monitor, so
    it passes identically with and without the hoist — it does not pin the
    ordering.  This test pins BOTH halves of the hoist:

    - out-of-range credit with ``memory_monitor = None`` MUST raise
      ``ValueError`` (not return None)
    - in-range credit with ``memory_monitor = None`` MUST still return None
      (the monitor-absent early return is not accidentally deleted)

    Mutation that must go RED: move the ``monitor = self.memory_monitor`` /
    ``if monitor is None: return None`` block back above the F4 validation.
    The out-of-range half will return None instead of raising.
    """
    scheduler = _make_scheduler()
    scheduler._prefill_memory_guard = True
    scheduler._memory_hard_limit_bytes = 10**18
    scheduler.memory_monitor = None

    # Out-of-range with no monitor: MUST raise, not return None.
    # resident_kv_tokens=900 > cached_tokens=500 → out of range.
    with pytest.raises(ValueError, match="out of range"):
        scheduler._admission_estimate(
            num_prompt_tokens=1000, cached_tokens=500, current=0,
            cached_kv_resident=False, resident_kv_tokens=900,
        )

    # In-range with no monitor: MUST still return None — the monitor-absent
    # early return must not be deleted by the hoist.
    est = scheduler._admission_estimate(
        num_prompt_tokens=1000, cached_tokens=500, current=0,
        cached_kv_resident=False, resident_kv_tokens=100,
    )
    assert est is None, (
        "in-range credit with no monitor must return None — the "
        "monitor-absent early return must not be deleted"
    )


def test_combined_probe_stateful_full_hit_not_underpriced():
    """F3/M4: the combined probe applies the stateful exact-hit clamp
    internally.  A stateful full-cache hit returns ``cached_tokens=0``
    and ``resident_kv_tokens=0``, so ``_admission_estimate`` charges the
    FULL prefill (KV + transient) — NOT the 1-token floored prefill that
    an unclamped total would produce.
    """
    block_size = 4
    scheduler, manager = _make_scheduler_with_cache(block_size)
    # Stateful: the exact-hit clamp triggers.
    scheduler._boundary_snapshot_required = True

    token_ids = list(range(1, 81))  # 80 tokens = 20 blocks, all resident
    _insert_resident_blocks(manager, token_ids, block_size)

    estimate = scheduler.estimate_cached_prefix_for_admission(token_ids)
    # Stateful full hit: clamp reduces cached to 0, resident clamped to 0.
    assert estimate.cached_tokens == 0, (
        "stateful full hit must clamp cached_tokens to 0"
    )
    assert estimate.resident_kv_tokens == 0, (
        "resident_kv_tokens must be clamped to cached_tokens=0"
    )

    # Through _admission_estimate: charges full KV + full transient.
    with _PATCHES[0], _PATCHES[1]:
        est = scheduler._admission_estimate(
            num_prompt_tokens=80,
            cached_tokens=estimate.cached_tokens,
            current=0,
            cached_kv_resident=False,
            resident_kv_tokens=estimate.resident_kv_tokens,
        )
    assert est is not None
    assert est.kv_exact > 0, (
        "stateful full hit must charge full KV (not under-priced)"
    )

    # Compare against the non-stateful full hit (under-priced if unclamped).
    scheduler._boundary_snapshot_required = False
    estimate_ns = scheduler.estimate_cached_prefix_for_admission(token_ids)
    assert estimate_ns.cached_tokens == 80
    assert estimate_ns.resident_kv_tokens == 80
    with _PATCHES[0], _PATCHES[1]:
        est_ns = scheduler._admission_estimate(
            num_prompt_tokens=80,
            cached_tokens=estimate_ns.cached_tokens,
            current=0,
            cached_kv_resident=False,
            resident_kv_tokens=estimate_ns.resident_kv_tokens,
        )
    assert est_ns is not None
    assert est_ns.kv_exact == 0, "non-stateful full hit charges 0 KV"
    assert est.estimated > est_ns.estimated, (
        "stateful full hit must be MORE expensive than non-stateful "
        "(not under-priced)"
    )


def test_combined_probe_returns_consistent_pair():
    """F6: the combined probe returns both values from one walk, so they
    are always consistent (``0 <= resident_kv_tokens <= cached_tokens``).
    """
    block_size = 4
    scheduler, manager = _make_scheduler_with_cache(block_size)

    # Partial resident hit: 10 blocks resident, 5 SSD-only.
    token_ids = list(range(1, 81))  # 20 blocks
    _insert_resident_blocks(manager, token_ids[:40], block_size)  # 10 blocks

    # SSD-only blocks 10..14
    hashes = _compute_block_hashes(token_ids, block_size)
    mock_ssd = MagicMock()
    ssd_hashes = set(hashes[10:15])
    mock_ssd.has_block = MagicMock(side_effect=lambda h: h in ssd_hashes)
    manager._paged_ssd_cache_manager = mock_ssd

    estimate = scheduler.estimate_cached_prefix_for_admission(token_ids)
    assert estimate.cached_tokens == 60, (
        f"expected 60 cached (40 resident + 20 ssd), got {estimate.cached_tokens}"
    )
    assert estimate.resident_kv_tokens == 40, (
        f"expected 40 resident, got {estimate.resident_kv_tokens}"
    )
    assert 0 <= estimate.resident_kv_tokens <= estimate.cached_tokens
