from tidemark.admission import (
    INTERVAL_SET,
    AdmissionMode,
    EngineLocalAdmission,
    SafeBudgetInputs,
    StepState,
    TpotGuard,
    classify_mode,
    safe_budget,
)


def _inputs(**kw):
    base = dict(token_budget=2048, decode_tokens=0, prefill_tokens=0, kv_free_blocks=1000, kv_total_blocks=1000)
    base.update(kw)
    return SafeBudgetInputs(**base)


def test_safe_budget_is_residual_capped_by_xmax():
    assert safe_budget(_inputs(decode_tokens=64, prefill_tokens=512)) == 1024
    assert safe_budget(_inputs(decode_tokens=64, prefill_tokens=1500)) == 2048 - 64 - 1500
    assert safe_budget(_inputs(decode_tokens=2048)) == 0


def test_safe_budget_respects_kv_headroom():
    # 8 % headroom of 1000 blocks = 80 blocks; only 90 free -> 10 blocks * 16 tokens.
    assert safe_budget(_inputs(kv_free_blocks=90)) == 160
    assert safe_budget(_inputs(kv_free_blocks=80)) == 0


def test_mode_classification():
    assert classify_mode(decode_tokens=0, prefill_tokens=0, safe_budget_tokens=0, guard_ok=False) is AdmissionMode.IDLE
    assert classify_mode(decode_tokens=32, prefill_tokens=0, safe_budget_tokens=512, guard_ok=True) is AdmissionMode.MIXED
    assert classify_mode(decode_tokens=32, prefill_tokens=0, safe_budget_tokens=512, guard_ok=False) is AdmissionMode.BLOCKED
    assert classify_mode(decode_tokens=32, prefill_tokens=0, safe_budget_tokens=0, guard_ok=True) is AdmissionMode.BLOCKED


def test_guard_calibrates_then_trips_at_gamma():
    g = TpotGuard(gamma=0.03, ewma_alpha=1.0, calibration_steps=5)
    assert not g.ok  # conservative before calibration
    for _ in range(5):
        g.observe(10.0)
    assert g.calibrated and g.tpot_ref_ms == 10.0
    g.observe(10.2)
    assert g.ok
    g.observe(10.4)  # > 10.3 threshold
    assert not g.ok


def _step(step_id, decode=0, prefill=0, arrivals=0, free=900, tpot=None):
    return StepState(step_id=step_id, token_budget=2048, decode_tokens=decode, prefill_tokens=prefill, kv_free_blocks=free, kv_total_blocks=1000, new_foreground_arrivals=arrivals, last_tpot_ms=tpot)


def _controller():
    g = TpotGuard(calibration_steps=1)
    g.calibrate([20.0])
    return EngineLocalAdmission(guard=g)


def test_idle_admits_largest_fitting_interval():
    c = _controller()
    d = c.decide(_step(1), ticket_id="t", delta_max=3000)
    assert d.mode is AdmissionMode.IDLE and d.admitted_delta == 1024


def test_mixed_admits_within_safe_budget():
    c = _controller()
    d = c.decide(_step(1, decode=64, prefill=1500, tpot=20.0), ticket_id="t", delta_max=3000)
    assert d.mode is AdmissionMode.MIXED
    assert d.admitted_delta == 256  # X_t = 484 -> largest interval <= 484


def test_blocked_when_guard_trips():
    c = _controller()
    d = c.decide(_step(1, decode=64, tpot=25.0), ticket_id="t", delta_max=3000)
    assert d.mode is AdmissionMode.BLOCKED and d.admitted_delta == 0 and d.reason == "guard"


def test_one_interval_per_iteration_then_cancel_on_arrival():
    c = _controller()
    d1 = c.decide(_step(1), ticket_id="t", delta_max=3000)
    assert d1.admitted
    d2 = c.decide(_step(1), ticket_id="u", delta_max=3000)
    assert not d2.admitted and d2.reason == "interval_inflight"
    assert c.should_cancel(_step(2, arrivals=1)) == "t"
    assert c.inflight_ticket is None


def test_short_tail_is_admitted_whole():
    c = _controller()
    d = c.decide(_step(1), ticket_id="t", delta_max=90)
    assert d.admitted_delta == 90


def test_interval_set_is_the_paper_default():
    assert INTERVAL_SET == (256, 512, 1024)
