import math

import pytest

from tidemark.catalog.frontier import FrontierKey
from tidemark.scheduler.ranking import (
    Candidate,
    RankingEpoch,
    TenantCaps,
    TenantLedger,
    expected_benefit_ms,
    resource_cost_ms,
    score,
)


def _cand(name, tenant, engine, benefit, cost):
    return Candidate(
        key=FrontierKey(name, engine, "cfg"),
        tenant_id=tenant,
        engine_id=engine,
        frontier=0,
        lag=1000,
        delta_max=512,
        p_future=0.7,
        benefit_ms=benefit,
        cost_ms=cost,
        score=score(benefit, cost),
    )


def test_score_is_per_token_marginal_value(rates):
    r = rates.get("cloud-0")
    for delta in (256, 512, 1024):
        b = expected_benefit_ms(r, 0.7, lag=4000, delta=delta)
        c = resource_cost_ms(r, delta, lambda_mem_ms_per_gib=64.0)
        assert math.isclose(score(b, c), score(expected_benefit_ms(r, 0.7, 4000, 256), resource_cost_ms(r, 256, 64.0)), rel_tol=1e-9)


def test_benefit_saturates_at_the_lag(rates):
    r = rates.get("edge-0")
    assert expected_benefit_ms(r, 1.0, lag=100, delta=1024) == pytest.approx(r.critical_path_ms(100))


def test_tiers_are_not_commensurable_in_raw_tokens(rates):
    """A device token costs far more to prepare than a cloud token, which is why
    the ranking divides by tau_bg-weighted compute time rather than tokens."""
    dev = rates.get("device-0")
    cloud = rates.get("cloud-0")
    same_p, lag, d = 0.5, 2000, 512
    s_dev = score(expected_benefit_ms(dev, same_p, lag, d), resource_cost_ms(dev, d, 64.0))
    s_cloud = score(expected_benefit_ms(cloud, same_p, lag, d), resource_cost_ms(cloud, d, 64.0))
    # Both remove latency in proportion to their own tau_fg; the cloud's much
    # lower tau_bg/tau_fg makes each unit of background time buy more.
    assert s_cloud > s_dev


def test_one_ticket_per_engine():
    ledger = TenantLedger(TenantCaps(kappa=4, beta=1.0))
    epoch = RankingEpoch(ledger)
    cands = [_cand("a", "t1", "cloud-0", 100, 1), _cand("b", "t2", "cloud-0", 90, 1), _cand("c", "t3", "edge-0", 50, 1)]
    out = epoch.run(cands, engines_busy=set(), g_total_ms=1e9)
    assert [c.key.session_id for c in out.issued] == ["a", "c"]
    assert out.skipped[FrontierKey("b", "cloud-0", "cfg")] == "engine_inflight"


def test_tenant_ticket_cap_binds_across_tiers():
    ledger = TenantLedger(TenantCaps(kappa=1, beta=1.0))
    epoch = RankingEpoch(ledger)
    cands = [_cand("a", "t1", "cloud-0", 100, 1), _cand("b", "t1", "edge-0", 90, 1), _cand("c", "t2", "device-0", 10, 1)]
    out = epoch.run(cands, engines_busy=set(), g_total_ms=1e9)
    assert [c.key.session_id for c in out.issued] == ["a", "c"]
    assert out.skipped[FrontierKey("b", "edge-0", "cfg")] == "tenant_ticket_cap"


def test_tenant_budget_share_binds():
    ledger = TenantLedger(TenantCaps(kappa=8, beta=0.35))
    epoch = RankingEpoch(ledger)
    cands = [_cand("a", "t1", "cloud-0", 100, 40), _cand("b", "t1", "edge-0", 90, 40), _cand("c", "t2", "device-0", 10, 10)]
    out = epoch.run(cands, engines_busy=set(), g_total_ms=100.0)  # t1 may spend 35 ms
    assert [c.key.session_id for c in out.issued] == ["c"]
    assert out.skipped[FrontierKey("a", "cloud-0", "cfg")] == "tenant_budget_share"


def test_aging_lifts_repeatedly_skipped_tenant():
    ledger = TenantLedger(TenantCaps(kappa=2, beta=1.0), aging_step=0.5)
    epoch = RankingEpoch(ledger)
    # t1 always has a slightly better candidate on the only engine.
    for _ in range(3):
        out = epoch.run([_cand("a", "t1", "cloud-0", 100, 1), _cand("b", "t2", "cloud-0", 95, 1)], engines_busy=set(), g_total_ms=1e9)
        ledger.settle("t1", 1.0)  # ticket retires before the next epoch
    assert ledger.aging["t2"] > 0
    out = epoch.run([_cand("a", "t1", "cloud-0", 100, 1), _cand("b", "t2", "cloud-0", 95, 1)], engines_busy=set(), g_total_ms=1e9)
    assert out.issued[0].tenant_id == "t2"
