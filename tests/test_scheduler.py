from tidemark.catalog import FrontierKey, VersionedFrontierCatalog
from tidemark.catalog.history import SessionHistory
from tidemark.scheduler import (
    DestinationPredictor,
    GlobalFrontierScheduler,
    SchedulerConfig,
    StaticRouterSignal,
    TicketResult,
    TicketStatus,
)
from tidemark.scheduler.global_scheduler import EngineSpec
from tidemark.scheduler.ranking import TenantCaps

ENGINES = [
    EngineSpec("device-0", "device-1b", "cfg", "device"),
    EngineSpec("edge-0", "edge-7b", "cfg", "edge"),
    EngineSpec("cloud-0", "cloud-14b", "cfg", "cloud"),
]


def _scheduler(rates, router=None, **cfg):
    # A single tenant in the tests: let it use the whole background budget so
    # the per-tenant share does not hide what each test is about.
    cfg.setdefault("tenant_caps", TenantCaps(kappa=2, beta=1.0))
    issued = []
    cat = VersionedFrontierCatalog()
    pred = DestinationPredictor(models=[e.model_id for e in ENGINES], router=router, alpha=1.0 if router else 0.0)
    sched = GlobalFrontierScheduler(cat, rates, pred, ENGINES, SchedulerConfig(**cfg), sink=issued.append)
    return sched, cat, issued


def _long_history(tokenizers, sid="s1", tenant="t1", words=600):
    h = SessionHistory(sid, tenant, tokenizers)
    h.append("user", " ".join(f"w{i}" for i in range(words)))
    return h


def test_turn_marks_others_lagging_and_issues_one_ticket_per_engine(tokenizers, rates):
    sched, cat, issued = _scheduler(rates)
    h = _long_history(tokenizers)
    sched.on_session_start(h)
    sched.on_turn_served("s1", served_model="edge-7b", runtime_config="cfg", resident_prefix=h.length("edge-7b"))
    engines = sorted(t.engine_id for t in issued)
    assert engines == ["cloud-0", "device-0"]
    assert all(t.delta_max <= 1024 for t in issued)
    assert cat.get(FrontierKey("s1", "edge-7b", "cfg")).frontier == h.length("edge-7b")


def test_commit_advances_and_reissues(tokenizers, rates):
    sched, cat, issued = _scheduler(rates)
    h = _long_history(tokenizers)
    sched.on_session_start(h)
    sched.on_turn_served("s1", served_model="device-1b", runtime_config="cfg", resident_prefix=h.length("device-1b"))
    t = next(x for x in issued if x.engine_id == "cloud-0")
    total = h.length("cloud-14b")
    res = TicketResult(t.ticket_id, "cloud-0", TicketStatus.COMMITTED, admitted_delta=t.delta_max, snapshot_hash=t.prefix_hash(t.delta_max), gpu_ms=t.delta_max * 0.073)
    sched.on_ticket_result(res)
    e = cat.get(t.key)
    assert e.frontier == t.delta_max
    assert sched.stats.committed == 1
    # The re-rank after the commit continues the frontier where it left off.
    nxt = [x for x in issued if x.engine_id == "cloud-0"][-1]
    assert nxt.frontier == t.delta_max
    assert nxt.frontier + nxt.delta_max <= total


def test_stale_completion_after_foreground_is_dropped(tokenizers, rates):
    sched, cat, issued = _scheduler(rates)
    h = _long_history(tokenizers)
    sched.on_session_start(h)
    sched.on_turn_served("s1", served_model="device-1b", runtime_config="cfg", resident_prefix=h.length("device-1b"))
    t = next(x for x in issued if x.engine_id == "cloud-0")
    # Meanwhile the session actually switches to the cloud; the foreground request advances the frontier.
    sched.on_turn_served("s1", served_model="cloud-14b", runtime_config="cfg", resident_prefix=h.length("cloud-14b"))
    sched.on_ticket_result(TicketResult(t.ticket_id, "cloud-0", TicketStatus.COMMITTED, admitted_delta=t.delta_max, snapshot_hash=t.prefix_hash(t.delta_max)))
    assert sched.stats.stale == 1
    assert cat.get(t.key).frontier == h.length("cloud-14b")


def test_router_signal_steers_ranking(tokenizers, rates):
    router = StaticRouterSignal({"s1": {"device-1b": 0.05, "edge-7b": 0.05, "cloud-14b": 0.9}, "s2": {"device-1b": 0.05, "edge-7b": 0.9, "cloud-14b": 0.05}})
    sched, cat, issued = _scheduler(rates, router=router, tenant_caps=TenantCaps(kappa=1, beta=1.0))
    h1, h2 = _long_history(tokenizers, "s1", "t1"), _long_history(tokenizers, "s2", "t1")
    sched.on_session_start(h1)
    sched.on_session_start(h2)
    sched.on_turn_served("s1", served_model="device-1b", runtime_config="cfg", resident_prefix=h1.length("device-1b"))
    sched.on_turn_served("s2", served_model="device-1b", runtime_config="cfg", resident_prefix=h2.length("device-1b"))
    # Tenant t1 may hold one ticket: the scheduler should spend it on the likeliest destination.
    live = sched.inflight()
    assert len(live) == 1
    t = live[0]
    assert (t.session_id, t.model_id) in {("s1", "cloud-14b"), ("s2", "edge-7b")}


def test_revision_transition_cancels_inflight(tokenizers, rates):
    sched, cat, issued = _scheduler(rates)
    h = _long_history(tokenizers)
    sched.on_session_start(h)
    sched.on_turn_served("s1", served_model="edge-7b", runtime_config="cfg", resident_prefix=h.length("edge-7b"))
    assert sched.inflight()
    lcp = h.rewrite(0, "brand new prompt")
    sched.on_revision_transition("s1", lcp)
    assert not sched.inflight()


def test_lease_expiry_frees_engine(tokenizers, rates):
    sched, cat, issued = _scheduler(rates, ticket_lease_s=0.0)
    h = _long_history(tokenizers)
    sched.on_session_start(h)
    sched.on_turn_served("s1", served_model="edge-7b", runtime_config="cfg", resident_prefix=h.length("edge-7b"))
    n = sched.expire_leases(now=1e12)
    assert n == 2 and not sched.inflight()
