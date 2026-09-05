from tidemark.admission.commit import CommitPath
from tidemark.admission.controller import EngineLocalAdmission
from tidemark.admission.guard import TpotGuard
from tidemark.catalog.history import HistorySnapshot
from tidemark.engines.vllm.prefill_only import PrefillOnlyRequest
from tidemark.engines.vllm.shim import SchedulerStepStats, VllmAdmissionShim
from tidemark.scheduler.ticket import AtomicTicket, TicketStatus


def _ticket(delta_max=1024, n=3000):
    snap = HistorySnapshot("s", "m", 0, tuple(range(n)))
    return AtomicTicket("tk1", "t", "s", "m", "cfg", "cloud-0", 0, 0, 0, delta_max, snap, 1.0, 10.0, 1.0, 0.7, 1)


def _shim(results):
    g = TpotGuard(calibration_steps=1)
    g.calibrate([9.0])
    return VllmAdmissionShim("cloud-0", CommitPath("cloud-0", results.append), EngineLocalAdmission(guard=g))


def _stats(step, decode=0, prefill=0, arrivals=0, tpot=None):
    return SchedulerStepStats(step, 4096, decode, prefill, 0, arrivals, 900, 1000, 16, tpot)


def test_prefill_only_payload_carries_metadata_and_prefix():
    req = PrefillOnlyRequest(_ticket(), 512, "qwen2.5-14b")
    p = req.payload()
    assert p["max_tokens"] == 1 and p["tidemark"]["tidemark_ticket"] is True
    assert len(p["prompt"]) == 512 and p["tidemark"]["admitted_delta"] == 512
    assert req.request_id.startswith("tidemark:m:s:g0:f512:")


def test_shim_admits_sizes_and_commits():
    results = []
    shim = _shim(results)
    t = _ticket()
    shim.enqueue(t, "rid")
    shim.begin_step(_stats(1))
    assert shim.size_background("rid") == 1024
    assert results[-1].status is TicketStatus.ADMITTED
    shim.account("rid", 1024, 40.0)
    shim.end_step({"rid": (0, 1024)})
    done = results[-1]
    assert done.status is TicketStatus.COMMITTED and done.admitted_delta == 1024
    assert done.snapshot_hash == t.prefix_hash(1024)
    assert done.gpu_ms == 40.0


def test_shim_cancels_on_foreground_arrival():
    results = []
    shim = _shim(results)
    shim.enqueue(_ticket(), "rid")
    shim.begin_step(_stats(1))
    assert shim.size_background("rid") > 0
    shim.account("rid", 256, 10.0)
    shim.begin_step(_stats(2, decode=32, arrivals=1, tpot=9.0))
    assert results[-1].status is TicketStatus.CANCELLED
    assert results[-1].prefilled_tokens == 256
    assert shim.size_background("rid") == 0


def test_shim_blocks_under_tpot_pressure():
    results = []
    shim = _shim(results)
    shim.enqueue(_ticket(), "rid")
    shim.begin_step(_stats(1, decode=64, tpot=12.0))  # 12 > 9 * 1.03
    assert shim.size_background("rid") == 0
    assert not results


def test_partial_residency_reports_shortfall():
    results = []
    shim = _shim(results)
    shim.enqueue(_ticket(), "rid")
    shim.begin_step(_stats(1))
    shim.size_background("rid")
    shim.end_step({"rid": (0, 600)})  # engine only kept 600 of 1024 resident
    assert results[-1].admitted_delta == 600 and results[-1].reason == "partial_residency"


def test_ticket_round_trips_through_request_metadata():
    t = _ticket(delta_max=512, n=900)
    req = PrefillOnlyRequest(t, 512, "m")
    rebuilt = AtomicTicket.from_metadata(req.payload()["tidemark"], req.payload()["prompt"])
    assert (rebuilt.ticket_id, rebuilt.frontier, rebuilt.delta_max, rebuilt.generation) == (t.ticket_id, 0, 512, 0)
    assert rebuilt.prefix_hash(512) == t.prefix_hash(512)
