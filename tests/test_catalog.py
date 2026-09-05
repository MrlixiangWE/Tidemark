from tidemark.catalog import FrontierKey, Placement, VersionedFrontierCatalog
from tidemark.catalog.history import content_hash
from tidemark.catalog.validity import CommitVerdict

PAIRS = [("device-1b", "cfg"), ("edge-7b", "cfg"), ("cloud-14b", "cfg")]


def _catalog(history):
    cat = VersionedFrontierCatalog()
    cat.register_session(history, PAIRS)
    return cat


def _commit(cat, hist, key, frontier, delta, generation=None, revision=None):
    entry = cat.get(key)
    ids = hist.token_ids(key.model_id)
    return cat.commit_background(
        key,
        ticket_id="t",
        ticket_frontier=frontier,
        ticket_generation=entry.generation if generation is None else generation,
        ticket_revision=hist.revision if revision is None else revision,
        admitted_delta=delta,
        requested_delta=delta,
        snapshot_hash=content_hash(ids[: frontier + delta]),
        engine_id="e",
    )


def test_every_model_starts_lagging(history):
    cat = _catalog(history)
    lag = {e.key.model_id: n for e, n in cat.lagging()}
    assert set(lag) == {m for m, _ in PAIRS}
    assert all(n > 0 for n in lag.values())


def test_foreground_commit_advances_only_serving_model(history):
    cat = _catalog(history)
    key = FrontierKey("s1", "edge-7b", "cfg")
    cat.commit_foreground(key, resident_prefix=history.length("edge-7b"), engine_id="edge-0")
    assert cat.get(key).frontier == history.length("edge-7b")
    assert cat.get(FrontierKey("s1", "cloud-14b", "cfg")).frontier == 0
    lagging = {e.key.model_id for e, _ in cat.lagging()}
    assert lagging == {"device-1b", "cloud-14b"}


def test_background_interval_commits_and_continues(history):
    cat = _catalog(history)
    key = FrontierKey("s1", "cloud-14b", "cfg")
    total = history.length("cloud-14b")
    first = min(8, total)
    assert _commit(cat, history, key, 0, first).ok
    e = cat.get(key)
    assert e.frontier == first and e.placement is Placement.RESIDENT
    rest = total - first
    if rest:
        assert _commit(cat, history, key, first, rest).ok
        assert cat.get(key).frontier == total


def test_stale_generation_is_rejected(history):
    cat = _catalog(history)
    key = FrontierKey("s1", "cloud-14b", "cfg")
    gen = cat.get(key).generation
    # A foreground request on the same model lands first and bumps the generation.
    cat.commit_foreground(key, resident_prefix=4, engine_id="cloud-0")
    v = _commit(cat, history, key, 0, 4, generation=gen)
    assert not v.ok and v.reason == CommitVerdict.STALE_GENERATION


def test_gap_or_overlap_is_rejected(history):
    cat = _catalog(history)
    key = FrontierKey("s1", "cloud-14b", "cfg")
    assert _commit(cat, history, key, 0, 4).ok
    v = _commit(cat, history, key, 2, 4)  # overlaps the committed interval
    assert not v.ok and v.reason == CommitVerdict.FRONTIER_MOVED


def test_edit_retracts_to_lcp_and_invalidates_inflight(history):
    cat = _catalog(history)
    key = FrontierKey("s1", "cloud-14b", "cfg")
    total = history.length("cloud-14b")
    assert _commit(cat, history, key, 0, total).ok
    gen = cat.get(key).generation
    assert cat.reserve(key, "inflight", total + 5) is False  # nothing beyond history yet
    lcp = history.rewrite(1, "a different reply entirely")
    cat.on_revision_transition("s1", lcp)
    e = cat.get(key)
    assert e.generation == gen + 1
    assert e.frontier == lcp["cloud-14b"] <= total
    assert e.inflight_target is None


def test_hash_mismatch_after_edit_inside_interval(history):
    cat = _catalog(history)
    key = FrontierKey("s1", "cloud-14b", "cfg")
    total = history.length("cloud-14b")
    stale_hash = content_hash(history.token_ids("cloud-14b")[:total])
    history.rewrite(1, "something else")
    cat.on_revision_transition("s1", {"cloud-14b": 3})
    e = cat.get(key)
    v = cat.commit_background(
        key,
        ticket_id="t",
        ticket_frontier=0,
        ticket_generation=e.generation,
        ticket_revision=history.revision,
        admitted_delta=min(total, history.length("cloud-14b")),
        requested_delta=min(total, history.length("cloud-14b")),
        snapshot_hash=stale_hash,
        engine_id="e",
    )
    assert not v.ok and v.reason == CommitVerdict.HASH_MISMATCH


def test_truncated_interval_does_not_commit(history):
    cat = _catalog(history)
    key = FrontierKey("s1", "edge-7b", "cfg")
    e = cat.get(key)
    v = cat.commit_background(
        key,
        ticket_id="t",
        ticket_frontier=0,
        ticket_generation=e.generation,
        ticket_revision=0,
        admitted_delta=3,
        requested_delta=8,
        snapshot_hash=content_hash(history.token_ids("edge-7b")[:3]),
        engine_id="e",
    )
    assert not v.ok and v.reason == CommitVerdict.TRUNCATED
    assert cat.get(key).frontier == 0


def test_eviction_retracts_frontier(history):
    cat = _catalog(history)
    key = FrontierKey("s1", "edge-7b", "cfg")
    total = history.length("edge-7b")
    assert _commit(cat, history, key, 0, total).ok
    cat.on_eviction(key, resident_prefix=total // 2)
    assert cat.get(key).frontier == total // 2
    cat.on_eviction(key, resident_prefix=0)
    assert cat.get(key).placement is Placement.EVICTED


def test_size_estimate_is_per_entry(history):
    cat = _catalog(history)
    assert cat.size_bytes() == 312 * len(PAIRS)
