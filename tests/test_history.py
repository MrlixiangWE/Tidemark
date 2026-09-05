from tidemark.catalog.history import content_hash, longest_common_prefix


def test_views_differ_per_model(history):
    assert history.length("device-1b") < history.length("edge-7b") < history.length("cloud-14b")


def test_append_keeps_revision_and_prefix(history):
    before = history.token_ids("edge-7b")
    h_before = history.prefix_hash("edge-7b", len(before))
    history.append("user", "one more turn")
    assert history.revision == 0
    after = history.token_ids("edge-7b")
    assert after[: len(before)] == before
    assert history.prefix_hash("edge-7b", len(before)) == h_before


def test_rewrite_bumps_revision_and_reports_lcp(history):
    old = history.token_ids("edge-7b")
    lcp = history.rewrite(1, "completely different answer")
    assert history.revision == 1
    new = history.token_ids("edge-7b")
    assert lcp["edge-7b"] == longest_common_prefix(old, new)
    # The first turn is untouched, so the LCP covers at least that turn.
    assert lcp["edge-7b"] >= 2 * len("hello there how are you".split())


def test_snapshot_is_immutable(history):
    snap = history.snapshot("cloud-14b")
    history.append("user", "x y z")
    assert len(snap) < history.length("cloud-14b")
    assert snap.prefix_hash(len(snap)) == content_hash(snap.token_ids)


def test_content_hash_is_order_sensitive():
    assert content_hash([1, 2, 3]) != content_hash([3, 2, 1])
    assert content_hash([]) == content_hash(())
