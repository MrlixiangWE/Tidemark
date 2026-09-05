import pytest

from tidemark.scheduler.predictor import DestinationPredictor, HistoryTransitionEstimator, StaticRouterSignal

MODELS = ("a", "b", "c")


def test_history_prior_is_uniform_before_evidence():
    est = HistoryTransitionEstimator(MODELS)
    p = est.probabilities("s")
    assert all(abs(v - 1 / 3) < 1e-9 for v in p.values())


def test_history_prior_learns_transitions():
    est = HistoryTransitionEstimator(MODELS, smoothing=0.1)
    for m in ("a", "b", "a", "c", "a", "b", "a", "b", "a"):
        est.observe("s", m)
    p = est.probabilities("s")  # from "a": b three times, c once, a never
    assert p["b"] > p["c"] > p["a"]


def test_alpha_one_is_pure_router():
    router = StaticRouterSignal({"s": {"a": 0.1, "b": 0.8, "c": 0.1}})
    pred = DestinationPredictor(MODELS, router=router, alpha=1.0)
    assert pred.probability("s", "b") == pytest.approx(0.8)


def test_alpha_zero_ignores_router():
    router = StaticRouterSignal({"s": {"a": 0.0, "b": 1.0, "c": 0.0}})
    pred = DestinationPredictor(MODELS, router=router, alpha=0.0)
    assert pred.probability("s", "b") == pytest.approx(1 / 3)


def test_missing_router_opinion_falls_back_to_prior():
    pred = DestinationPredictor(MODELS, router=StaticRouterSignal(), alpha=1.0)
    assert pred.probability("unknown", "a") == pytest.approx(1 / 3)


def test_alpha_out_of_range_rejected():
    with pytest.raises(ValueError):
        DestinationPredictor(MODELS, alpha=1.5)
