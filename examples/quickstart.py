"""Tidemark in fifty lines, no GPU required.

Builds a two-engine deployment from fitted rates, opens a session, replays a
few turns that bounce between the models, and prints what the catalog and
scheduler did. Run with ``python examples/quickstart.py``.
"""

from tidemark import TidemarkConfig, TidemarkRuntime
from tidemark.catalog.history import TokenizerRegistry
from tidemark.runtime.config import EngineConfig
from tidemark.scheduler.predictor import StaticRouterSignal

# 1. Describe the engines. Rates come from scripts/calibrate_rates.py; these
#    are our edge and cloud numbers.
config = TidemarkConfig(
    engines=[
        EngineConfig("edge-0", "qwen2.5-7b", "edge", "http://edge:8000", tau_fg_ms_per_ktok=732, tau_bg_ms_per_ktok=344, kv_bytes_per_token=57344),
        EngineConfig("cloud-0", "qwen2.5-14b", "cloud", "http://cloud:8000", tau_fg_ms_per_ktok=191, tau_bg_ms_per_ktok=73, kv_bytes_per_token=98304),
    ]
)
config.scheduler.beta = 1.0  # a single tenant in this demo

# 2. Register a tokenizer per model. In production this is the engine's own
#    tokenizer; here a word splitter is enough to show the mechanics.
tokenizers = TokenizerRegistry()
tokenizers.register("qwen2.5-7b", lambda s: [hash(w) & 0xFFFF for w in s.split()])
tokenizers.register("qwen2.5-14b", lambda s: [hash(w) & 0xFFFF for w in s.split() for _ in (0, 1)])

# 3. The router tells Tidemark where the next turn is likely to go.
router = StaticRouterSignal()
issued = []
rt = TidemarkRuntime(config, tokenizers, router=router)
rt.scheduler.set_sink(issued.append)  # no real engines here; just collect tickets

hist = rt.open_session("demo", tenant_id="alice")
route = ["qwen2.5-7b", "qwen2.5-7b", "qwen2.5-14b", "qwen2.5-7b", "qwen2.5-14b"]
for i, model in enumerate(route):
    hist.append("user", " ".join(f"turn{i}-word{j}" for j in range(300)))
    router.set("demo", {"qwen2.5-7b": 0.3, "qwen2.5-14b": 0.7} if model == "qwen2.5-7b" else {"qwen2.5-7b": 0.7, "qwen2.5-14b": 0.3})
    # The engine serving this turn ends with the whole history resident.
    rt.turn_served("demo", model_id=model, resident_prefix=hist.length(model))
    other = "qwen2.5-14b" if model == "qwen2.5-7b" else "qwen2.5-7b"
    lag = [n for e, n in rt.catalog.lagging() if e.key.model_id == other]
    print(f"turn {i}: served by {model:<12} {other} lags by {lag[0] if lag else 0:>5} tokens; tickets issued so far: {len(issued)}")

t = issued[-1]
print(f"\nlast ticket -> engine={t.engine_id} interval=[{t.frontier}, {t.frontier + t.delta_max}) "
      f"p_future={t.p_future:.2f} benefit={t.expected_benefit_ms:.0f}ms cost={t.predicted_cost_ms:.0f}ms score={t.score:.2f}")
print("catalog:", rt.catalog.size_bytes(), "bytes for", sum(1 for _ in rt.catalog.entries()), "entries")
