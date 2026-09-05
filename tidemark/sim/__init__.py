"""A small discrete-event replay of the Tidemark control loop.

The simulator exists so the scheduling logic can be exercised without a GPU:
it drives the real catalog, scheduler and admission code against fitted rates
and a route trace, and reports the switch TTFT each policy would have paid. It
is not a substitute for the testbed, and the numbers in the paper come from
real engines; the simulator is for development, regression tests and the
quick-start demo.
"""

from tidemark.sim.replay import Policy, ReplayEngine, ReplayReport, replay_trace

__all__ = ["ReplayEngine", "ReplayReport", "Policy", "replay_trace"]
