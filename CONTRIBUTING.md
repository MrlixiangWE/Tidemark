# Contributing

Thanks for taking a look. Tidemark is a research runtime and we are happy to
take fixes, new engine adapters, and measurements from hardware we do not have.

## Setting up

```bash
git clone https://github.com/MrlixiangWE/Tidemark && cd Tidemark
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Everything under `tidemark/` runs without a GPU. The engine patches under
`engines/` need a real vLLM or llama.cpp checkout to test; CI only checks that
the vLLM patch still applies.

## What goes where

| you want to change | look in |
|---|---|
| what the catalog records or how validity is checked | `tidemark/catalog/` |
| how candidates are scored, capped or ordered | `tidemark/scheduler/ranking.py`, `global_scheduler.py` |
| how an engine decides whether an interval fits | `tidemark/admission/` |
| how a particular engine is driven | `tidemark/engines/<engine>/` and `engines/<engine>/` |
| defaults | `tidemark/runtime/config.py` and `configs/tidemark.yaml` |

If a change alters scheduling behaviour, please run the replay demo before and
after and paste both tables in the PR:

```bash
tidemark replay --trace examples/traces/demo.jsonl --load 0.6
tidemark replay --trace examples/traces/demo.jsonl --load 0.9
```

## Style

`ruff` with the settings in `pyproject.toml`. Type hints on public functions.
Docstrings explain *why* a piece of code exists, not what each line does; the
formulas from the design notes are welcome in comments where they help a reader
map code to the design.

Keep engine-specific knowledge inside the adapter for that engine. The
scheduler and the admission controller should only ever see plain numbers and
the `AtomicTicket` / `TicketResult` types.

## Adding an engine adapter

1. Subclass `tidemark.engines.base.EngineAdapter`.
2. Implement `step_state()`, `submit()`, `cancel()` and `resident_prefix()`.
3. Make sure a completed interval reports the *physically resident* prefix, not
   the requested one. The commit path treats any shortfall as a rejection.
4. Add a config example under `configs/examples/` and a short README under
   `engines/<name>/` saying which upstream versions you tested against.

## Reporting measurements

Rates for new hardware are welcome in `configs/testbed/rates.yaml`. Please
include the calibration command, the engine version, and the model
quantisation, and keep `tau_fg` and `tau_bg` from the same session.
