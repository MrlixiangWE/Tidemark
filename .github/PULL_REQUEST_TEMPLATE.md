## What

## Why

## How was it tested?

- [ ] `pytest` passes
- [ ] `ruff check tidemark tests scripts` passes
- [ ] `tidemark replay --trace examples/traces/demo.jsonl` still shows Tidemark below APC on P95 and below full-prefetch on background compute
- [ ] If the engine patches changed: `engines/vllm/install.sh` applies cleanly on the supported vLLM versions
