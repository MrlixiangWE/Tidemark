# vLLM admission shim

Tidemark's server-side engines run stock vLLM (V1 engine, `>= 0.10`) plus a
small patch to `vllm/v1/core/sched/scheduler.py` that lets the scheduler size
background prefill against the safe budget of the current iteration. The patch
does not touch the KV cache manager, the block pool, or the model runner; it
reuses vLLM's scheduler iteration, PagedAttention allocator, block hashing and
prefix-cache lookup as they are.

What the patch does, in order, inside `Scheduler.schedule()`:

1. After running requests (decode and in-progress prefill) have been scheduled
   and before the waiting queue is walked, it builds a `SchedulerStepStats` and
   calls `shim.begin_step(...)`. This is where the safe budget `X_t` and the
   three-mode classification are computed for the iteration.
2. When it pops a waiting request whose `extra_args` carry a `tidemark` block,
   it asks `shim.size_background(request_id)` for the number of new tokens it
   may schedule for that request in this step. A zero answer leaves the request
   in the queue; foreground requests are never held back by it.
3. After the step's outputs are known it calls `shim.end_step(...)` with the
   finished background requests and their `num_cached_tokens`, plus any
   preempted ones. That produces the ticket results the global scheduler
   validates and commits.

The prefill-only request itself is an ordinary `/v1/completions` call with
`max_tokens=1` at the lowest priority; the one generated token is discarded.
What Tidemark reads back is `usage.prompt_tokens_details.cached_tokens`, which
vLLM already reports when prefix caching is enabled.

## Install

```bash
# on each server-tier machine
pip install "vllm==0.10.1" tidemark
cd $(python -c "import vllm, os; print(os.path.dirname(vllm.__file__))")/..
patch -p1 < /path/to/Tidemark/engines/vllm/tidemark_v1_admission.patch

# start the engine with the shim enabled
TIDEMARK_ENGINE_ID=agx-orin-a \
TIDEMARK_SCHEDULER=http://10.0.2.11:7420 \
TIDEMARK_GAMMA=0.03 \
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ --enable-prefix-caching \
  --max-num-batched-tokens 4096 --scheduling-policy priority --port 8000
```

`install.sh` wraps the above, verifies the base file hashes before patching,
and keeps a timestamped copy of the original files for `--rollback`.

## Environment variables read by the shim

| variable | default | meaning |
|---|---|---|
| `TIDEMARK_ENGINE_ID` | hostname | engine id as listed in the scheduler config |
| `TIDEMARK_SCHEDULER` | unset | scheduler URL to post ticket results to; unset disables the shim |
| `TIDEMARK_GAMMA` | `0.03` | decode-TPOT guard tolerance |
| `TIDEMARK_INTERVALS` | `256,512,1024` | admitted interval sizes |
| `TIDEMARK_KV_HEADROOM` | `0.08` | fraction of KV blocks the safe budget keeps free |
| `TIDEMARK_STEP_LOG` | unset | path of a JSONL file to write one row per scheduler iteration |

## Without the patch

A stock vLLM still accepts the prefill-only request as a plain low-priority
completion. Nothing breaks; the engine simply prefills the whole requested
interval whenever the priority scheduler gets to it, with no safe budget and no
guard. That is precisely the whole-suffix prefetch baseline we compare against, which
is convenient for A/B runs.
