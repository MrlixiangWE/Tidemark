# Deploying on vLLM

Server tiers run stock vLLM (V1 engine) with a small patch to the scheduler.
See [`engines/vllm/README.md`](../../engines/vllm/README.md) for what the patch
does line by line; this page is the operational checklist.

## Requirements

- vLLM 0.10.x with the V1 engine (the default). Older releases lack
  request-level `cached_tokens` in the usage block, which Tidemark needs.
- `--enable-prefix-caching` (on by default in V1).
- `--scheduling-policy priority`, so background requests can be tagged with
  the lowest priority and never jump ahead of foreground work in the queue.
- The `tidemark` package importable from the engine's Python environment. The
  shim is imported lazily by the patched scheduler and only when
  `TIDEMARK_SCHEDULER` is set.

## Steps

1. Install and patch:

   ```bash
   pip install "vllm==0.10.1" tidemark
   /path/to/Tidemark/engines/vllm/install.sh
   ```

2. Start the engine with the shim enabled:

   ```bash
   TIDEMARK_ENGINE_ID=cloud-2gpu \
   TIDEMARK_SCHEDULER=http://10.0.2.11:7420 \
   python -m vllm.entrypoints.openai.api_server \
     --model Qwen/Qwen2.5-14B-Instruct --tensor-parallel-size 2 \
     --enable-prefix-caching --scheduling-policy priority \
     --max-num-batched-tokens 8192 --port 8000
   ```

3. List the engine in the scheduler config with the same `engine_id`, then
   start the scheduler:

   ```bash
   tidemark serve -c configs/testbed/device_edge_cloud.yaml
   ```

4. Check the shim is live. The engine log prints
   `tidemark: admission shim active (engine=cloud-2gpu, gamma=0.03)` at the
   first scheduler iteration, and `GET /v1/status` on the scheduler shows the
   engine under `rates`.

## Calibration

The guard's `TPOT_ref` calibrates itself from the first 200 foreground decode
steps after start. If you restart the engine under load, set
`TIDEMARK_TPOT_REF_MS` explicitly from a previous run instead.

Prefill rates come from `scripts/calibrate_rates.py`. Run it against an idle
engine; it needs about two minutes per engine.

## Verifying reuse

A committed interval must be visible to the very next foreground request. To
check:

```bash
python scripts/bench_switch_ttft.py --endpoint http://10.0.1.10:8000 \
    --model Qwen/Qwen2.5-14B-Instruct --history 8192 --lags 0 2048 4096 8192
```

TTFT should scale linearly with the lag and the `cached` column should equal
`history − lag`. Over 240 switches per tier on our testbed, a committed frontier
and a clean full prefill yielded identical next-token ids.

## Tensor parallelism

The patch touches only the scheduler, which runs once per engine regardless
of TP degree. Nothing changes for TP > 1.

## Rolling back

```bash
/path/to/Tidemark/engines/vllm/install.sh --rollback
```

or simply unset `TIDEMARK_SCHEDULER`: every hook is a no-op without it and
the engine behaves exactly like unpatched vLLM.
