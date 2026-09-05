# Deploying on llama.cpp (device tier)

Device platforms run `llama-server`. Tidemark drives it from the outside
through the adapter in `tidemark/engines/llamacpp/`; a small server patch adds
commit accounting to the completion response so the adapter does not need a
second round trip over the wireless link. See
[`engines/llamacpp/README.md`](../../engines/llamacpp/README.md) for the patch.

## Build

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout b6100
git apply /path/to/Tidemark/engines/llamacpp/tidemark_commit.patch
cmake -B build -DGGML_VULKAN=ON        # Galaxy S23+ under Termux
cmake -B build -DGGML_CUDA=ON          # Jetson Orin NX
cmake -B build                         # Raspberry Pi 5 (CPU)
cmake --build build --config Release -j
```

## Run

```bash
./build/bin/llama-server -m Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    --host 0.0.0.0 --port 8080 --slots --cache-prompt -np 1 -c 4096 -b 512
```

`--slots` exposes `/slots`, which the adapter uses for residency;
`--cache-prompt` keeps the prompt's KV in the slot between requests; `-np 1`
is the single-slot configuration we evaluate (multi-slot works but the mode
logic treats any busy slot as `Blocked`).

## Scheduler side

```yaml
engines:
  - engine_id: phone-s23
    model: meta-llama/Llama-3.2-1B-Instruct
    tier: device
    backend: llamacpp
    endpoint: http://10.0.3.21:8080
    runtime_config: llamacpp-q4km-4k
```

The `model` string is used to load a tokenizer on the scheduler side so token
offsets line up with what the device hashes. Make sure the GGUF you serve was
converted from that exact tokenizer.

## What happens during an outage

When the device loses the link it keeps serving locally with whatever is in its
slot. The scheduler cannot reach it, so tickets for the device engine are
refused with `engine_error` and the device's frontier holds at its last
committed position. Server-side frontiers keep advancing from the history the
scheduler has. On reassociation the application posts the buffered device turns
to the scheduler, the device's entries retract to whatever `/slots` reports
resident, and ranking resumes.

## Energy

Background prefill on a phone costs energy for state that may never be read.
In a 20-turn session with six switches on our testbed, shorter foreground prefill
and less radio idle-wait returned more energy than the background prefill
added (a net 5.5 % saving); where no committed frontier was ever read the loss
was bounded at 12.5 %. If your workload rarely returns to the device, set
`kappa: 0` for its tenant or list the device with a large `tau_bg` so the
ranking discounts it.
