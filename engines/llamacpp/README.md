# llama.cpp commit/reject adapter

The device tier runs `llama-server` from llama.cpp. It has no batching
scheduler iteration to hook, so Tidemark drives it from the outside through
the adapter in `tidemark/engines/llamacpp/adapter.py` and needs only a small
patch to the server so that a prefill-only request reports what it cached.

## What the patch adds

* `n_predict: 0` is accepted as a legitimate request that runs the prompt
  through the model, stores the KV state in the slot, and returns without
  sampling. Upstream treats it as "generate until EOS" in some versions.
* The completion response carries a `tidemark_commit` object:

  ```json
  {"tidemark_commit": {"tokens_cached": 1536, "tokens_evaluated": 512,
                       "prefix_hash": "3f9a...", "slot": 0}}
  ```

  `prefix_hash` is a BLAKE2b-128 over the token ids now held in the slot, the
  same hash the catalog computes, so the commit check does not need a second
  round trip to `/slots`.
* `/slots` includes the full `prompt_tokens` array for each slot, which the
  adapter uses to answer "how much of this session is still resident" after an
  outage or a slot reuse.

## Install

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout b6100
git apply /path/to/Tidemark/engines/llamacpp/tidemark_commit.patch
cmake -B build -DGGML_CUDA=ON   # or -DGGML_VULKAN=ON on the phone, -DGGML_METAL=ON etc.
cmake --build build --config Release -j
./build/bin/llama-server -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    --port 8080 --slots --cache-prompt -np 1 -c 4096
```

On the Galaxy S23+ we run the server under Termux with the Vulkan backend; on
the Jetson Orin NX and Raspberry Pi 5 the CUDA and CPU builds respectively.

## Behaviour without the patch

An unpatched server still works with the adapter: `cache_prompt: true` keeps
the prompt in the slot and `timings.cache_n` reports the cached count. The
adapter then re-hashes the prefix from `/slots` before committing, which costs
one extra request per interval on a link that is often the bottleneck. The
patch exists to remove that round trip.
