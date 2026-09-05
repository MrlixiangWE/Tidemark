#!/usr/bin/env bash
# Start every engine listed in a Tidemark config on the local machine.
#
#   scripts/launch_engines.sh configs/examples/single_node.yaml
#
# Reads engine entries with `yq`, starts vLLM engines with prefix caching and
# the priority scheduler (which the Tidemark shim relies on), and llama.cpp
# engines with prompt caching and slot reporting. Writes PIDs to .engines.pid
# so `scripts/stop_engines.sh` can tear them down.

set -euo pipefail
CFG="${1:?usage: launch_engines.sh <config.yaml>}"
command -v yq >/dev/null || { echo "yq is required (https://github.com/mikefarah/yq)"; exit 1; }

SCHED="${TIDEMARK_SCHEDULER:-http://127.0.0.1:7420}"
: > .engines.pid

n=$(yq '.engines | length' "$CFG")
for ((i = 0; i < n; i++)); do
  id=$(yq -r ".engines[$i].engine_id" "$CFG")
  model=$(yq -r ".engines[$i].model" "$CFG")
  backend=$(yq -r ".engines[$i].backend // \"vllm\"" "$CFG")
  endpoint=$(yq -r ".engines[$i].endpoint" "$CFG")
  port="${endpoint##*:}"
  case "$backend" in
    vllm)
      echo "starting vLLM engine $id ($model) on :$port"
      TIDEMARK_ENGINE_ID="$id" TIDEMARK_SCHEDULER="$SCHED" \
      nohup python -m vllm.entrypoints.openai.api_server \
        --model "$model" --port "$port" \
        --enable-prefix-caching --scheduling-policy priority \
        --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-4096}" \
        --gpu-memory-utilization "${GPU_MEM_UTIL:-0.85}" \
        > "logs/engine-$id.log" 2>&1 &
      echo "$! $id" >> .engines.pid
      ;;
    llamacpp)
      gguf="${GGUF_DIR:-models}/$(basename "$model").Q4_K_M.gguf"
      echo "starting llama.cpp engine $id ($gguf) on :$port"
      nohup llama-server -m "$gguf" --port "$port" --slots --cache-prompt -np 1 -c "${LLAMA_CTX:-4096}" \
        > "logs/engine-$id.log" 2>&1 &
      echo "$! $id" >> .engines.pid
      ;;
    *) echo "unknown backend $backend for $id"; exit 1 ;;
  esac
done

echo "waiting for engines to answer..."
for ((i = 0; i < n; i++)); do
  endpoint=$(yq -r ".engines[$i].endpoint" "$CFG")
  until curl -sf "$endpoint/health" >/dev/null 2>&1 || curl -sf "$endpoint/v1/models" >/dev/null 2>&1; do sleep 2; done
  echo "  $endpoint up"
done
echo "all engines up; start the scheduler with: tidemark serve -c $CFG"
