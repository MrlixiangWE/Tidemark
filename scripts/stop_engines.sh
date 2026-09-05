#!/usr/bin/env bash
# Stop engines started by scripts/launch_engines.sh.
set -euo pipefail
[[ -f .engines.pid ]] || { echo "nothing to stop"; exit 0; }
while read -r pid id; do
  if kill -0 "$pid" 2>/dev/null; then
    echo "stopping $id ($pid)"; kill "$pid"
  fi
done < .engines.pid
rm -f .engines.pid
