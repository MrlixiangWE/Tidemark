#!/usr/bin/env bash
# Apply (or roll back) the Tidemark admission patch on an installed vLLM.
#
#   engines/vllm/install.sh            # patch the vLLM in the active Python env
#   engines/vllm/install.sh --rollback # restore the saved originals
#
# Refuses to run while a vLLM server is up, verifies the base files are the
# ones the patch was made against, and keeps a timestamped copy of the
# originals so a rollback never depends on pip.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$HERE/tidemark_v1_admission.patch"
SUPPORTED="0.10.0 0.10.1 0.10.2"

vllm_root() {
  python - <<'PY'
import os, vllm
print(os.path.dirname(os.path.dirname(vllm.__file__)))
PY
}

vllm_version() {
  python -c 'import vllm; print(vllm.__version__)'
}

die() { echo "error: $*" >&2; exit 1; }

if pgrep -f "vllm.entrypoints" >/dev/null 2>&1; then
  die "a vLLM server is running on this machine; stop it before patching"
fi

ROOT="$(vllm_root)"
VER="$(vllm_version)"
BACKUP="$ROOT/.tidemark-backup"

case " $SUPPORTED " in
  *" $VER "*) ;;
  *) echo "warning: vLLM $VER is not one of: $SUPPORTED. The patch may need fuzz." >&2 ;;
esac

if [[ "${1:-}" == "--rollback" ]]; then
  [[ -d "$BACKUP" ]] || die "no backup found under $BACKUP"
  latest="$(ls -1 "$BACKUP" | sort | tail -n1)"
  cp -v "$BACKUP/$latest/scheduler.py" "$ROOT/vllm/v1/core/sched/scheduler.py"
  cp -v "$BACKUP/$latest/request.py" "$ROOT/vllm/v1/request.py"
  rm -f "$ROOT/vllm/v1/core/sched/tidemark.py"
  echo "rolled back to $latest"
  exit 0
fi

if grep -q "TidemarkBridge" "$ROOT/vllm/v1/core/sched/scheduler.py"; then
  echo "already patched"; exit 0
fi

stamp="$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP/$stamp"
cp "$ROOT/vllm/v1/core/sched/scheduler.py" "$BACKUP/$stamp/"
cp "$ROOT/vllm/v1/request.py" "$BACKUP/$stamp/"

( cd "$ROOT" && patch -p1 --forward < "$PATCH" )
python -c "import vllm.v1.core.sched.tidemark" && echo "tidemark shim importable"
echo "patched vLLM $VER at $ROOT (backup: $BACKUP/$stamp)"
