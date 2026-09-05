"""Engine adapters.

An adapter is the only code in Tidemark that knows how a particular inference
engine works. It has three jobs: report the engine's per-iteration state to the
local admission controller, run a prefill-only request for an admitted
interval, and tell the catalog what prefix is still physically resident when
asked. Two adapters ship with the repository:

* :mod:`tidemark.engines.vllm` -- an admission shim that hooks into the vLLM V1
  scheduler iteration (installed as a small patch, see ``engines/vllm/``), plus
  an OpenAI-compatible client that speaks token ids and reads request-level
  ``cached_tokens`` accounting.
* :mod:`tidemark.engines.llamacpp` -- a commit/reject adapter for the
  on-device ``llama-server``, which has no scheduler iteration to hook and is
  driven through its slot and prompt-cache endpoints instead.
"""

from tidemark.engines.base import EngineAdapter, EngineDescriptor, ResidentPrefix

__all__ = ["EngineAdapter", "EngineDescriptor", "ResidentPrefix"]
