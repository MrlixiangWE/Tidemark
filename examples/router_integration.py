"""Wiring an application router to a running Tidemark scheduler.

The application keeps full control of routing. After every foreground request
it tells the scheduler three things: the text that was appended, which model
served it, and how much of the prompt that engine now holds. Everything else
happens in the background.

Requires a scheduler started with ``tidemark serve -c <config>``.
"""

from __future__ import annotations

import requests

from tidemark.engines.vllm.client import TokenIdClient
from tidemark.router import DifficultyRouter

SCHEDULER = "http://127.0.0.1:7420"
ENGINES = {
    "Qwen/Qwen2.5-1.5B-Instruct": TokenIdClient("http://127.0.0.1:8001", "Qwen/Qwen2.5-1.5B-Instruct"),
    "Qwen/Qwen2.5-7B-Instruct": TokenIdClient("http://127.0.0.1:8002", "Qwen/Qwen2.5-7B-Instruct"),
}
ENGINE_IDS = {"Qwen/Qwen2.5-1.5B-Instruct": "small", "Qwen/Qwen2.5-7B-Instruct": "large"}


def difficulty(prompt: str, _ctx) -> str:
    # Stand-in for a real difficulty classifier.
    return "cloud" if len(prompt.split()) > 40 or "explain" in prompt.lower() else "device"


router = DifficultyRouter(
    tier_models={"device": "Qwen/Qwen2.5-1.5B-Instruct", "cloud": "Qwen/Qwen2.5-7B-Instruct"},
    runtime_config="default",
    difficulty_fn=difficulty,
)


def chat(session: str, tenant: str, turns: list[str]) -> None:
    requests.post(f"{SCHEDULER}/v1/sessions", json={"session": session, "tenant": tenant}).raise_for_status()
    from transformers import AutoTokenizer

    toks = {m: AutoTokenizer.from_pretrained(m) for m in ENGINES}
    history: list[dict] = []
    last = None
    for i, text in enumerate(turns):
        decision = router.route(session, i, text, {})
        model = decision.model_id
        history.append({"role": "user", "content": text})
        requests.post(f"{SCHEDULER}/v1/sessions/{session}/turns", json={"role": "user", "text": text}).raise_for_status()

        prompt_ids = toks[model].apply_chat_template(history, add_generation_prompt=True)
        usage = ENGINES[model].foreground(prompt_ids, max_tokens=256)
        reply = "..."  # collect from the stream in a real client
        history.append({"role": "assistant", "content": reply})
        requests.post(f"{SCHEDULER}/v1/sessions/{session}/turns", json={"role": "assistant", "text": reply}).raise_for_status()

        # Tell Tidemark what happened. resident_prefix is the prompt length:
        # after a foreground request the engine holds the whole prompt.
        requests.post(
            f"{SCHEDULER}/v1/sessions/{session}/served",
            json={
                "model": model,
                "engine": ENGINE_IDS[model],
                "resident_prefix": usage.prompt_tokens,
                "cached_tokens": usage.cached_tokens,
                "uncached_tokens": usage.uncached_tokens,
                "ttft_ms": usage.ttft_ms,
                "switch": last is not None and last != model,
            },
        ).raise_for_status()
        print(f"turn {i}: {model.split('/')[-1]:<24} cached {usage.cached_ratio:5.1%}  ttft {usage.ttft_ms:7.1f} ms  {'SWITCH' if last and last != model else ''}")
        last = model
    requests.delete(f"{SCHEDULER}/v1/sessions/{session}")


if __name__ == "__main__":
    chat(
        "s-demo",
        "tenant-a",
        [
            "hi, what's the weather like for a run tonight?",
            "explain in detail why the dew point matters more than humidity for how a run feels, with the physics",
            "ok thanks. remind me to bring water",
            "explain how I should adjust pace for a 28 C dew point compared with my usual 14 C, step by step",
        ],
    )
