# The device-edge-cloud testbed

This is the physical setup we evaluate on. It is here so the numbers
in `configs/testbed/` have a referent and so someone assembling a similar
testbed knows what to expect.

## Platforms

| tier | platform | compute | memory | model | engine |
|---|---|---|---|---|---|
| device | Samsung Galaxy S23+ | Snapdragon 8 Gen 2 | 8 GB | Llama-3.2-1B-Instruct (Q4_K_M) | llama.cpp, Vulkan |
| device | NVIDIA Jetson Orin NX | Ampere, 16 SM | 16 GB | Phi-3.5-mini-instruct (Q4_K_M) | llama.cpp, CUDA |
| device | Raspberry Pi 5 | Cortex-A76 ×4 | 8 GB | Gemma-3-1B-it (Q4_K_M) | llama.cpp, CPU |
| edge | 2 × NVIDIA Jetson AGX Orin | Ampere, 64 GB each | 64 GB | Qwen2.5-7B-Instruct-AWQ | vLLM V1 |
| cloud | server, 2 × PCIe GPU | Ampere | 160 GB | Qwen2.5-14B-Instruct | vLLM V1 |

Devices reach the edge over a 100 Mbps wireless link; the edge reaches the
cloud over wired Ethernet. The Tidemark scheduler process runs on one of the
two AGX Orin modules, one wireless hop from the devices, so server-side
frontiers remain available while a device is unreachable and new device
history is synchronised when the link returns.

## Model relationships

None of the models share KV state. The device platforms carry three different
compact models chosen for their SoCs; both server tiers run Qwen2.5 at
different sizes, and even two sizes of one family produce KV tensors of
different shapes. Every switch therefore requires the destination to run its
own prefill, which is what Tidemark schedules.

## Fitted rates

`configs/testbed/rates.yaml` holds `τ_fg` and `τ_bg` for every engine.
Regenerate them with

```bash
python scripts/calibrate_rates.py --config configs/testbed/device_edge_cloud.yaml --out configs/testbed/rates.yaml
```

The device engines are calibrated on the device itself with
`scripts/bench_switch_ttft.py` against the local `llama-server`, since driving
them over the wireless link would fold link latency into the fit.

## Link emulation

Variable-link experiments replay recorded traces on the device-edge hop with
`tc`/`netem`: a walking trace (median RTT 18 ms, outages of 0.8 s to 4.1 s) and
a vehicular trace (P95 RTT 310 ms, three cell handovers). During an outage the
device serves locally and buffers its output; server frontiers hold at the
last token that crossed, and the buffered tokens are prefilled after
reassociation.

## Power measurement

Device energy is sampled at 5 kHz by an inline DC power monitor on each
platform's supply rail and reported above the platform's idle baseline. The
radio component is separated by differencing against a loopback replay.
