# Configuration reference

Tidemark reads one YAML (or JSON) file. Every key below has the default shown;
`configs/tidemark.yaml` is the annotated template and
`configs/testbed/device_edge_cloud.yaml` a complete example.

## `scheduler`

| key | default | symbol | meaning |
|---|---|---|---|
| `listen` | `0.0.0.0:7420` | | address of the scheduler's HTTP API |
| `delta_max` | `1024` | Δmax | largest interval a ticket may advance, in tokens |
| `alpha` | `1.0` | α | predictor weight; 1 = router signal only, 0 = history prior only |
| `lambda_mem_ms_per_gib` | `64` | λ_M | converts incremental KV occupancy into compute-time units |
| `kappa` | `2` | κ_u | outstanding admitted tickets per tenant |
| `beta` | `0.35` | β_u | share of the aggregate background budget per tenant |
| `ticket_lease_s` | `30` | | a ticket with no terminal report after this long is expired |

## `admission`

| key | default | symbol | meaning |
|---|---|---|---|
| `intervals` | `[256, 512, 1024]` | D | admitted sizes an engine may choose from |
| `x_max` | `1024` | X_max | cap on a single admission |
| `kv_headroom` | `0.08` | | fraction of KV blocks the safe budget keeps free |
| `gamma` | `0.03` | γ | decode-TPOT guard tolerance over `TPOT_ref` |
| `tpot_ewma_alpha` | `0.2` | | smoothing of the TPOT moving average |
| `calibration_steps` | `200` | | foreground-only steps used to fix `TPOT_ref` |

These are read by the scheduler process for the replay tool and passed to
engine shims through environment variables (`TIDEMARK_GAMMA`,
`TIDEMARK_INTERVALS`, `TIDEMARK_KV_HEADROOM`) at engine start.

## `engines[]`

| key | required | meaning |
|---|---|---|
| `engine_id` | yes | unique id; must match `TIDEMARK_ENGINE_ID` on the engine |
| `model` | yes | model id; also used to load a tokenizer on the scheduler side |
| `tier` | yes | `device`, `edge` or `cloud` (informational; ranking uses rates, not tiers) |
| `backend` | no | `vllm` (default) or `llamacpp` |
| `endpoint` | yes | base URL of the engine's HTTP API |
| `runtime_config` | no | string naming quantisation / backend / block size; part of the catalog key |
| `block_size` | no | KV block size, default 16 |
| `tau_fg_ms_per_ktok` | if no `rates_file` | foreground prefill rate |
| `tau_bg_ms_per_ktok` | if no `rates_file` | background prefill rate |
| `kv_bytes_per_token` | if no `rates_file` | KV footprint per token on this engine |
| `tpot_ref_ms` | no | reference decode TPOT; otherwise self-calibrated |

## `rates_file`

Path (relative to the config) of a rates YAML produced by
`scripts/calibrate_rates.py`. When present it overrides the inline rate fields.

## `telemetry`

| key | default | meaning |
|---|---|---|
| `directory` | `./telemetry` | where JSONL logs are written |
| `request_log` | `true` | one row per foreground request reported to the scheduler |
| `step_log` | `true` | one row per scheduler iteration with the pressure reason (engine-side) |
| `ticket_log` | `true` | one row per ticket state transition and catalog event |

## Sensitivity, briefly

From our sweeps under the high-load paced setting:

- **α.** Moving from the pure history prior (`α = 0`) to the first blend that
  includes the router signal (`α = 0.25`) raised top-1 destination accuracy at
  switches from 0.20 to 0.72 and cut the switch tail by 20.8 %. Above 0.25 the
  settings differ by less than run-to-run spread.
- **λ_M.** Moves the tail by no more than ±3.8 % over a 16× range; peak KV
  occupancy falls monotonically as it grows.
- **Tenant caps.** In an eight-tenant stress test with two heavy tenants, `κ`
  and the single issue point cut the largest tenant's share of background
  compute from 37.5 % to 26.1 %.
