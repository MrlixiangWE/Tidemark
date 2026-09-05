# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.1] - 2026-09-05

### Changed
- `AtomicTicket` carries `engine_id`, `p_future` and `score` in its request
  metadata so an engine can rebuild the full ticket from the request alone.
- The llama.cpp adapter no longer has a `Mixed` mode on single-slot servers;
  it is `Idle` or `Blocked`, which matches what we measured on the device tier.

### Fixed
- A ticket whose target ran past the end of the tokenised history could be
  reserved when a foreground request and an append raced. The catalog now
  refuses the reservation.

## [0.3.0] - 2026-08-27

### Added
- Engine-local admission rewritten around the safe budget and the three-mode
  decision function; interval set fixed to `{256, 512, 1024}`.
- Decode-TPOT guard with self-calibrating reference (`gamma = 0.03`).
- Per-tenant caps on outstanding tickets and on the share of the aggregate
  background budget, plus queue aging.
- `tidemark replay`: a GPU-free replay of the control loop for development.
- llama.cpp adapter and server patch for the device tier.

### Changed
- The catalog key is now `(session, model, runtime_config)`; a session-level
  residency bit is gone for good.
- Ranking divides by `tau_bg`-weighted compute time instead of raw tokens.

### Removed
- The v2 "slack controller" and its pressure levels. The three-mode function
  is simpler and does the same job.

## [0.2.0] - 2026-07-24

### Added
- Versioned frontier catalog with generation numbers and LCP retraction on
  edits.
- Atomic tickets; a cancelled or stale ticket leaves the frontier unchanged.
- vLLM V1 scheduler shim and installer.

## [0.1.0] - 2026-06-16

- First internal prototype: whole-suffix prefetch of the predicted destination
  on top of vLLM's prefix cache. Kept as the `full-prefetch` policy in the
  replay tool for comparison.
