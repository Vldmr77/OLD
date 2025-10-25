# Changelog

## [2025-10-26] Stage 1 — Circuit breakers & hedging coverage
### Added
- Tests asserting trailing-stop tightening on both long and short legs now capture emergency halt reasons and exposure-based
  hedge/reduce actions, ensuring circuit breaker paths match the risk specification.
- Regression guard verifying loss streak counters reset after profitable trades to prevent perpetual cooldown states.
### Changed
- RiskEngine exposes the latest emergency halt reason for downstream diagnostics and resets consecutive loss counters whenever
  a trade realizes profit, keeping cooldown logic faithful to the specification.

## [2025-10-25] Stage 1 — VWAP/IOC routing hints
### Changed
- RiskEngine now derives routing strategy, aggressiveness ratio and slice count through a dedicated helper so >5 lot orders are
  tagged for VWAP execution while smaller clips remain IOC, aligning execution hints with the trading specification.
### Added
- Regression tests covering IOC vs VWAP selection, aggressiveness hints and slice accounting within `tests/test_risk_engine.py`.

## [2025-10-25] Stage 1 — Risk control alignment kickoff
### Changed
- Refined RiskEngine trailing stop computation to use entry price ratios and the specification's spread floor, ensuring stop adjustments respect position direction and emit precise tighten/closure signals.
- Updated Tinkoff async client helper to only forward gRPC option parameters when configured, keeping compatibility with lightweight stubs used in tests.
### Added
- Introduced `scripts/run_tests.sh` as the canonical pytest runner for local and CI executions, enabling consistent invocation across stages.
### Fixed
- Adjusted unit tests to assert sandbox target prefixes rather than legacy sentinel values so they remain compatible with real endpoint strings.
