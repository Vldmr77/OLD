# Implementation Plan — Tinkoff Scalping System v10.4

## Stage 1 — Risk control alignment with v10.4 spec
- **Subtasks**
  1. ✅ *(2025-10-25)* Reconcile stop-loss and trailing logic with the specification formula that references entry price, ATR(5) and spread hard floor.
  2. Ensure RiskEngine-produced order plans respect VWAP/IOC routing split (>5 lots vs 1-5 lots) and expose aggressiveness/slices for execution.
  3. Extend unit coverage for stop management, circuit breaker triggers and hedge actions when imbalance thresholds are crossed.
- **Readiness criteria**
  - Stop recalculation uses entry price and enforces the `max(spread, 0.01)` guard everywhere.
  - Position adjustments emitted for stop-loss and hedging scenarios match spec thresholds (cooldown after 3 consecutive losses, LTE failover hook).
  - Tests in `tests/test_risk_engine.py` cover stop tightening, emergency halts and hedging decisions with the new formulas.
- **Artifacts**
  - Updated `scalp_system/risk/engine.py`, accompanying fixtures/tests, risk documentation snippet in `CHANGELOG.md`.
- **Test plan**
  - Unit focus: `pytest tests/test_risk_engine.py` plus targeted suites covering orchestration fallbacks if touched.
  - Full regression: `./scripts/run_tests.sh` (pytest wrapper to be introduced in Stage 1).
- **Metrics**
  - Stop-loss relative error ≤ 1e-6 against spec formula in tests.
  - 100% pass rate on new risk-engine specific tests.
- **Risks & mitigations**
  - *Risk*: Changing stop-loss math could invalidate hedging heuristics. *Mitigation*: Add regression tests for unaffected paths and compare exposures.
  - *Risk*: Added cooldown logic may conflict with orchestrator state restores. *Mitigation*: Validate snapshot/restore paths in tests.

## Stage 2 — Data & feature pipeline compliance
- **Subtasks**
  1. Align feature extraction defaults with depth-20 order books and verify 28+ feature coverage, including adaptive components.
  2. Validate DataEngine TTL/historical buffers, memory guard rails and instrument rotation side effects.
  3. Harden storage fallbacks (JSONL spillover, adaptive write cadence) with deterministic tests.
- **Readiness criteria**
  - Feature vectors remain stable (length + ordering) with lob_levels=20 configuration.
  - DataEngine rotations refresh active instruments and metrics at 1 Hz without leaks.
  - Storage layer surfaces backup artifacts when SQLite writes fail.
- **Artifacts**
  - Updates under `scalp_system/features`, `scalp_system/data`, `scalp_system/storage`, new tests and docstrings.
- **Test plan**
  - Targeted: `pytest tests/test_features.py tests/test_data_engine.py tests/test_historical_storage.py`.
  - Regression: `./scripts/run_tests.sh`.
- **Metrics**
  - Feature vector length ≥ 32 with lob_levels=20.
  - Cache eviction latency < 5 ms in synthetic benchmarks (unit-level measurement via timing assertions).
- **Risks & mitigations**
  - *Risk*: Enlarged feature vectors impact ML stub models. *Mitigation*: Update test doubles and ensure WeightedEnsemble accepts new dimensionality.
  - *Risk*: Tight TTL purging may drop data for orchestrator. *Mitigation*: Provide configuration knobs & tests for boundary values.

## Stage 3 — Monitoring, calibration & recovery hardening
- **Subtasks**
  1. Finalise drift detector thresholds/logging (KS test, mean diff σ bands) and ensure RiskEngine/Calibration integration.
  2. Cover disaster recovery paths: checkpoint restore, LTE failover, model corruption detection and rehydration workflow.
  3. Expand monitoring outputs (latency stats, resource alerts, audit log tags) aligning with ISO 27001 traceability.
- **Readiness criteria**
  - Drift events push calibration flags and emergency halts according to spec thresholds and log JSONL entries with timestamps.
  - Recovery scripts rebuild corrupted models and checkpoint state inside integration tests.
  - Monitoring exports metrics for latency, CPU/GPU load and sends notifications for circuit breakers.
- **Artifacts**
  - Changes in `scalp_system/monitoring`, `scalp_system/ml`, `scalp_system/storage/disaster_recovery.py`, plus documentation updates.
- **Test plan**
  - Targeted: `pytest tests/test_drift_detector.py tests/test_disaster_recovery.py tests/test_latency_monitor.py tests/test_notifications.py`.
  - Regression: `./scripts/run_tests.sh`.
- **Metrics**
  - Drift false-positive rate ≤5% in synthetic test harness.
  - Recovery completion time logged < 2 min simulated via timed tests.
- **Risks & mitigations**
  - *Risk*: Mock KS implementation diverges from SciPy accuracy. *Mitigation*: Cross-validate against reference vectors inside tests.
  - *Risk*: Recovery flows may be flaky due to filesystem timing. *Mitigation*: Use temp directories and deterministic mocks.

## Progress tracking
- Current stage: **Stage 1 — Risk control alignment**
- Completed work: Stage 1 Subtask 1 delivered on 2025-10-25 (stop-loss refactor + test runner).
- Remaining stages: Stage 1 Subtasks 2-3, Stage 2, Stage 3
- Next action: Implement Stage 1 Subtask 2 (VWAP/IOC routing validation and exposure signalling) with focused unit tests.
