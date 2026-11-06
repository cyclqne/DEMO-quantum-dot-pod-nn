# Validation record

Generated code was tested in a CPU environment with PyTorch, NumPy, SciPy,
Pandas and scikit-learn installed.

- `pytest` unit tests: research-rule score/widths/signatures, 49×49 metadata,
  finite-difference matrix Hermiticity, geometry split leakage, adaptive-width
  packed inference numerical equivalence, checkpoint round trip.
- 17×17 / 6-state / 10-mode smoke: completed full DNS/POD/training/evaluation.
- **49×49 / 40-state / 80-mode shape integration:** completed DNS, POD,
  146-feature input construction, original-style KMeans K=2..8 selection,
  global and multiple local-expert training, latency auditing, accuracy and
  summary tables. This shape test intentionally ran only 1+1 epochs, so its
  errors are **not** measurements of fully trained model performance.

This repository has not yet demonstrated that the 160+40-epoch *default*
run beats the global model. The goal is a reproducible, scientifically honest
comparison of the research clustering method—not a guaranteed outcome.

**Differences from original high-fidelity run:** 49×49 grid vs original
97×97, default 81 vs 1,296 geometries, shorter demo training, max-magnitude
component sign alignment rather than a near-zero grid-corner sign rule,
and a newly added independently trained global baseline. The original
KMeans rule and width thresholds are unchanged.
