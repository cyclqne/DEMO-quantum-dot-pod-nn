# Cluster provenance and mathematical selection rules

Source of record: user's `final_lasso_cluster_40state_80mode_public.py` (Oct
2026), specifically `compute_state_mode_signatures`,
`remap_labels_by_state_order`, `validation_score_for_labels`,
`choose_validated_state_clusters`, and `choose_width`.

`src/qdpod/clustering.py` implements their selection logic directly using
`coeff[geometry, state, mode]` instead of the original flattened sample index.

Let `a[g,s,j]` be the train-only POD coefficient for geometry `g`, state `s`,
mode `j`. For each state compute:

```
b[g,s,j] = sign(a[g,s,j]) * max(abs(a[g,s,j]) - 0.0001, 0)
q[s,j] = mean_{g in split}(b[g,s,j]^2)
signature[s,j] = q[s,j] / max(sum_j(q[s,j]), 1e-12)
```

Fit `KMeans(n_clusters=K, random_state=42, n_init=30)` to *training*
signatures for `K in 2..8`. Renumber the cluster labels by the mean index of
the assigned states. Construct a training centroid for each renumbered group
from the training signatures. Evaluate the **same** state assignment against
the validation signatures:

```
compactness = mean_s ||signature_val[s] - centroid[assigned[s]]||^2
counts = number of states assigned to each expert
min_penalty = sum_k max(0, 1 - counts[k])^2
max_penalty = sum_k max(0, counts[k] - 18)^2
imbalance = std(counts) / max(mean(counts), 1e-12)
score = compactness + 0.08*min_penalty + 0.02*max_penalty + 0.005*imbalance
```

Choose the first candidate with minimum score. The candidate order is
`[2,3,4,5,6,7,8]`; a tie keeps the earlier winner. The chosen cluster
assignment determines **all** test-state routing; there is **no** KMeans
prediction on each held-out geometry and no use of its ground-truth POD
coefficients for routing.

Neural width from original source:

```
if state_count <= 3:   width = 64
elif state_count >= 10: width = 160
elif state_count >= 7:  width = 128
else:                  width = 96
```

The simplified 6-state smoke config uses K=2..4 only to avoid asking KMeans
to form more clusters than states. Full 40-state configs retain K=2..8.

**Not copied from the earlier streamlined demo:** latency-weighted KMeans,
normalized blend of latency and quality scores, reduced 6-state clustering,
and alternative expert-width thresholds.
