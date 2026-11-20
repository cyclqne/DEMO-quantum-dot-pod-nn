# Accuracy and POD timing methodology

## Wavefunction accuracy

LSE percent is defined by the normalized **squared** spatial residual:

`100 * ∫ |ψ_pred(r) − ψ_DNS(r)|² d²r / ∫ |ψ_DNS(r)|² d²r`.

The 2D grid is equispaced, so the factor `dx * dy` cancels. Real-valued
Schrödinger eigenvectors carry arbitrary signs; the calculated numerator is
the smaller of the residuals for `ψ_pred` and `−ψ_pred`. No rescaling or
additional fitting is allowed. Report this as "sign-aligned wavefunction
LSE (%)", not as ordinary relative L2 or as an "accuracy percentage".

The density LSE percentage uses the same squared ratio with `|ψ|²` in place
of ψ. Energy is the average **absolute relative** error, multiplied by 100.

CSV outputs retain original raw relative L2 diagnostics for comparison with
prior experiments, but accuracy plots include **only the global and local
neural networks**. POD remains in the CSV accuracy tables and in the spatial
reconstruction figures as an unscored field; POD LSE titles and POD residual
maps against DNS are omitted.

## Per-geometry online timings

`metrics/timings.csv` includes individual entries for DNS, POD-Galerkin, global
NN and local NN, all for the same held-out geometry identifiers. Its POD row
contains `pod_assembly_ms`, `pod_eigensolve_only_ms`,
`pod_reconstruction_ms`, `pod_full_online_ms`, and `total_online_ms`.

- **POD assembly**: evaluate the four-dot potential and assemble the
  geometry-specific reduced Hamiltonian using the offline POD basis and
  reduced kinetic term.
- **POD eigensolve only**: time only `np.linalg.eigh(H_red)`. The matrix is
  already built. This is not a complete online solver time.
- **POD reconstruction**: reconstruct the retained wavefunctions using the
  eigenvectors and all POD modes.
- **POD full online**: measured from start of potential assembly through end
  of spatial reconstruction, with the same outer `perf_counter` timer.

Each geometry is warmed up then measured repeatedly; each reported value is
its median over those repetitions. Since the median of a sum need not equal
the sum of medians, component medians may differ slightly from the full-time
median. Building the training POD basis and precomputing the reduced kinetic
matrix happen offline and are excluded from *both* online POD definitions.

NN online time includes construction of operator features + forward inference
+ full-space reconstruction; the forward-only timing is reported separately.
DNS timing covers **sparse eigensolver only**, as measured in DNS generation.
Therefore the DNS timing is not a complete full-order assembly + solve
comparison; the figures explicitly label it "DNS (solve only)".

The **one** POD timing comparison figure is
`online_time_by_geometry_reduced_methods_pod_eigensolve_only.png`. It displays
POD **reduced eigensolve-only time** beside complete online global/local NN
runtime per held-out geometry. No full-POD-online timing figure is exported;
assembly, eigensolve, reconstruction, and full timing remain in CSV outputs.
The compared computational scopes differ and are identified in the plot.
