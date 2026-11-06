# Stronger training and validation diagnostics

## Why the baseline is inaccurate

The existing 81-geometry default trains 40 eigenstates with 80 POD modes
using 160 + 40 epochs, batch size 256, four-layer tanh MLPs (global width
96). Local experts train on only their assigned states. The weighted loss
reported during optimization includes coefficients, active-coefficient
weighting, energy, and physics-informed losses. Its numerical size is **not**
the same as the reported wavefunction L2 error. The near-zero POD-Galerkin
error on the supplied plots suggests the reduced representation is good, but
it does **not** establish that the neural mapping can generalize given only
81 potential combinations.

## Controlled longer run: `configs/train_longer_40state.yaml`

- Exactly the **same 49x49, 40-state, 80-mode, q=3, 81-geometry** distribution.
- Identical KMeans candidates, signature soft threshold, validation
  compactness objective and original 64/96/128/160 width policy.
- Original tanh architecture, deeper optimization: up to 650 + 150 epochs;
  64-example minibatches, cosine learning-rate annealing, and lower decay.
- Log validation sign-invariant **full-space relative wavefunction L2** and
  energy relative error each 10 epochs. Use only validation geometries to
  choose checkpoint and stop early after 20 validation evaluations without
  sufficient wave-error improvement.

## Capacity + dataset run: `configs/strong_40state.yaml`

- Still **49x49, 40 states, 80 modes**, but q=4 -> 256 potential geometries.
- Identical research KMeans selection objective/candidates/penalties.
- Wider four-layer networks with SiLU and widths 96/128/160/192 for
  increasing expert-state counts; global width 160.
- Up to 850 + 200 epochs, batch size 128, annealing and validation checkpoint
  selection. Expected to take substantially longer to run on CPU.
- This run is *not* an equal-parameter comparison to the original global
  network. Compare parameter counts and measured online runtime transparently.

## Validation metric derivation

For orthonormal POD basis Phi, true full-space wavefunction psi, its POD
coefficients c = Phi^T psi, and predicted coefficients a:

    ||psi +/- Phi a||^2 = ||psi||^2 + ||a||^2 +/- 2 c^T a.

Therefore the sign-invariant relative L2 on validation examples is:

    sqrt(max(0, ||psi||^2 + ||a||^2 - 2|c^T a|) / ||psi||^2)

which is evaluated without explicitly reconstructing the high-dimensional
wavefunction on every validation step. The sign is selected per state.
It matches direct reconstruction numerically (verified by a unit test).

Checkpoint selection and early stopping use validation wave L2 and **never**
look at test examples. The best checkpoint is restored at the end of each
training stage. The final evaluation still uses fully independent held-out
geometries. The original clustering logic, train/validation/test split
procedure, Lasso-style preprocessing and POD projection have not changed.

## Interpretation

If training L2 **and** validation L2 decrease, longer optimization can help.
If training improves but validation is flat/worse, focus on training-set
coverage and stability of near-degenerate eigenvectors. Increasing width
alone may overfit. A relative L2 error of 0.66 means 66% *error*, not 66%
accuracy. The old plots show a substantially better local than global NN,
but neither neural network is yet accurate enough for strong surrogate claims.

With 40 states at 49x49 grid, the highest states may be sensitive to grid
resolution and near-degenerate rotations; verify convergence and consider
subspace-invariant errors around nearly degenerate groups. Training changes
do **not** guarantee that the local system outperforms the global network.
