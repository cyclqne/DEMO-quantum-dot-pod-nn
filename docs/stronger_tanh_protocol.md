# Strong-tanh research demo: protocol and validation criteria

## Precommitted configuration

- 49×49 periodic grid, 16 nm domain, four independent Gaussian potential wells.
- 40 eigenstates, 80 train-only POD modes, 256 random geometries (q=4 per dot).
- 70:15:15 split of **entire geometries**, seed 42.
- Original modal-energy KMeans K=2…8 selection with Lasso-style alpha 1e-4;
  *no test coefficient access* and *no runtime penalty* in clustering.
- Global tanh: 4×160 hidden units; local tanh experts: 4 layers, adaptively
  96/128/160/192 hidden units, according to the original cluster-size thresholds.
- 850 maximum coefficient epochs + 200 maximum full-field epochs, two
  independently cosine-scheduled AdamW phases, validation every 10 epochs.
- Full-space sign-invariant validation L2 checkpoint selected across both
  stages; early stop based on the validation history. No test-set selection.

## Additional training modification

A POD coefficient sign anchor for each eigenstate is selected by the largest
mean *absolute* coordinate in the **training** split. This anchor orients
all coefficient/wave targets (including validation labels), without changing
physical wavefunctions up to the arbitrary global sign. The training loss
retains its original sign-invariant coefficient/full-space, density, energy,
PINN residual, and sparse active-mode terms, with a new optional direct
orientation-supervised coefficient penalty of weight 0.20.

The original DNS/POD caches remain unmodified by the gauge step, which is
recomputed deterministically on each run. Sign orientation is not a solution
to near-degenerate subspace rotation; compare projectors if that dominates
high eigenstate errors.

## What constitutes a successful demo?

`metrics/goal_check.csv` records diagnostic thresholds only:

1. Mean local sign-invariant relative wave L2 < 0.20.
2. Mean local relative energy error < 0.02.
3. Local total online latency divided by global total online latency ≤ 1.20.
4. Local wave error lower than global wave error.

No algorithm guarantees these thresholds. They are evaluated on a test split
not used for training. To make a credible research claim, repeat with
several random seeds and check grid resolution, K selection stability,
architecture parameter counts, and per-state wave/subspace errors.

## Timing protocol and plotting

For each held-out geometry, `metrics/timings.csv` stores DNS solve-only,
POD-Galerkin *complete online* path, and global and local NN *complete online*
paths. NN time includes operator-feature formation, network inference,
reconstruction, and all required per-geometry steps. It excludes offline
model fitting and initial POD-basis construction for all models.

Timing figures show each geometry ID without averaging away outliers. An
all-method log-scale figure prevents DNS dominance from hiding the NN and
POD curves; linear-scale figures isolate the smaller online times. A
local/global per-geometry latency ratio figure makes overhead explicit.

Fixed state IDs (1, 10, 20, 40) and the first two held-out geometries are
used in wavefunction/density reconstruction panels. DNS, POD-Galerkin,
Global POD-NN and Local POD-NN use the same row-specific color ranges. No
hand-selected "good" reconstruction case is substituted.

## Computational limitation

The 850+200 configuration with 256 geometries and 40 states per geometry
requires significantly more CPU work than the old 160+40, 81-geometry demo.
The provided integration checks do not prove performance after convergence;
a completed full experiment and independent repeatability checks are needed.
