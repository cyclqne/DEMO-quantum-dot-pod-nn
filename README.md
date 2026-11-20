# Quantum Dot POD–Neural Network

A self-contained Python project that compares **global neural networks**, **local neural networks**, and **POD-Galerkin** for predicting quantum states in a four-quantum-dot system.

The program generates its own quantum-dot geometries, solves the Schrödinger equation, builds a reduced-order POD basis, trains the neural networks, and produces accuracy and runtime comparisons. It runs entirely on the CPU.

## What the project does

1. Generates randomized configurations of four quantum dots.
2. Solves for their quantum-state wavefunctions and energies.
3. Builds a Proper Orthogonal Decomposition (POD) basis from training data.
4. Trains a global neural network and KMeans-selected local expert networks.
5. Evaluates each method on unseen quantum-dot geometries.
6. Saves comparison graphs, numerical results, and plots of signed wavefunctions and densities.

## Default experiment

| Setting | Value |
|---|---|
| Spatial grid | 49 × 49 |
| Quantum states | 40 per geometry |
| POD modes | 80 |
| Quantum-dot geometries | 256 |
| Neural-network activation | Tanh |
| Hidden layers | 4 |
| Global network width | 160 |
| Local networks | Automatically clustered experts |
| Training | Up to 850 coefficient-training epochs + 200 reconstruction-training epochs |
| Data split | 70% training / 15% validation / 15% testing |
| Hardware | CPU |

## Customize the demo

All run settings are centralized in **[`configs/demo_parameters.yaml`](configs/demo_parameters.yaml)**. Open that file, change the values, save it, and run `python main.py` (or the PowerShell command below); no Python source edits are required.

| Setting | YAML parameter | Explanation |
|---|---|---|
| Grid size | `physics.grid` | Number of grid points along each spatial direction. |
| Eigenstates | `physics.states` | Number of quantum states predicted per geometry. |
| POD modes | `pod.modes` | Reduced-basis size and number of NN coefficients. |
| Number of geometries | `physics.random_options_per_well` | Four-dot sampling uses this number to the fourth power (4 gives 256 geometries). |
| Quantum-dot variations | `physics.center_perturb_nm`, `physics.radius_nm`, `physics.depth_ev` | Control how strongly the dot positions, sizes, and depths vary. |
| Number of local networks | `clustering.fixed_k` | Uncomment and set `fixed_k: 4` for exactly four KMeans experts; leave it unset for automatic selection. |
| Automatic expert selection | `clustering.candidate_k` | Candidate values used when `fixed_k` is not set. |
| Global and expert model sizes | `network.global_width`, `network.expert_widths`, `network.hidden_layers` | Set the width and depth of the neural models. |
| Activation function | `network.activation` | Choose `tanh`, `relu`, or `silu`. |
| Training epochs | `network.fast_epochs`, `network.finetune_epochs` | Upper bounds on the two training stages. |
| CPU and timing | `runtime.cpu_threads`, `runtime.benchmark_repetitions` | Control CPU threading and timing repetitions. |
| Wavefunction plots | `runtime.reconstruction_states`, `runtime.reconstruction_geometries` | Choose which eigenstates and how many held-out geometries receive reconstruction figures. |

Additional controls for sampling, eigensolvers, POD construction, train/test splits, input features, clustering penalties, loss weights, schedules, early stopping, and inference are described **in the same YAML file**, with a one-sentence explanation for every setting. The [complete parameter reference](docs/parameter_reference.md) lists all 77 active settings and optional controls.

**When changing state count:** keep `pod.modes >= physics.states`, keep `features.spectral_eigenvalues <= pod.modes`, and adjust `runtime.reconstruction_states` to valid state indices. When changing parameters, start a new run; don't resume a checkpoint produced with a different configuration.

You can run a different preset explicitly with `python main.py --config configs/smoke_test.yaml`.

## Getting started

### Requirements

- Python 3.12
- Windows, macOS, or Linux
- Python packages listed in `requirements.txt`

### Installation

Clone the repository and open a terminal inside the folder containing `main.py`:

```bash
git clone https://github.com/cyclqne/quantum-dot-pod-nn.git
cd quantum-dot-pod-nn
```

Create an environment and install the packages. **Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**macOS or Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Run the experiment

On Windows (PowerShell):

```powershell
.\.venv\Scripts\python.exe main.py
```

On macOS or Linux, with the environment activated:

```bash
python main.py
```

The program will generate the dataset, train the networks, evaluate their predictions, and save the figures and results. Training time depends on the CPU.

## Results

Each run creates a new folder under:

```text
results/tanh_40state_80mode_49grid/<run_timestamp>/
├── figures/       # Plots and colored reconstruction images
├── metrics/       # Accuracy, training, clustering, and timing CSVs
├── models/        # Trained neural-network checkpoints
├── data/          # Generated quantum-dot and POD data
└── run_summary.json
```

### Figures

The figures include the selected comparisons below; numeric timing components remain available in `metrics/pod_timing_breakdown.csv` and `metrics/timings.csv`.

| Figure | Description |
|---|---|
| `density_accuracy.png` | Global and local NN density LSE (%) |
| `energy_accuracy.png` | Global and local NN relative energy error (%) |
| `per_state_wave_lse_percent.png` | Sign-aligned wavefunction LSE (%) for each eigenstate, NN models only |
| `accuracy_vs_online_latency_by_geometry.png` | Wavefunction LSE (%) against total NN online runtime |
| `reconstruction_geometry_XXXX_state_YY.png` | Signed wavefunction and density maps for DNS, POD-Galerkin, global and local NN; colorbars are on the left |
| `wavefunction_sum_a_eta_geometry_XXXX_state_YY.png` | Complete signed wavefunction modal sum, DNS residual maps and spatial cross-section |
| `wavefunction_3d_geometry_XXXX_state_YY.png` | 3D wavefunction surfaces |
| `wavefunction_modal_contributions_geometry_XXXX_state_YY.png` | Individual `a_j η_j` and the full 80-mode sum |
| `online_time_by_geometry_reduced_methods_pod_eigensolve_only.png` | The **only** POD timing comparison: reduced eigensolve-only time alongside complete global/local NN online times for each test geometry |
| `training_loss_by_model.png` | Stage-wise training objective |
| `validation_wave_error_by_model.png` | Sign-aligned validation wavefunction LSE (%) |
| `kmeans_selection.png` | Local-expert clustering results |

Only one POD timing comparison graph is generated. Full POD online timings remain available in the CSV files but are **not plotted**. 

### Accuracy definition

The displayed wavefunction least-squares error uses the integrated squared
residual normalized by DNS wavefunction energy, multiplied by 100:

```text
Wavefunction LSE (%) = 100 × ∫ |ψ_model(r) − ψ_DNS(r)|² d²r
                             / ∫ |ψ_DNS(r)|² d²r
```

On the uniform 2-D grid, the cell-area factor cancels from the numerator and
denominator. Real eigenvectors can be multiplied by −1 without changing the
physical state, so wavefunction LSE is computed after choosing the global sign
of each predicted state that minimizes the numerator (sign-invariant LSE).
**The percentage is a squared relative error**, not an accuracy percentage or
the ordinary relative L2 norm times 100. The same squared normalization is
used for the density LSE; energy uses mean absolute relative error × 100.

Accuracy charts display only Global and Local POD-NN. POD-Galerkin accuracy
remains in CSV tables, but no POD accuracy score or residual map is displayed
in the figures. Its reconstructed signed wavefunction is still shown alongside
DNS and neural predictions for qualitative comparison.

### POD-Galerkin timing definitions

**Full POD online time:** potential evaluation and reduced Hamiltonian matrix
assembly + the 80×80 symmetric eigensolve + full wavefunction reconstruction
`ψ_POD = Σ a_j η_j` for the 40 retained eigenstates. It excludes offline POD
basis generation and construction of the precomputed reduced kinetic operator.

**Reduced eigensolve only:** exclusively the call to `numpy.linalg.eigh` on the
already assembled 80×80 reduced Hamiltonian. It excludes matrix assembly,
potential creation and wavefunction reconstruction.

Both POD timing definitions are measured for the CSV outputs, but only the
**reduced eigensolve-only** time is plotted. The single graph compares it with
the complete online timings of both neural networks, which include feature
construction, neural inference, and wavefunction reconstruction. This is an
explicit comparison of different computational scopes, not of equal full
online workloads. DNS sparse eigensolve timings are retained in the CSVs.

A per-geometry POD audit is saved as `metrics/pod_timing_breakdown.csv` in
addition to the main `metrics/timings.csv`.

### Rebuild plots without retraining

If you already have a completed run, keep its data and model checkpoints.
Run this command from the project folder (replace the example timestamp):

```powershell
.\.venv\Scripts\python.exe main.py --plots-only --output "results/tanh_40state_80mode_49grid/<run_timestamp>"
```

This refreshes held-out accuracy tables, both POD timing measurements in CSV,
and the single reduced-eigensolve comparison figure **without retraining**.
It also removes retired full-online and duplicate POD timing plots left by
older versions of the repository.

### Wavefunction reconstruction

The plots compare the **wavefunction itself**, not only its probability density.
Both reduced models use the same POD basis functions `η_j(x, y)` (saved as columns
of `phi` in `data/pod.npz`). The POD-Galerkin coefficients come from the
reduced Hamiltonian eigenproblem. The global and local neural networks predict
separate coefficient vectors. Every reconstructed field uses **all 80 modes**:

```text
POD-Galerkin:  psi_POD(x,y)   = sum[j=1..80] a_POD[j]   * eta[j](x,y)
Global NN:     psi_global(x,y)= sum[j=1..80] a_global[j]* eta[j](x,y)
Local NN:      psi_local(x,y) = sum[j=1..80] a_local[j] * eta[j](x,y)
```

The first reference panel is the DNS wavefunction. All color maps display
**signed amplitudes** with a common scale; the density row displays `|psi|²`.
For visualization only, each approximate wavefunction's overall sign is aligned
to DNS (real eigenfunctions may differ by an arbitrary sign). No prediction is
modified during evaluation or training.

For each plotted geometry and eigenstate, the raw coefficients and **unaligned**
wavefunction vectors are saved under `data/wavefunction_reconstructions/` as
`.npz` files. This allows direct verification of the modal summation.

The default output includes states 1, 10, 20, and 40 on two held-out geometries.
Change these in `configs/demo_parameters.yaml` under
`runtime.reconstruction_states` and `runtime.reconstruction_geometries`.


### Numerical results

The `metrics/` folder contains CSV files such as:

- `summary.csv` — overall accuracy, including wavefunction/density LSE (%) and relative energy error (%).
- `per_state_errors.csv` — per-state, per-geometry LSE percentages and original relative norms.
- `timings.csv` — per-geometry timings, including the POD full online and eigensolve-only columns.
- `pod_timing_breakdown.csv` — POD matrix assembly, eigensolve, reconstruction and full online timing for every test geometry.
- `training_history.csv` — training and validation history.
- `cluster_validation_scores.csv` — KMeans expert-selection results.

## Project structure

```text
quantum-dot-pod-nn/
├── main.py                 # Run the full experiment
├── configs/                # Experiment settings
├── src/qdpod/              # Physics, POD, clustering, and NN code
├── tests/                  # Automated tests
├── docs/                   # Additional technical documentation
├── requirements.txt        # Python dependencies
└── README.md
```

To run the automated tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The demo uses `configs/demo_parameters.yaml` by default. Every adjustable parameter is explained beside its value, and the complete reference is in [Parameter reference](docs/parameter_reference.md).
