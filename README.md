# Quantum Dot POD–Neural Network

A self-contained Python project that compares **global neural networks**, **local neural networks**, and **POD-Galerkin** for predicting quantum states in a four-quantum-dot system.

The program generates its own quantum-dot geometries, solves the Schrödinger equation, builds a reduced-order POD basis, trains the neural networks, and produces accuracy and runtime comparisons. It runs entirely on the CPU.

## What the project does

1. Generates randomized configurations of four quantum dots.
2. Solves for their quantum-state wavefunctions and energies.
3. Builds a Proper Orthogonal Decomposition (POD) basis from training data.
4. Trains a global neural network and KMeans-selected local expert networks.
5. Evaluates each method on unseen quantum-dot geometries.
6. Saves comparison graphs, numerical results, and colored wavefunction reconstructions.

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

The main figures include:

| Figure | Description |
|---|---|
| `wavefunction_accuracy.png` | Mean wavefunction reconstruction error by method |
| `density_accuracy.png` | Probability-density reconstruction error |
| `energy_accuracy.png` | Predicted energy error |
| `reconstruction_geometry_XXXX_state_YY.png` | Color maps comparing DNS, POD-Galerkin, global NN, and local NN wavefunctions and densities |
| `online_time_by_geometry_all_methods.png` | Online runtime for every test geometry and method |
| `online_time_by_geometry_neural_networks.png` | Global vs. local network runtime for each geometry |
| `local_global_latency_ratio_by_geometry.png` | Local-to-global runtime ratio |
| `accuracy_vs_online_latency_by_geometry.png` | Accuracy and runtime for individual test geometries |
| `training_loss_by_model.png` | Training loss curves |
| `validation_wave_error_by_model.png` | Validation wavefunction error during training |
| `kmeans_selection.png` | Automatic local-expert clustering results |

The wavefunction reconstruction figures display signed wavefunctions and probability densities with consistent color scales, making the predictions easy to compare visually.

### Numerical results

The `metrics/` folder contains CSV files such as:

- `summary.csv` — overall model accuracy.
- `per_state_errors.csv` — errors for individual states and geometries.
- `timings.csv` — online runtime for each test geometry.
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

The experiment can be configured in `configs/default_40state_tanh.yaml`.
