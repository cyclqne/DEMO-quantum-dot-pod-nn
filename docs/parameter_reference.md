# Demo parameter reference

Edit **`configs/demo_parameters.yaml`** to change the next run; it is the default file read by `python main.py`. This reference describes every active YAML setting in one sentence.

## Experiment name

| Parameter | What it changes |
|---|---|
| `name` | Result-folder label for this experiment. |

## Random seed

| Parameter | What it changes |
|---|---|
| `seed` | Seed for model initialization, split sampling, and reproducibility. |

## Physics and four-dot geometries

| Parameter | What it changes |
|---|---|
| `physics.grid` | Number of grid points along each spatial direction (total degrees of freedom = grid squared). |
| `physics.length_nm` | Width and height of the simulated square domain in nanometers. |
| `physics.electron_mass_ratio` | Effective electron mass divided by the free-electron mass. |
| `physics.states` | Number of lowest eigenstates retained from each DNS solve. |
| `physics.random_options_per_well` | Independent samples per dot; with four dots the geometry count is this value to the fourth power. |
| `physics.center_perturb_nm` | Maximum random shift of each quantum-dot center in nanometers. |
| `physics.radius_nm` | Minimum and maximum Gaussian-dot radii in nanometers. |
| `physics.depth_ev` | Minimum and maximum Gaussian-dot depths in electronvolts. |
| `physics.geometry_seed` | Random seed for the four-dot parameter samples. |
| `physics.eigsh_tol` | Numerical convergence tolerance for the full-order sparse eigensolver. |

## Data split

| Parameter | What it changes |
|---|---|
| `splits.train` | Fraction of complete geometries used for model fitting and POD construction. |
| `splits.validation` | Fraction of geometries used for validation and expert-count selection. |
| `splits.test` | Fraction of geometries reserved for the final held-out comparison. |

## POD basis

| Parameter | What it changes |
|---|---|
| `pod.modes` | Number of POD basis modes retained for the neural outputs and reduced eigensolve. |
| `pod.oversampling` | Extra vectors used internally by randomized POD/SVD. |
| `pod.power_iterations` | Randomized-SVD power iterations, trading construction time for accuracy. |
| `pod.seed` | Random seed for POD basis construction. |

## Neural input features

| Parameter | What it changes |
|---|---|
| `features.spectral_eigenvalues` | Number of reduced Hamiltonian eigenvalues included in neural inputs (must not exceed POD modes). |
| `features.eigengaps` | Add energy gaps between consecutive reduced eigenvalues as features. |
| `features.geometry_features` | Include the Gaussian-dot geometry and potential parameters as features. |
| `features.state_features` | Include the eigenstate index/identity as neural input features. |

## Local-expert clustering

| Parameter | What it changes |
|---|---|
| `clustering.candidate_k` | Candidate numbers of local experts evaluated by validation KMeans when fixed_k is unset. |
| `clustering.use_lasso_signatures` | Soft-threshold POD coefficients before computing the state-clustering signatures. |
| `clustering.lasso_alpha` | Soft-threshold strength for the clustering signatures. |
| `clustering.kmeans_seed` | Random seed used in KMeans initialization. |
| `clustering.n_init` | Number of independent KMeans restarts per candidate expert count. |
| `clustering.soft_min_states` | Preferred minimum number of eigenstates assigned to each expert. |
| `clustering.max_states_target` | Preferred maximum eigenstates per expert before a penalty applies. |
| `clustering.soft_min_penalty` | Weight for clusters smaller than soft_min_states. |
| `clustering.soft_max_penalty` | Weight for clusters larger than max_states_target. |
| `clustering.balance_penalty` | Weight penalizing uneven numbers of eigenstates per expert. |

## Neural networks and training

| Parameter | What it changes |
|---|---|
| `network.hidden_layers` | Number of fully connected hidden layers in each neural network. |
| `network.activation` | Activation function: tanh, relu, or silu. |
| `network.global_width` | Number of hidden neurons per layer in the global network. |
| `network.expert_widths.small` | Hidden width for experts with few assigned eigenstates. |
| `network.expert_widths.default` | Hidden width for ordinary-sized expert groups. |
| `network.expert_widths.large` | Hidden width for larger expert groups. |
| `network.expert_widths.very_large` | Hidden width for the largest expert groups. |
| `network.batch_size` | Training examples processed per optimizer minibatch. |
| `network.fast_epochs` | Maximum epochs for coefficient/energy pretraining (stage 1). |
| `network.finetune_epochs` | Maximum epochs for wavefunction/physics fine-tuning (stage 2). |
| `network.learning_rate` | Initial optimizer learning rate for stage 1. |
| `network.finetune_learning_rate` | Initial optimizer learning rate for stage 2. |
| `network.grad_clip` | Maximum gradient norm used to stabilize optimization. |
| `network.pinn_warmup_epochs` | Warmup duration before reaching the scheduled physics-informed loss. |
| `network.pinn_ramp_epochs` | Duration of the physics-informed loss ramp. |
| `network.clip_predicted_coefficients` | Bound predicted coefficients to padded training-data ranges during inference. |
| `network.clip_padding_factor` | Padding around observed training coefficients when clipping is enabled. |
| `network.active_coeff_energy_fraction` | Cumulative POD coefficient energy used to define active targets. |
| `network.active_coeff_multiplier` | Relative weight/multiplier for active coefficient supervision. |
| `network.active_coeff_floor` | Minimum active-coefficient threshold. |
| `network.lambda_coeff` | Stage 1 coefficient reconstruction loss weight. |
| `network.lambda_active_coeff` | Stage 1 emphasis on active POD coefficient terms. |
| `network.lambda_energy` | Stage 1 relative energy prediction loss weight. |
| `network.lambda_pinn` | Stage 1 reduced Schrödinger-equation residual loss weight. |
| `network.lambda_reg` | Stage 1 neural output regularization weight. |
| `network.lambda_finetune_wave` | Stage 2 signed wavefunction reconstruction loss weight. |
| `network.lambda_finetune_density` | Stage 2 probability-density reconstruction loss weight. |
| `network.lambda_finetune_active_coeff` | Stage 2 active coefficient supervision weight. |
| `network.lambda_finetune_energy` | Stage 2 energy prediction loss weight. |
| `network.lambda_finetune_pinn` | Stage 2 physics-informed residual weight. |
| `network.weight_decay` | Optimizer weight decay used for regularization. |
| `network.lr_schedule` | Learning-rate schedule (cosine or constant). |
| `network.validation_every` | Validate model performance every N epochs. |
| `network.restore_best_validation` | Reload the lowest-validation-error network weights after training. |
| `network.validation_patience` | Maximum number of validation checks without improvement before stopping. |
| `network.validation_min_delta` | Minimum validation improvement treated as significant. |
| `network.gauge_align_coefficients` | Orient coefficient signs consistently using training-only POD anchors. |
| `network.lambda_gauge_supervision` | Extra coefficient supervision weight for gauge-aligned targets. |

## Runtime and figures

| Parameter | What it changes |
|---|---|
| `runtime.cpu_threads` | PyTorch CPU threads used for training and preprocessing. |
| `runtime.infer_thread_candidates` | CPU thread counts compared for neural inference. |
| `runtime.use_packed_experts` | Enable optimized batching of local experts at prediction time. |
| `runtime.benchmark_repetitions` | Repeated latency measurements per test geometry, summarized using medians. |
| `runtime.reconstruction_geometries` | How many held-out geometries receive detailed wavefunction figures. |
| `runtime.reconstruction_states` | One-based eigenstate indices included in the detailed wavefunction figures. |

## Optional controls

| Parameter | What it changes |
|---|---|
| `clustering.fixed_k` | Uncomment `fixed_k: 4` under clustering to force four local networks instead of automatically choosing K. |
| `network.expert_width_state_thresholds.small_max` | Largest state-group size using the `small` expert width (default 3). |
| `network.expert_width_state_thresholds.large_min` | Smallest group size using the `large` expert width (default 7). |
| `network.expert_width_state_thresholds.very_large_min` | Smallest group size using the `very_large` expert width (default 10). |
| `runtime.reconstruction_3d` | Uncomment and set to `false` to skip the signed 3D wavefunction figures. |

## Usage

1. Change the values in `configs/demo_parameters.yaml`.
2. Run `python main.py` from the folder containing `main.py`.
3. Look in `results/<name>/<timestamp>/` for the figures and CSVs.

For an optional fixed expert count, change the commented `# fixed_k: null` to `fixed_k: 4`; use `fixed_k: null` or comment it out again to restore automatic validation-based selection.

**Constraints:** `physics.states <= pod.modes <= physics.grid ** 2`; the number of spectral input eigenvalues cannot exceed the number of POD modes, and reconstruction state indices must lie between 1 and `physics.states`. The train/validation/test fractions must sum to one.

Because the number of geometries is `physics.random_options_per_well ** 4`, increasing it significantly increases the dataset generation and training workload.

Changing any active config field changes the run fingerprint; use a new output folder instead of resuming older checkpoints.
