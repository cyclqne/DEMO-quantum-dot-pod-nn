# Actual signed POD wavefunction reconstruction

The numerical field plotted as a wavefunction is **not the density**. It is the
signed real eigenfunction, reconstructed from the 80 POD basis functions
computed on training geometries. The same POD modes are used by each method.

For each spatial grid point `(x, y)` and eigenstate index `n`, the code uses

- POD-Galerkin: `psi_pod_n(x,y) = sum_{j=1}^{80} a_pod[n,j] * eta_j(x,y)`
- Global network: `psi_global_n(x,y) = sum_{j=1}^{80} a_global[n,j] * eta_j(x,y)`
- Local network: `psi_local_n(x,y) = sum_{j=1}^{80} a_local[n,j] * eta_j(x,y)`

Here `eta_j` is column `j` of `data/pod.npz -> phi`, flattened in
**Fortran order**. The POD coefficient vector `a_pod` comes from the
**reduced-Hamiltonian eigenvalue problem**, not from projecting the DNS
wavefunction. The NN coefficients are outputs of the trained models.

The implementation in `src/qdpod/wavefunction.py` is a vectorized equivalent
of `sum(a[j]*phi[:,j] for j in range(80))`. Both this identity and the reduced
eigenvector consistency are tested by `tests/test_modal_wavefunction_plots.py`.

## Output figures

Each selected held-out geometry and eigenstate receives four figures:

1. `reconstruction_geometry_XXXX_state_YY.png`: signed wavefunctions in the
   first row and densities `|psi|²` in the second row; DNS / POD / global / local
2. `wavefunction_sum_a_eta_geometry_XXXX_state_YY.png`: signed 2D wavefunction
   fields, signed residual maps, and a horizontal wavefunction cross-section
3. `wavefunction_3d_geometry_XXXX_state_YY.png`: 3D signed wavefunction surfaces
4. `wavefunction_modal_contributions_geometry_XXXX_state_YY.png`: modal terms
   `a_j * eta_j`, all remaining modal terms, and the complete 80-mode sum for
   POD and the local neural network

Across methods, the signed fields have a common symmetric color scale. For
visualization only, each predicted field is multiplied by +1 or -1 to match
the DNS orientation; the original coefficient vectors are **not** changed.
The density is calculated only for its own separate row.

## Verify a plotted field directly

Each plot has a matching file under
`data/wavefunction_reconstructions/wavefunction_geometry_XXXX_state_YY.npz`.
The arrays `a_pod`, `a_global`, `a_local`, `psi_pod`, `psi_global`,
`psi_local`, and `psi_dns` correspond to a specific geometry/state. The raw
psi arrays are **unmodified**, before display sign alignment.

```python
import numpy as np

with np.load("data/pod.npz") as f:
    phi = f["phi"]  # (49*49 pixels, 80 POD modes)
with np.load("data/wavefunction_reconstructions/wavefunction_geometry_XXXX_state_YY.npz") as f:
    a_pod = f["a_pod"]
    a_local = f["a_local"]
    psi_pod = (phi @ a_pod).reshape((49, 49), order="F")
    psi_local = (phi @ a_local).reshape((49, 49), order="F")
    assert np.allclose(psi_pod.ravel(order="F"), f["psi_pod"], atol=1e-6)
    assert np.allclose(psi_local.ravel(order="F"), f["psi_local"], atol=1e-6)
```

Substitute a real `XXXX` and `YY` from the run's output files.

## Generate these figures without retraining

From the project folder containing `main.py`, with the same configuration
that produced the saved run:

```powershell
$python = "$env:USERPROFILE\Desktop\quantum-dot-pod-nn-repository\.venv\Scripts\python.exe"
$run = Get-ChildItem ".\results\tanh_40state_80mode_49grid" -Directory |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
& $python main.py --config configs/demo_parameters.yaml --plots-only --output $run.FullName
```

The plots-only command loads the saved DNS, POD basis, reduced Hamiltonians,
geometry split, KMeans assignments, and global/local checkpoints. It does not
rerun DNS, train the models, or recompute timings or accuracy metrics.
