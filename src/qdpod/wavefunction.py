"""Signed quantum wavefunction reconstruction from POD modal coefficients.

    psi_POD(x,y) = sum_{j=1}^M a_j^POD * eta_j(x,y)
    psi_NN(x,y)  = sum_{j=1}^M a_hat_j^NN * eta_j(x,y)

The POD basis ``phi`` has shape (grid**2, modes), with each column
``phi[:,j]`` representing the same spatial mode eta_j(x,y) for every model.
The coefficient order is the order of those columns. The numerical solutions
are real here, so sign alignment is used ONLY for visual comparisons.
"""
from __future__ import annotations

import numpy as np


def modal_sum(phi: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Reconstruct signed psi from all retained POD modes, without rescaling.

    Parameters
    ----------
    phi : array, (pixels, modes)
        Orthonormal POD modes, eta_j(x,y), flattened in Fortran order.
    coefficients : array, (..., modes)
        Mode amplitudes a_j, such as POD-Galerkin eigenvectors or NN output.

    Returns
    -------
    psi : array, (..., pixels)
        Exact discrete modal sum sum_j coefficients[...,j] * phi[:,j].
    """
    basis = np.asarray(phi, dtype=np.float64)
    a = np.asarray(coefficients, dtype=np.float64)
    if basis.ndim != 2 or a.ndim < 1 or a.shape[-1] != basis.shape[1]:
        raise ValueError('Expected POD basis (pixels,modes) and coefficients (...,modes)')
    return np.einsum('...m,pm->...p', a, basis, optimize=True)


def pod_galerkin_amplitudes(reduced_hamiltonian: np.ndarray, nstates: int) -> np.ndarray:
    """Extract POD-Galerkin modal amplitudes directly from reduced eigensolve.

    This is not a projection of the DNS wavefunction, nor learned coefficients:
    it solves the same reduced operator used in the POD-Galerkin baseline.
    """
    matrix = np.asarray(reduced_hamiltonian, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError('Expected square reduced Hamiltonian')
    if not 0 < nstates <= matrix.shape[0]:
        raise ValueError('Invalid eigenstate count')
    _, vectors = np.linalg.eigh((matrix + matrix.T)*0.5)
    return vectors[:, :nstates].T


def align_display_sign(reference: np.ndarray, reconstructed: np.ndarray):
    """Return wavefunction with sign matched to DNS *for display only*.

    Does not change coefficients or modify the underlying saved predictions.
    """
    r = np.asarray(reference, dtype=np.float64)
    w = np.asarray(reconstructed, dtype=np.float64)
    if r.shape != w.shape:
        raise ValueError('Reference and reconstructed wavefunction shape mismatch')
    sign = -1 if float(np.vdot(r, w).real) < 0 else 1
    return sign*w, sign
