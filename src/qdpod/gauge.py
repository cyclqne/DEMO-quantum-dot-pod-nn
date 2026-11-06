"""Train-defined sign conventions for deterministic POD coefficient regression.

The sign of a real Schrödinger eigenvector is arbitrary. Here, a POD-mode
anchor is picked *only from training geometries* for each state index, and all
geometries are oriented to positive projection onto that mode. This changes no
wavefunction density or sign-invariant error. Near-degenerate subspaces may
still rotate: this is not a general degeneracy-tracking algorithm.
"""
from __future__ import annotations
import numpy as np


def align_pod_gauge(coeff, psi, train_ids):
    """Return oriented coefficient/wave labels and train-only anchor audit.

    Inputs: coeff (geometries, states, modes), psi (geometries, states, pixels).
    Output labels are copies; DNS and original POD caches are not modified.
    """
    a = np.asarray(coeff)
    waves = np.asarray(psi)
    if a.ndim != 3 or waves.shape[:2] != a.shape[:2]:
        raise ValueError('Expected matching (geometry,state,*) coefficient and wave arrays')
    # Mode with largest average absolute coefficient among TRAIN geometries.
    anchor_modes = np.mean(np.abs(a[train_ids]), axis=0).argmax(axis=1)
    selected = np.take_along_axis(a, anchor_modes[None, :, None], axis=2)[:, :, 0]
    signs = np.where(selected < 0, -1.0, 1.0).astype(a.dtype)
    oriented_coeff = (a * signs[:, :, None]).astype(np.float32)
    oriented_waves = (waves * signs[:, :, None]).astype(np.float32)
    audit = [dict(state=j+1, anchor_mode=int(anchor_modes[j])+1,
                  sign_flips_train=int((signs[train_ids,j]<0).sum()),
                  sign_flips_all=int((signs[:,j]<0).sum())) for j in range(a.shape[1])]
    return oriented_coeff, oriented_waves, audit
