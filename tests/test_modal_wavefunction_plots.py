"""Physics-to-plot reproducibility: wavefunctions are sums of POD modes."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

import numpy as np
import pytest

from qdpod.wavefunction import (modal_sum, pod_galerkin_amplitudes,
                                align_display_sign)
from qdpod.plotting import make_reconstruction_figures


def test_modal_sum_matches_explicit_a_times_eta_and_is_signed():
    rng = np.random.default_rng(21)
    pixels, modes = 25, 6
    phi = np.linalg.qr(rng.normal(size=(pixels,modes)))[0]
    a = rng.normal(size=(2,3,modes))
    expected = np.empty((2,3,pixels))
    for g in range(2):
        for s in range(3):
            expected[g,s] = sum(a[g,s,j] * phi[:,j] for j in range(modes))
    np.testing.assert_allclose(modal_sum(phi,a),expected,rtol=1e-13,atol=1e-13)
    assert (modal_sum(phi,a) < 0).any()  # This is ψ, not |ψ|².
    with pytest.raises(ValueError):
        modal_sum(phi,np.zeros((1,modes+1)))


def test_pod_galerkin_coefficients_reconstruct_reduced_eigenfunction():
    rng=np.random.default_rng(10)
    phi=np.linalg.qr(rng.normal(size=(49,5)))[0]
    q=np.linalg.qr(rng.normal(size=(5,5)))[0]
    h=q @ np.diag([-2.0,-1.0,0.5,2.0,4.0]) @ q.T
    amplitudes=pod_galerkin_amplitudes(h,3)
    reconstructed=modal_sum(phi,amplitudes)
    for k in range(3):
        np.testing.assert_allclose(phi.T @ reconstructed[k], amplitudes[k], atol=1e-13)
        np.testing.assert_allclose(h @ amplitudes[k],
                                   np.linalg.eigvalsh(h)[k]*amplitudes[k],atol=1e-13)
    oriented,sign=align_display_sign(reconstructed[0],-reconstructed[0])
    assert sign==-1
    np.testing.assert_allclose(oriented,reconstructed[0])


def test_plots_export_modal_coefficients_and_real_signed_fields(tmp_path,monkeypatch):
    import qdpod.features
    import qdpod.evaluation
    rng=np.random.default_rng(1)
    grid,states,modes=7,3,4
    phi=np.linalg.qr(rng.normal(size=(grid*grid,modes)))[0].astype('float32')
    red=np.diag([-3.,-2.,-1.,1.]).reshape(1,modes,modes).astype('float32')
    apod=pod_galerkin_amplitudes(red[0],states)
    psi=modal_sum(phi,apod).astype('float32')
    data={'params':np.zeros((1,16)), 'psi':psi[None]}
    gcoeff=apod[None].astype('float32')*.9
    lcoeff=apod[None].astype('float32')*.95
    monkeypatch.setattr(qdpod.features,'build_features',lambda *args,**kwargs:np.zeros((1,states,8),dtype='float32'))
    monkeypatch.setattr(qdpod.evaluation,'predict_global',lambda *args,**kwargs:(gcoeff,np.zeros((1,states))))
    class Local:
        def predict(self,x):return lcoeff,np.zeros((1,states))
    cfg={'physics':{'grid':grid,'states':states,'length_nm':16.},
         'runtime':{'reconstruction_geometries':1,'reconstruction_states':[1,3],
                    'reconstruction_3d':True}}
    make_reconstruction_figures(tmp_path,data,phi,red,{'global':None},None,
                                {'test':np.array([0])},cfg,Local())
    figures=tmp_path/'figures'
    for state in (1,3):
        prefix=f'geometry_0000_state_{state:02d}'
        for name in (f'reconstruction_{prefix}.png',
                     f'wavefunction_sum_a_eta_{prefix}.png',
                     f'wavefunction_3d_{prefix}.png',
                     f'wavefunction_modal_contributions_{prefix}.png'):
            assert (figures/name).stat().st_size>7000, name
        with np.load(tmp_path/'data'/'wavefunction_reconstructions'/f'wavefunction_{prefix}.npz') as fields:
            expected=modal_sum(phi,fields['a_local'])
            np.testing.assert_allclose(fields['psi_local'],expected,rtol=1e-6,atol=1e-6)
            np.testing.assert_allclose(fields['psi_pod'],modal_sum(phi,fields['a_pod']),atol=1e-6)
            assert fields['n_modes']==modes
