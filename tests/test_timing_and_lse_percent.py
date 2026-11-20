"""Regression tests for the two POD timing definitions and the supplied LSE formula."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import numpy as np
import pandas as pd
import pytest

from qdpod.evaluation import percent_lse, measure_pod_online, reduced_operator_setup
from qdpod.plotting import _timing_matrix, make_timing_detail_figures
from qdpod.config import load_config


def test_percentage_matches_discrete_integral_formula():
    true=np.array([[1.,2.,3.], [2.,-2.,1.]])
    pred=np.array([[1.,1.,3.],[1.,-2.,0.]])
    direct=100.*np.sum((true-pred)**2,axis=1)/np.sum(true**2,axis=1)
    np.testing.assert_allclose(percent_lse(true,pred,sign_invariant=False),direct)
    # Equispaced area integration multiplies numerator and denominator equally.
    dxdy=.025
    weighted=100.*np.sum((true-pred)**2*dxdy,axis=1)/np.sum(true**2*dxdy,axis=1)
    np.testing.assert_allclose(direct,weighted)


def test_global_wavefunction_sign_is_immaterial_not_shape_or_scale():
    psi=np.array([[1.,0.,1.,-1.]])
    np.testing.assert_allclose(percent_lse(psi,-psi),[0.])
    np.testing.assert_allclose(percent_lse(psi,-psi,sign_invariant=False),[400.])
    np.testing.assert_allclose(percent_lse(psi,0.5*psi),[25.])
    # The squared ratio, not the ordinary relative L2 norm times 100.
    np.testing.assert_allclose(percent_lse(psi,np.zeros_like(psi)),[100.])
    with pytest.raises(ValueError):percent_lse(psi,psi[..., :1])


def test_pod_full_is_assembly_plus_eigensolve_plus_reconstruction():
    cfg=load_config('configs/smoke_test.yaml')
    rng=np.random.default_rng(40)
    n=cfg['physics']['grid']**2
    m=cfg['pod']['modes']
    phi=np.linalg.qr(rng.normal(size=(n,m)))[0]
    basis,t_red=reduced_operator_setup(phi,cfg)
    params=np.array([[5.2,5.4,1.5,2.2], [5.3,10.7,1.5,2.2],
                     [10.6,5.4,1.5,2.2], [10.7,10.5,1.5,2.2]])
    times=measure_pod_online(basis,t_red,params,cfg)
    assert all(np.isfinite(v) and v>=0 for v in times.values())
    assert times['pod_full_online_ms']>=times['pod_eigensolve_only_ms']
    np.testing.assert_allclose(times['pod_full_online_ms'],
        times['pod_assembly_ms']+times['pod_eigensolve_only_ms']+times['pod_reconstruction_ms'],
        atol=1e-8)


def test_only_one_pod_timing_figure_and_old_plots_are_cleaned(tmp_path):
    rows=[]
    for g in (3,9,12):
        for model,ms in [('DNS (solve only)',6.),('POD-Galerkin',1.1),
                         ('Global POD-NN',1.5),('Local POD-NN',1.7)]:
            r=dict(geometry=g,model=model,total_online_ms=ms+g*.015,
                   forward_ms=.18)
            if model=='POD-Galerkin':
                r.update(pod_assembly_ms=.65,pod_eigensolve_only_ms=.24,
                         pod_reconstruction_ms=.21,
                         pod_full_online_ms=ms+g*.015)
            rows.append(r)
    df=pd.DataFrame(rows)
    # Numeric full-online timing still exists and differs from solve-only.
    full=_timing_matrix(df,'full_online')
    solve=_timing_matrix(df,'eigensolve_only')
    assert (full['POD-Galerkin']>solve['POD-Galerkin']).all()
    for name in ('DNS (solve only)','Global POD-NN','Local POD-NN'):
        np.testing.assert_allclose(full[name],solve[name])
    figs=tmp_path/'figures';figs.mkdir()
    old_files=('online_time_by_geometry_all_methods_pod_full_online.png',
               'online_time_by_geometry_all_methods_pod_eigensolve_only.png',
               'online_time_by_geometry_reduced_methods_pod_full_online.png',
               'pod_vs_nn_timing_by_geometry_pod_eigensolve_only.png')
    for name in old_files:
        (figs/name).write_bytes(b'legacy')
    make_timing_detail_figures(tmp_path,df)
    kept='online_time_by_geometry_reduced_methods_pod_eigensolve_only.png'
    assert (figs/kept).stat().st_size>1000
    assert [p.name for p in figs.glob('*.png')]==[kept]
    assert all(not (figs/f).exists() for f in old_files)
