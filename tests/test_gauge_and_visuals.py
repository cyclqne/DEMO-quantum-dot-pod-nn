from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd

from qdpod.config import load_config
from qdpod.gauge import align_pod_gauge
from qdpod.clustering import state_mode_signatures
from qdpod.plotting import make_timing_detail_figures


def test_850_200_tanh_default_preserves_research_clustering():
    old = load_config('configs/demo_40state.yaml')
    new = load_config('configs/default_40state_tanh.yaml')
    assert (new['physics']['grid'],new['physics']['states'],new['pod']['modes'])==(49,40,80)
    assert new['network']['fast_epochs']==850
    assert new['network']['finetune_epochs']==200
    assert new['network']['activation']=='tanh'
    assert new['physics']['random_options_per_well']==4
    assert old['clustering']==new['clustering']
    assert new['network']['gauge_align_coefficients']
    assert new['network']['lambda_gauge_supervision']>0


def test_train_only_gauge_is_sign_invariant_and_no_test_anchor_leakage():
    rng=np.random.default_rng(3)
    raw=rng.normal(size=(9,4,7)).astype('float32')
    psi=rng.normal(size=(9,4,21)).astype('float32')
    tr=np.array([0,1,2,3,4])
    a,w,audit=align_pod_gauge(raw,psi,tr)
    assert len(audit)==4
    np.testing.assert_allclose(np.abs(a),np.abs(raw))
    np.testing.assert_allclose(w**2,psi**2)
    # All original sign-invariant clustering signatures must remain unchanged.
    cfg=load_config('configs/default_40state_tanh.yaml')
    np.testing.assert_allclose(state_mode_signatures(raw,tr,cfg),
                               state_mode_signatures(a,tr,cfg),rtol=1e-6,atol=1e-8)
    assert all(np.all(a[:,j,row['anchor_mode']-1]>=0) for j,row in enumerate(audit))
    other=raw.copy();other[5:]*=1000
    _,_,audit_other=align_pod_gauge(other,psi,tr)
    assert [x['anchor_mode'] for x in audit]==[x['anchor_mode'] for x in audit_other]


def test_timing_plots_are_per_geometry(tmp_path):
    (tmp_path/'figures').mkdir()
    models=['DNS (solve only)','POD-Galerkin','Global POD-NN','Local POD-NN']
    rows=[]
    for g in (2,5,9):
        for j,m in enumerate(models):
            row=dict(geometry=g,model=m,total_online_ms=(g+1)*(j+1)/20,
                     forward_ms=(g+1)*(j+1)/60)
            if m=='POD-Galerkin':
                row.update(pod_eigensolve_only_ms=.035*(g+1),
                           pod_assembly_ms=.02*(g+1),pod_reconstruction_ms=.01*(g+1),
                           pod_full_online_ms=(g+1)*(j+1)/20)
            rows.append(row)
    make_timing_detail_figures(tmp_path,pd.DataFrame(rows))
    figs=tmp_path/'figures'
    # Exactly one POD timing figure: reduced eigensolve-only vs both NNs.
    kept = 'online_time_by_geometry_reduced_methods_pod_eigensolve_only.png'
    assert (figs / kept).stat().st_size > 1000
    assert len(list(figs.glob('*.png'))) == 1
    forbidden = [
        'online_time_by_geometry_all_methods_pod_full_online.png',
        'online_time_by_geometry_all_methods_pod_eigensolve_only.png',
        'online_time_by_geometry_reduced_methods_pod_full_online.png',
        'pod_timing_components_by_geometry.png',
        'pod_vs_nn_timing_by_geometry_pod_full_online.png',
        'online_time_by_geometry_neural_networks.png',
        'nn_forward_time_by_geometry.png',
    ]
    assert all(not (figs / filename).exists() for filename in forbidden)


def test_reconstruction_plot_does_not_display_pod_lse_annotation():
    from pathlib import Path
    source=(Path(__file__).resolve().parents[1]/'src/qdpod/plotting.py').read_text()
    assert "elif j == 1:\n                    caption = ''" in source
    assert "elif j==1:" in source  # Residual panel disabled for POD in modal-sum comparison.
