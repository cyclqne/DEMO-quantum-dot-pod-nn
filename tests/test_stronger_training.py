from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
import yaml
from qdpod.config import load_config
from qdpod.training import validation_wave_metrics
from qdpod.models import EigenpairMLP


def test_new_configs_keep_40_80_49_and_research_clustering():
    base=load_config('configs/demo_40state.yaml')
    for name in ('configs/train_longer_40state.yaml','configs/strong_40state.yaml'):
        new=load_config(name)
        assert new['physics']['grid']==49
        assert new['physics']['states']==40
        assert new['pod']['modes']==80
        assert new['clustering']==base['clustering']
        assert new['network']['restore_best_validation']
        assert new['network']['validation_every']>0
    assert load_config('configs/strong_40state.yaml')['physics']['random_options_per_well']==4


def test_fast_validation_l2_matches_direct_orthonormal_wave_reconstruction():
    # Orthonormal Phi, two example true wavefunctions + one arbitrary coefficient prediction.
    torch.manual_seed(1)
    phi=np.linalg.qr(np.random.default_rng(3).normal(size=(12,4)))[0].astype('float32')
    model=EigenpairMLP(3,4,6,layers=2,activation='tanh')
    x=torch.randn(2,3)
    true_psi=torch.randn(2,12)
    true_coeff=true_psi@torch.tensor(phi)
    scalers={'a_std':np.ones(4,dtype='float32'),'a_mean':np.zeros(4,dtype='float32'),
             'e_std':np.float32(1),'e_mean':np.float32(0),
             'clip_lo':np.full(4,-100,dtype='float32'),
             'clip_hi':np.full(4,100,dtype='float32')}
    energy=torch.ones(2)
    model.eval()
    quick,_=validation_wave_metrics(model,scalers,x,true_coeff,true_psi.square().sum(dim=1),energy,True)
    with torch.no_grad():
        a,_=model(x)
    wave=a@torch.tensor(phi).T
    exact=torch.minimum(torch.linalg.norm(true_psi-wave,dim=1),
                        torch.linalg.norm(true_psi+wave,dim=1)).div(torch.linalg.norm(true_psi,dim=1)).mean().item()
    assert abs(quick-exact)<2e-6
