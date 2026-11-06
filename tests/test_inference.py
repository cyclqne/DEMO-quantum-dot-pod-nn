import numpy as np
import torch
from qdpod.inference import SequentialExperts,PackedExperts
from qdpod.models import EigenpairMLP,Bundle,save_bundle,load_bundle


def test_packed_equals_independent_adaptive_widths(tmp_path):
    torch.manual_seed(27)
    rng=np.random.default_rng(27)
    labels=np.array([0,0,1,1,1,2,2,2,2,3,3])
    bundles={}
    for key,w in enumerate([64,96,128,160]):
        model=EigenpairMLP(12,10,w,4,'tanh')
        sc=dict(x_mean=rng.normal(size=12).astype('float32'),
                x_std=(rng.random(12)+0.3).astype('float32'),
                a_mean=rng.normal(size=10).astype('float32'),
                a_std=(rng.random(10)+0.1).astype('float32'),
                e_mean=np.float32(1),e_std=np.float32(2),
                clip_lo=np.full(10,-3,'float32'),clip_hi=np.full(10,3,'float32'))
        bundles[key]=Bundle(model,sc)
    x=rng.normal(size=(3,11,12)).astype('float32')
    seq=SequentialExperts(bundles,labels)
    packed=PackedExperts(bundles,labels)
    for a,b in zip(seq.predict(x),packed.predict(x)):
        np.testing.assert_allclose(a,b,atol=2e-5,rtol=3e-5)
    fn=tmp_path/'expert.pt'
    save_bundle(bundles[0],fn)
    back=load_bundle(fn)
    a1,e1=bundles[0].model(torch.from_numpy(x.reshape(-1,12)))
    a2,e2=back.model(torch.from_numpy(x.reshape(-1,12)))
    assert torch.allclose(a1,a2)
    assert torch.allclose(e1,e2)
