import numpy as np
from qdpod.config import load_config
from qdpod.physics import kinetic_matrix,potential_vector,generate_geometries,solve_one
from qdpod.pod import split_geometries
from qdpod.features import geometry_descriptors


def test_default_dimensions_and_source_sampling():
    cfg=load_config('configs/demo_40state.yaml')
    assert cfg['physics']['grid']==49
    assert cfg['physics']['states']==40
    assert cfg['pod']['modes']==80
    p=generate_geometries(cfg)
    assert p.shape==(81,16)
    for j in range(4):
        assert len(np.unique(p[:,4*j:4*j+4],axis=0))==3
    v=potential_vector(p[0],cfg)
    assert v.shape==(2401,)
    assert v.min()<0
    assert len(geometry_descriptors(p[0],cfg))==75


def test_periodic_kinetic_hermiticity_and_lowest_states():
    cfg=load_config('configs/smoke_test.yaml')
    k=kinetic_matrix(cfg)
    assert k.shape==(289,289)
    assert np.max(abs((k-k.T).data),initial=0)<1e-12
    e,psi,_=solve_one(k,generate_geometries(cfg)[0],cfg)
    assert e.shape==(6,)
    assert psi.shape==(289,6)
    assert np.all(np.diff(e)>=-1e-10)
    assert np.allclose(psi.T@psi,np.eye(6),atol=1e-4)


def test_split_geometries_no_leakage():
    cfg=load_config('configs/demo_40state.yaml')
    ids=split_geometries(81,cfg)
    assert [len(ids[k]) for k in ('train','validation','test')]==[56,12,13]
    combined=np.concatenate(list(ids.values()))
    assert len(np.unique(combined))==81
