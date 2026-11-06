import numpy as np
from qdpod.clustering import (state_mode_signatures,choose_validated_state_clusters,
    validation_score_for_labels,expert_width,remap_labels_by_state_order)
from qdpod.config import load_config


def test_exact_original_expert_width_thresholds():
    cfg=load_config('configs/demo_40state.yaml')
    assert [(n,expert_width(n,cfg)) for n in (1,3,4,6,7,9,10,18)] == [
        (1,64),(3,64),(4,96),(6,96),(7,128),(9,128),(10,160),(18,160)]


def test_exact_research_cluster_score_formula():
    cfg=load_config('configs/demo_40state.yaml')
    rng=np.random.default_rng(11)
    tr=rng.random((40,80));va=tr+rng.normal(0,0.01,size=tr.shape)
    labels=np.repeat([0,1,2,3],[2,16,20,2]);centers=np.stack([tr[labels==k].mean(axis=0) for k in range(4)])
    score,parts=validation_score_for_labels(tr,va,labels,centers,cfg)
    counts=np.array([2,16,20,2]);c=np.mean(np.sum((va-centers[labels])**2,axis=1))
    assert np.isclose(score,c+0.02*(20-18)**2+0.005*counts.std()/counts.mean())
    assert parts['soft_min_penalty_raw']==0
    assert parts['soft_max_penalty_raw']==4.


def test_state_signatures_train_only_and_lasso():
    cfg=load_config('configs/demo_40state.yaml')
    data=np.ones((4,40,80),dtype='float32')*0.1
    data[2:]*=100
    sig=state_mode_signatures(data,np.array([0,1]),cfg)
    baseline=state_mode_signatures(data[:2],np.array([0,1]),cfg)
    assert np.allclose(sig,baseline)
    assert np.allclose(sig.sum(axis=1),1)
    data[:,0,0]=0.00001
    sig2=state_mode_signatures(data,np.array([0,1]),cfg)
    assert sig2[0,0]==0


def test_validation_select_40_state_candidates_and_sorted_labels():
    cfg=load_config('configs/demo_40state.yaml')
    rng=np.random.default_rng(3)
    tr=rng.random((40,80));va=tr+rng.normal(0,.015,tr.shape)
    labels,rows=choose_validated_state_clusters(tr,va,cfg)
    assert len(rows)==7
    assert {r['k'] for r in rows}==set(range(2,9))
    assert 2<=len(np.unique(labels))<=8
    assert labels.min()==0
    assert len([r for r in rows if r['selected']])==1
    means=[np.flatnonzero(labels==i).mean() for i in np.unique(labels)]
    assert means==sorted(means)
