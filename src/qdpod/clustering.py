"""Exact research-rule KMeans partitioning adapted to (geometry,state,mode) arrays.

Research source: final_lasso_cluster_40state_80mode_public.py, methods
compute_state_mode_signatures, validation_score_for_labels,
choose_validated_state_clusters, choose_width. Importantly NO latency term
in K selection. KMeans trained only on TRAIN signatures, and validated on
VALIDATION signatures. The test set is not used.
"""
import numpy as np
from sklearn.cluster import KMeans


def state_mode_signatures(coeffs,indices,cfg):
    c=cfg['clustering']; x=np.asarray(coeffs[indices],dtype='float64')
    if c['use_lasso_signatures']:
        a=float(c['lasso_alpha']);x=np.sign(x)*np.maximum(np.abs(x)-a,0.)
    energy=np.mean(x*x,axis=0)
    return energy/np.maximum(energy.sum(axis=1,keepdims=True),1e-12)


def remap_labels_by_state_order(labels):
    groups={int(lab):[] for lab in np.unique(labels)}
    for state,lab in enumerate(labels,start=1):groups[int(lab)].append(state)
    ordered=sorted(groups,key=lambda label:np.mean(groups[label]))
    remap={old:i for i,old in enumerate(ordered)}
    return np.array([remap[int(label)] for label in labels],dtype=int)


def validation_score_for_labels(train_sigs,val_sigs,labels,centers,cfg):
    settings=cfg['clustering']
    compact=float(np.mean(np.sum((val_sigs-centers[labels])**2,axis=1)))
    counts=np.array([sum(labels==lab) for lab in sorted(set(labels))],dtype=float)
    min_penalty=float(np.sum(np.maximum(0.,settings['soft_min_states']-counts)**2))
    max_penalty=float(np.sum(np.maximum(0.,counts-settings['max_states_target'])**2))
    imbalance=float(np.std(counts)/max(np.mean(counts),1e-12))
    score=(compact+settings['soft_min_penalty']*min_penalty+
           settings['soft_max_penalty']*max_penalty+settings['balance_penalty']*imbalance)
    return score,dict(validation_compactness=compact,soft_min_penalty_raw=min_penalty,
                      soft_max_penalty_raw=max_penalty,balance_penalty_raw=imbalance,total_score=score)


def choose_validated_state_clusters(train_sigs,val_sigs,cfg):
    """Identical objective, candidates, n_init, remapping, and tie rule as original."""
    settings=cfg['clustering']; rows=[];best=None
    for k in settings['candidate_k']:
        k_eff=min(int(k),len(train_sigs))
        if k_eff<1:continue
        model=KMeans(n_clusters=k_eff,random_state=int(settings['kmeans_seed']),
                     n_init=int(settings['n_init']))
        raw=model.fit_predict(train_sigs)
        labels=remap_labels_by_state_order(raw)
        centers=np.stack([train_sigs[labels==j].mean(axis=0) for j in range(k_eff)])
        score,parts=validation_score_for_labels(train_sigs,val_sigs,labels,centers,cfg)
        counts=[int(np.sum(labels==j)) for j in range(k_eff)]
        rows.append(dict(k=k_eff,counts=' '.join(map(str,counts)),min_count=min(counts),max_count=max(counts),**parts))
        if best is None or score<best[0]:best=(score,labels)
    if best is None:raise ValueError('No KMeans candidates')
    labels=best[1]
    for r in rows:r['selected']=abs(r['total_score']-best[0])<1e-15 and r['k']==len(np.unique(labels))
    return labels,rows


def cluster_state_groups(labels,train_sigs):
    rows=[]
    for i,l in enumerate(labels):
        sig=train_sigs[i]
        row=dict(state=i+1,expert=int(l),top_mode_by_signature=int(sig.argmax()+1),
                 top_mode_energy_fraction=float(sig.max()))
        for fraction,name in ((0.95,'95pct'),(0.99,'99pct'),(0.999,'99p9pct')):
            tot=np.cumsum(np.sort(sig)[::-1]);
            row[f'signature_modes_to_{name}_energy']=int(np.searchsorted(tot,fraction,side='left')+1)
        rows.append(row)
    return rows


def expert_width(num_states,cfg):
    sizes=cfg['network']['expert_widths']
    if num_states<=3:return int(sizes['small'])
    if num_states>=10:return int(sizes['very_large'])
    if num_states>=7:return int(sizes['large'])
    return int(sizes['default'])
