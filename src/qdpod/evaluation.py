"""Held-out accuracy and comparable end-to-end online CPU latency."""
from time import perf_counter
import numpy as np
from .models import model_predict
from .physics import kinetic_matrix,potential_vector
from .features import build_features
from .pod import pod_galerkin


def predict_global(bundle,features,cfg):
    n,s,d=features.shape
    a,e=model_predict(bundle,features.reshape(-1,d),clip=cfg['network']['clip_predicted_coefficients'])
    return a.reshape(n,s,-1),e.reshape(n,s)


def errors(true,pred,energy_true=None,energy_pred=None):
    norm=np.maximum(np.linalg.norm(true,axis=-1),1e-12)
    e1=np.linalg.norm(true-pred,axis=-1)/norm
    e2=np.linalg.norm(true+pred,axis=-1)/norm
    wave=np.minimum(e1,e2)
    density=np.linalg.norm(true*true-pred*pred,axis=-1)/np.maximum(np.linalg.norm(true*true,axis=-1),1e-12)
    if energy_true is None:energy=np.full_like(wave,np.nan)
    else:energy=np.abs(energy_true-energy_pred)/np.maximum(np.abs(energy_true),1e-12)
    return wave,density,energy


def reduced_operator_setup(phi,cfg):
    """One-time POD operator setup, shared by the NN and POD online paths."""
    kin=kinetic_matrix(cfg)
    m=phi.astype('float64')
    return m,m.T@(kin@m)


def projected_hamiltonian_one(phi_double,t_red,params,cfg):
    pot=potential_vector(params,cfg)
    h=t_red+phi_double.T@(pot[:,None]*phi_double)
    return (h+h.T)*0.5


def compute_feature_overhead(m,t_red,params,cfg):
    """Time potential + projected operator + eigengap + geometry construction."""
    start=perf_counter()
    h=projected_hamiltonian_one(m,t_red,params,cfg)
    _=build_features({'params':np.asarray(params)[None,:]},h[None,:,:],cfg)
    return perf_counter()-start


def compute_pod_online(m,t_red,params,cfg):
    """Time POD required path: projected operator + reduced eigensolve + reconstruction."""
    start=perf_counter()
    h=projected_hamiltonian_one(m,t_red,params,cfg)
    vals,vecs=np.linalg.eigh(h)
    ns=cfg['physics']['states']
    _=vecs[:,:ns].T@m.T
    return perf_counter()-start


def evaluate(data,features,phi,red,models,labels,split,cfg,local_pred):
    indices=split['test'];ft=features[indices]; gt=data['psi'][indices].astype('float32')
    energies=data['energies'][indices]
    pred_a,pred_e=predict_global(models['global'],ft,cfg)
    loc_a,loc_e=local_pred.predict(ft)
    gal_e,gal_wave=pod_galerkin({'energies':energies},phi,red[indices])
    method={'Global POD-NN':(pred_a,pred_e), 'Local POD-NN':(loc_a,loc_e)}
    records=[];table=[]
    for name,(a,e) in method.items():
        wave=np.einsum('gsm,im->gsi',a,phi,optimize=True)
        we,de,ee=errors(gt,wave,energies,e)
        records.append(dict(model=name,mean_wave_rel_l2=float(we.mean()),mean_density_rel_l2=float(de.mean()),
                            mean_energy_rel_abs=float(ee.mean())))
        for i,g in enumerate(indices):
            for j in range(gt.shape[1]):table.append(dict(model=name,geometry=int(g),state=j+1,
                wave_rel_l2=float(we[i,j]),density_rel_l2=float(de[i,j]),energy_rel_abs=float(ee[i,j])))
    we,de,ee=errors(gt,gal_wave,energies,gal_e)
    records.append(dict(model='POD-Galerkin',mean_wave_rel_l2=float(we.mean()),mean_density_rel_l2=float(de.mean()),
                        mean_energy_rel_abs=float(ee.mean())))
    for i,g in enumerate(indices):
        for j in range(gt.shape[1]):table.append(dict(model='POD-Galerkin',geometry=int(g),state=j+1,
            wave_rel_l2=float(we[i,j]),density_rel_l2=float(de[i,j]),energy_rel_abs=float(ee[i,j])))
    return records,table


def benchmark_online(data,features,phi,red,models,indices,cfg,local_predictor):
    """Online = projected-operator feature construction + NN + reconstruction.

    DNS is direct-solver compute only. POD is reduced build + eigensolve + reconstruction.
    Shared feature generation is measured rather than silently omitted for NN methods.
    """
    rows=[];ns=cfg['physics']['states'];repeats=int(cfg['runtime']['benchmark_repetitions'])
    m,t_red=reduced_operator_setup(phi,cfg)
    for idx in indices:
        f=features[idx:idx+1]
        feature_times=[]
        for _ in range(max(3,min(5,repeats))):
            feature_times.append(compute_feature_overhead(m,t_red,data['params'][idx],cfg))
        feat_ms=float(np.median(feature_times))*1000
        for name,fn in [('Global POD-NN',lambda:predict_global(models['global'],f,cfg)),
                        ('Local POD-NN',lambda:local_predictor.predict(f))]:
            for _ in range(2):fn()
            forward=[];reconstruct=[]
            for _ in range(repeats):
                start=perf_counter();coeff,_=fn(); mid=perf_counter()
                _=coeff@phi.T;end=perf_counter()
                forward.append((mid-start)*1000);reconstruct.append((end-mid)*1000)
            fw=float(np.median(forward));rec=float(np.median(reconstruct))
            rows.append(dict(geometry=int(idx),model=name,feature_ms=feat_ms,forward_ms=fw,
                             reconstruction_ms=rec,total_online_ms=feat_ms+fw+rec))
        # POD total: its *own* required reduced Hamiltonian construction,
        # reduced eigensolve, and wavefunction reconstruction. Do not charge
        # it for NN-only operator-feature eigengap creation.
        pod_s=[compute_pod_online(m,t_red,data['params'][idx],cfg)*1000
               for _ in range(max(3,repeats))]
        solve_ms=float(np.median(pod_s))
        rows.append(dict(geometry=int(idx),model='POD-Galerkin',feature_ms=0.,forward_ms=solve_ms,
                         reconstruction_ms=0.,total_online_ms=solve_ms))
        rows.append(dict(geometry=int(idx),model='DNS (solve only)',feature_ms=0.,
                         forward_ms=float(data['dns_s'][idx])*1000,reconstruction_ms=0.,
                         total_online_ms=float(data['dns_s'][idx])*1000))
    return rows
