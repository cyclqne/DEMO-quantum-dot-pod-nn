"""Held-out accuracy and explicitly itemized online CPU timings.

Wave LSE percent implements the user's discrete integral formula:
100 * integral |psi_pred - psi_DNS|^2 / integral |psi_DNS|^2,
with a global +/- eigenvector gauge alignment for real wavefunctions.
On the common uniform grid, the dx*dy factor cancels from this ratio.
"""
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


def percent_lse(reference, prediction, sign_invariant=True):
    """Per-state LSE %, with spatial pixels on the last axis.

    The integral weights cancel for equally spaced 2-D grid samples. Because
    real Schrödinger eigenvectors have arbitrary global phase (+/-), the lower
    of two *squared* residuals is used by default. No rescaling or fitting of
    the predicted field is permitted.
    """
    truth=np.asarray(reference,dtype=np.float64)
    pred=np.asarray(prediction,dtype=np.float64)
    if truth.shape != pred.shape:
        raise ValueError('Reference and prediction shapes must agree')
    denom=np.maximum(np.sum(truth**2,axis=-1),1e-24)
    err=np.sum((truth-pred)**2,axis=-1)
    if sign_invariant:
        err=np.minimum(err,np.sum((truth+pred)**2,axis=-1))
    return 100.*err/denom


def errors(true,pred,energy_true=None,energy_pred=None):
    # Retain raw relative L2 fields for continuity with established diagnostics.
    norm=np.maximum(np.linalg.norm(true,axis=-1),1e-12)
    e1=np.linalg.norm(true-pred,axis=-1)/norm
    e2=np.linalg.norm(true+pred,axis=-1)/norm
    wave=np.minimum(e1,e2)
    density=np.linalg.norm(true*true-pred*pred,axis=-1)/np.maximum(np.linalg.norm(true*true,axis=-1),1e-12)
    if energy_true is None:energy=np.full_like(wave,np.nan)
    else:energy=np.abs(energy_true-energy_pred)/np.maximum(np.abs(energy_true),1e-12)
    return wave,density,energy


def reduced_operator_setup(phi,cfg):
    """One-time POD operator setup; deliberately not timed online."""
    kin=kinetic_matrix(cfg)
    m=phi.astype('float64')
    return m,m.T@(kin@m)


def projected_hamiltonian_one(phi_double,t_red,params,cfg):
    pot=potential_vector(params,cfg)
    h=t_red+phi_double.T@(pot[:,None]*phi_double)
    return (h+h.T)*0.5


def compute_feature_overhead(m,t_red,params,cfg):
    """NN online geometry cost: potential, projected operator and feature build."""
    start=perf_counter()
    h=projected_hamiltonian_one(m,t_red,params,cfg)
    _=build_features({'params':np.asarray(params)[None,:]},h[None,:,:],cfg)
    return perf_counter()-start


def measure_pod_online(m,t_red,params,cfg):
    """Time every required POD online step and *isolate* the eigensolve.

    ``t_red`` and the POD basis are offline precomputations. Assembly includes
    geometry-specific potential creation and projected Hamiltonian assembly.
    Eigensolve is ONLY np.linalg.eigh(Hred), excluding matrix construction.
    Reconstruction is the dense modal sum using the resulting eigenvectors.
    The total is measured using the same outer timer, not just a sum of medians.
    """
    t0=perf_counter()
    h=projected_hamiltonian_one(m,t_red,params,cfg)
    t1=perf_counter()
    vals,vecs=np.linalg.eigh(h)
    t2=perf_counter()
    ns=int(cfg['physics']['states'])
    _=vecs[:,:ns].T@m.T
    t3=perf_counter()
    return dict(pod_assembly_ms=(t1-t0)*1000,
                pod_eigensolve_only_ms=(t2-t1)*1000,
                pod_reconstruction_ms=(t3-t2)*1000,
                pod_full_online_ms=(t3-t0)*1000)


def compute_pod_online(m,t_red,params,cfg):
    """Backward-compatible full online POD time in seconds."""
    return measure_pod_online(m,t_red,params,cfg)['pod_full_online_ms']/1000


def evaluate(data,features,phi,red,models,labels,split,cfg,local_pred):
    indices=split['test'];ft=features[indices];gt=data['psi'][indices].astype('float32')
    energies=data['energies'][indices]
    pred_a,pred_e=predict_global(models['global'],ft,cfg)
    loc_a,loc_e=local_pred.predict(ft)
    gal_e,gal_wave=pod_galerkin({'energies':energies},phi,red[indices])
    method={'Global POD-NN':(pred_a,pred_e), 'Local POD-NN':(loc_a,loc_e)}
    records=[];table=[]

    def record_method(name,wave,e):
        we,de,ee=errors(gt,wave,energies,e)
        w_pct=percent_lse(gt,wave,sign_invariant=True)
        d_pct=percent_lse(gt*gt,wave*wave,sign_invariant=False)
        records.append(dict(model=name,
             mean_wave_lse_pct=float(w_pct.mean()),
             mean_density_lse_pct=float(d_pct.mean()),
             mean_energy_rel_pct=float(100*ee.mean()),
             mean_wave_rel_l2=float(we.mean()),
             mean_density_rel_l2=float(de.mean()),
             mean_energy_rel_abs=float(ee.mean())))
        for i,g in enumerate(indices):
            for j in range(gt.shape[1]):
                table.append(dict(model=name,geometry=int(g),state=j+1,
                    wave_lse_pct=float(w_pct[i,j]),density_lse_pct=float(d_pct[i,j]),
                    energy_rel_pct=float(100*ee[i,j]),
                    wave_rel_l2=float(we[i,j]),density_rel_l2=float(de[i,j]),
                    energy_rel_abs=float(ee[i,j])))

    for name,(a,e) in method.items():
        wave=np.einsum('gsm,im->gsi',a,phi,optimize=True)
        record_method(name,wave,e)
    record_method('POD-Galerkin',gal_wave,gal_e)
    return records,table


def benchmark_online(data,features,phi,red,models,indices,cfg,local_predictor):
    """Measure POD as *full online* and *eigensolve-only* per held-out geometry.

    DNS is saved sparse-eigensolver time, not DNS assembly. NN online cost
    includes recomputed input-feature overhead + model + full-space recovery.
    POD full online is potential/assembly + eigensolve + reconstruction; POD
    eigensolve-only excludes assembly and reconstruction. Same 80x80 operator.
    """
    rows=[];repeats=int(cfg['runtime']['benchmark_repetitions'])
    m,t_red=reduced_operator_setup(phi,cfg)
    pod_fields=('pod_assembly_ms','pod_eigensolve_only_ms','pod_reconstruction_ms','pod_full_online_ms')
    for idx in indices:
        f=features[idx:idx+1]
        feature_times=[compute_feature_overhead(m,t_red,data['params'][idx],cfg)*1000
                       for _ in range(max(3,min(5,repeats)))]
        feat_ms=float(np.median(feature_times))
        for name,fn in [('Global POD-NN',lambda:predict_global(models['global'],f,cfg)),
                        ('Local POD-NN',lambda:local_predictor.predict(f))]:
            for _ in range(2):fn()
            forward=[];reconstruct=[]
            for _ in range(repeats):
                start=perf_counter();coeff,_=fn();mid=perf_counter()
                _=coeff@phi.T;end=perf_counter()
                forward.append((mid-start)*1000);reconstruct.append((end-mid)*1000)
            fw=float(np.median(forward));rec=float(np.median(reconstruct))
            rows.append(dict(geometry=int(idx),model=name,feature_ms=feat_ms,
                             forward_ms=fw,reconstruction_ms=rec,total_online_ms=feat_ms+fw+rec,
                             **{k:np.nan for k in pod_fields}))
        for _ in range(2):measure_pod_online(m,t_red,data['params'][idx],cfg)
        measurements=[measure_pod_online(m,t_red,data['params'][idx],cfg)
                      for _ in range(max(3,repeats))]
        p={key:float(np.median([r[key] for r in measurements])) for key in pod_fields}
        rows.append(dict(geometry=int(idx),model='POD-Galerkin',feature_ms=p['pod_assembly_ms'],
                         forward_ms=p['pod_eigensolve_only_ms'],
                         reconstruction_ms=p['pod_reconstruction_ms'],
                         total_online_ms=p['pod_full_online_ms'],**p))
        rows.append(dict(geometry=int(idx),model='DNS (solve only)',feature_ms=0.,
                         forward_ms=float(data['dns_s'][idx])*1000,reconstruction_ms=0.,
                         total_online_ms=float(data['dns_s'][idx])*1000,
                         **{k:np.nan for k in pod_fields}))
    return rows
