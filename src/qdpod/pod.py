"""Train-only POD basis, reduced Hamiltonians and projected eigenstate labels."""
import numpy as np
from sklearn.utils.extmath import randomized_svd
from .physics import potential_vector


def split_geometries(ng,cfg):
    rng=np.random.default_rng(cfg['seed']); ids=rng.permutation(ng)
    ntr=int(cfg['splits']['train']*ng); nval=int(cfg['splits']['validation']*ng)
    if min(ntr,nval,ng-ntr-nval)<1: raise ValueError('Need nonempty train, validation and test geometry splits')
    return dict(train=np.sort(ids[:ntr]), validation=np.sort(ids[ntr:ntr+nval]),
                test=np.sort(ids[ntr+nval:]))


def construct_pod(data,splits,cfg):
    phi=data['psi'][splits['train']].reshape(-1,cfg['physics']['grid']**2).T
    m=cfg['pod']['modes']
    if m>min(phi.shape): raise ValueError(f'Requested {m} POD modes but only {min(phi.shape)} train-snapshot dimensions')
    u,s,_=randomized_svd(phi,n_components=m,n_oversamples=int(cfg['pod']['oversampling']),
                       n_iter=int(cfg['pod']['power_iterations']),random_state=int(cfg['pod']['seed']))
    # Sign orientation of modes is arbitrary but deterministic for fixed seed.
    coeff=np.einsum('gsi,im->gsm',data['psi'],u,optimize=True).astype('float32')
    return u.astype('float32'),coeff,s.astype('float32')


def reduced_hamiltonians(data,phi,cfg):
    from .physics import kinetic_matrix
    t=kinetic_matrix(cfg)
    modes=phi.astype('float64')
    t_red=modes.T@(t@modes)
    H=np.empty((len(data['params']),modes.shape[1],modes.shape[1]),dtype='float32')
    for i,params in enumerate(data['params']):
        v=potential_vector(params,cfg)
        pot=modes.T@(v[:,None]*modes)
        mat=t_red+pot
        H[i]=((mat+mat.T)/2).astype('float32')
    return H


def pod_galerkin(data,phi,red):
    """Return projection-based reduced eigenvectors/energies on each geometry."""
    ns=data['energies'].shape[1]
    eig=np.linalg.eigh(red.astype('float64'))
    e=eig.eigenvalues[:,:ns]
    a=eig.eigenvectors[:,:,:ns].transpose(0,2,1)
    wave=np.einsum('gsm,im->gsi',a,phi,optimize=True)
    return e, wave
