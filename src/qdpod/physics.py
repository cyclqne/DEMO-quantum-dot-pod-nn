"""Four-QD periodic finite-difference Schrödinger model in electron-volts.

The same Gaussian anchors, per-well Cartesian-product sampling, effective mass,
16 nm square and periodic kinetic stencil as the original Python QEM port are
used. This repo uses 49 points per axis: dx=16/49 nm for a periodic grid.
"""
from itertools import product
from time import perf_counter
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh

HBAR=1.054571726e-34
ME=9.10938215e-31
E_CHARGE=1.60218e-19


def anchors(length=16.):
    return np.array([[length/3,length/3], [length/3,2*length/3],
                     [2*length/3,length/3],[2*length/3,2*length/3]],dtype=float)


def generate_geometries(cfg):
    p=cfg['physics']; rng=np.random.default_rng(p['geometry_seed']); q=p['random_options_per_well']
    options=[]
    for cx,cy in anchors(p['length_nm']):
        a=rng.uniform(0,2*np.pi,q)
        r=p['center_perturb_nm']*np.sqrt(rng.uniform(0,1,q))
        x=cx+r*np.cos(a); y=cy+r*np.sin(a)
        widths=rng.uniform(*p['radius_nm'],q)
        depths=rng.uniform(*p['depth_ev'],q)
        options.append(np.column_stack((x,y,widths,depths)))
    return np.asarray([np.concatenate(entry) for entry in product(*options)], dtype=np.float64)


def kinetic_matrix(cfg):
    p=cfg['physics']; n=p['grid']; dx_nm=p['length_nm']/n
    prefactor=HBAR**2/(2*ME*p['electron_mass_ratio']*(dx_nm*1e-9)**2*E_CHARGE)
    grid=np.arange(n*n,dtype=int).reshape(n,n,order='F')
    at=grid.ravel(order='F')
    rows=np.concatenate([at]*5)
    cols=np.concatenate([at]+[np.roll(grid,shift,axis=axis).ravel(order='F')
                                for axis,shift in ((0,-1),(0,1),(1,-1),(1,1))])
    vals=np.concatenate([np.full(n*n,4*prefactor)]+[np.full(n*n,-prefactor)]*4)
    return coo_matrix((vals,(rows,cols)),shape=(n*n,n*n)).tocsr()


def potential_vector(params,cfg):
    p=cfg['physics']; n=p['grid']; length=p['length_nm']
    x=np.arange(n)*length/n
    X,Y=np.meshgrid(x,x,indexing='xy')
    v=np.zeros((n,n),dtype=np.float64)
    for cx,cy,r,d in np.asarray(params).reshape(4,4):
        v-=d*np.exp(-((X-cx)**2+(Y-cy)**2)/(2*r*r))
    return v.ravel(order='F')


def solve_one(kinetic,params,cfg):
    p=cfg['physics']; potential=potential_vector(params,cfg)
    h=kinetic+diags(potential,format='csr')
    t=perf_counter()
    ev,vecs=eigsh(h,k=p['states'],which='SA',tol=float(p['eigsh_tol']))
    sec=perf_counter()-t
    order=np.argsort(ev); ev=ev[order];vecs=vecs[:,order]
    # Robust sign convention. Dense near-degenerate subspaces still have arbitrary rotations.
    imax=np.argmax(abs(vecs),axis=0)
    flip=np.sign(vecs[imax,np.arange(len(ev))]); flip[flip==0]=1
    vecs=vecs*flip
    return ev.astype('float64'),vecs.astype('float32'),float(sec)


def generate_dns(cfg,verbose=True):
    geo=generate_geometries(cfg); kin=kinetic_matrix(cfg)
    ng=len(geo); ns=cfg['physics']['states']; n=cfg['physics']['grid']**2
    e=np.empty((ng,ns),dtype=np.float64)
    psi=np.empty((ng,ns,n),dtype=np.float32)
    sec=np.empty(ng,dtype=np.float64)
    for i,param in enumerate(geo):
        e[i],waves,sec[i]=solve_one(kin,param,cfg)
        psi[i]=waves.T
        if verbose and (i%max(1,ng//12)==0 or i+1==ng):
            print(f'  DNS {i+1}/{ng} geometries, last solve {sec[i]:.3f}s',flush=True)
    return dict(params=geo, energies=e, psi=psi, dns_s=sec)
