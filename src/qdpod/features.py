"""Operator-eigenvalue/eigengap and four-dot geometry features from the research script."""
import math
import numpy as np
from .physics import anchors


def geometry_descriptors(params,cfg):
    p=cfg['physics']; wells=np.asarray(params,dtype=float).reshape(4,4)
    width=float(p['length_nm']); diag=width*np.sqrt(2.)
    r_min,r_max=p['radius_nm']; d_min,d_max=p['depth_ev']
    vals=[]
    for well,(cx,cy,r,d) in enumerate(wells):
        vals.extend((cx,cy,r,d))                   # physical values
        vals.extend((cx/width,cy/width,(r-r_min)/(r_max-r_min),
                     (d-d_min)/(d_max-d_min)))       # normalized values
        anchor=anchors(width)[well]; dx=cx-anchor[0];dy=cy-anchor[1]
        disp=np.hypot(dx,dy);theta=np.arctan2(dy,dx) if disp>1e-14 else 0.
        vals.extend((dx/width,dy/width,disp/diag,np.sin(theta),np.cos(theta)))
    centers=wells[:,:2]
    for i in range(4):
        for j in range(i+1,4):
            d=np.linalg.norm(centers[i]-centers[j]);vals.extend((d,d/diag))
    radii=wells[:,2];depths=wells[:,3];centroid=centers.mean(axis=0)
    vals.extend((np.mean(radii),np.std(radii),np.min(radii),np.max(radii),
                 np.mean(depths),np.std(depths),np.min(depths),np.max(depths),
                 centroid[0]/width,centroid[1]/width,
                 np.mean(np.linalg.norm(centers-centroid,axis=1))))
    return np.array(vals,dtype='float32')


def build_features(data,red,cfg):
    k=int(cfg['features']['spectral_eigenvalues']);eig=np.linalg.eigvalsh(red.astype('float64'))[:,:k]
    parts=[eig]
    if cfg['features']['eigengaps']:parts.append(np.diff(eig,axis=1))
    if cfg['features']['geometry_features']:
        parts.append(np.stack([geometry_descriptors(p,cfg) for p in data['params']]))
    geom=np.concatenate(parts,axis=1).astype('float32')
    ns=cfg['physics']['states']; ng=len(geom)
    f=np.broadcast_to(geom[:,None,:],(ng,ns,geom.shape[1]))
    if cfg['features']['state_features']:
        nums=np.arange(1,ns+1,dtype='float32')
        state=np.broadcast_to(np.stack([nums,nums/ns],axis=-1)[None,:,:],(ng,ns,2))
        f=np.concatenate([f,state],axis=-1)
    return f.astype('float32')
