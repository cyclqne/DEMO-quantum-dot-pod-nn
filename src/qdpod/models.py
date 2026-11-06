"""Separate wavefunction-coefficient and eigenenergy output heads."""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn


class EigenpairMLP(nn.Module):
    def __init__(self,input_size,modes,width,layers=4,activation='tanh'):
        super().__init__()
        acts={'tanh':nn.Tanh,'relu':nn.ReLU,'silu':nn.SiLU}
        if activation not in acts:raise ValueError('Unsupported activation')
        net=[];in_f=input_size
        for _ in range(layers):
            net.extend([nn.Linear(in_f,width),acts[activation]()]);in_f=width
        self.backbone=nn.Sequential(*net)
        self.coeff_head=nn.Linear(width,modes)
        self.energy_head=nn.Linear(width,1)
    def forward(self,x):
        h=self.backbone(x)
        return self.coeff_head(h),self.energy_head(h).squeeze(-1)


@dataclass
class Bundle:
    model: EigenpairMLP
    scalers: dict


def create_scalers(x,y,e,cfg):
    x_mean=x.mean(axis=0);x_std=x.std(axis=0)
    a_mean=y.mean(axis=0);a_std=y.std(axis=0)
    x_std=np.where(x_std>1e-8,x_std,1.0)
    a_std=np.where(a_std>1e-8,a_std,1.0)
    e_mean=np.float32(e.mean());e_std=np.float32(max(e.std(),1e-8))
    pad=float(cfg['network']['clip_padding_factor'])*(y.max(axis=0)-y.min(axis=0))
    return dict(x_mean=x_mean.astype('float32'),x_std=x_std.astype('float32'),
                a_mean=a_mean.astype('float32'),a_std=a_std.astype('float32'),
                e_mean=e_mean,e_std=e_std,
                clip_lo=(y.min(axis=0)-pad).astype('float32'),
                clip_hi=(y.max(axis=0)+pad).astype('float32'))


def model_predict(bundle,features,clip=True):
    x=np.asarray(features,dtype='float32')
    s=bundle.scalers
    x=(x-s['x_mean'])/s['x_std']
    bundle.model.eval()
    with torch.inference_mode():
        z,e=bundle.model(torch.from_numpy(x))
    a=z.numpy()*s['a_std']+s['a_mean']
    en=e.numpy()*s['e_std']+s['e_mean']
    if clip:a=np.clip(a,s['clip_lo'],s['clip_hi'])
    return a.astype('float32'),en.astype('float32')


def save_bundle(bundle,path):
    m=bundle.model
    h=[l for l in m.backbone if isinstance(l,nn.Linear)]
    state={k:v.detach().cpu() for k,v in m.state_dict().items()}
    act=next(type(l).__name__.lower() for l in m.backbone if isinstance(l,(nn.Tanh,nn.SiLU,nn.ReLU)))
    torch.save(dict(state=state,scalers={k:torch.as_tensor(v) for k,v in bundle.scalers.items()},architecture=dict(
        input_size=h[0].in_features,modes=m.coeff_head.out_features,width=h[0].out_features,
        layers=len(h),activation=act)),path)


def load_bundle(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    model=EigenpairMLP(**saved['architecture'])
    model.load_state_dict(saved['state']);model.eval()
    return Bundle(model,{k:v.detach().cpu().numpy() for k,v in saved['scalers'].items()})
