"""Sequential and packed adaptive-width local expert inference on CPU."""
from time import perf_counter
import numpy as np
import torch
from torch import nn
from .models import model_predict


class SequentialExperts:
    def __init__(self,bundles,labels,clip=True):
        self.bundles=bundles;self.labels=np.asarray(labels);self.clip=clip
    def predict(self,features):
        ng,ns,f=features.shape;nm=next(iter(self.bundles.values())).model.coeff_head.out_features
        a=np.empty((ng,ns,nm),dtype='float32');en=np.empty((ng,ns),dtype='float32')
        for k,b in self.bundles.items():
            states=np.flatnonzero(self.labels==k)
            x=features[:,states,:].reshape(-1,f)
            pred,energy=model_predict(b,x,clip=self.clip)
            a[:,states,:]=pred.reshape(ng,len(states),nm)
            en[:,states]=energy.reshape(ng,len(states))
        return a,en


class PackedExperts:
    """Zero-padded batched MLPs; equivalent to independently trained experts."""
    def __init__(self,bundles,labels,clip=True):
        self.clip=clip;self.labels=np.asarray(labels,dtype=int)
        self.keys=sorted(bundles)
        if set(self.keys)!=set(np.unique(labels)):
            raise ValueError('Every state must be assigned to exactly one expert')
        nets=[bundles[k].model.cpu().eval() for k in self.keys]
        layer_lists=[[x for x in net.backbone if isinstance(x,nn.Linear)] for net in nets]
        activations=[[type(x).__name__.lower() for x in net.backbone if isinstance(x,(nn.Tanh,nn.ReLU,nn.SiLU))] for net in nets]
        if not all(a==activations[0] for a in activations):raise ValueError('Expert activations/depths differ')
        self.act=activations[0][0]
        self.nstates=len(labels);self.mode_count=nets[0].coeff_head.out_features
        self.nfeat=layer_lists[0][0].in_features
        self.state_ids=[np.flatnonzero(self.labels==k) for k in self.keys]
        self.max_states=max(map(len,self.state_ids));K=len(self.keys)
        lookup=np.zeros((K,self.max_states),dtype='int64')
        for j,ids in enumerate(self.state_ids):lookup[j,:len(ids)]=ids
        self.lookup=torch.tensor(lookup)
        self.scatter_k=torch.tensor([j for j,ids in enumerate(self.state_ids) for _ in ids],dtype=torch.long)
        self.scatter_local=torch.tensor([i for ids in self.state_ids for i in range(len(ids))],dtype=torch.long)
        self.scatter_states=torch.tensor(np.concatenate(self.state_ids),dtype=torch.long)
        def stat(key):
            return torch.stack([torch.as_tensor(bundles[k].scalers[key],dtype=torch.float32) for k in self.keys])
        for key in ('x_mean','x_std','a_mean','a_std','e_mean','e_std','clip_lo','clip_hi'):
            setattr(self,key,stat(key))
        self.W=[];self.b=[]
        with torch.no_grad():
            for l in range(len(layer_lists[0])):
                maxin=max(seq[l].in_features for seq in layer_lists)
                maxout=max(seq[l].out_features for seq in layer_lists)
                weights=torch.zeros((K,maxout,maxin),dtype=torch.float32)
                biases=torch.zeros((K,maxout),dtype=torch.float32)
                for j,seq in enumerate(layer_lists):
                    lin=seq[l];weights[j,:lin.out_features,:lin.in_features]=lin.weight
                    biases[j,:lin.out_features]=lin.bias
                self.W.append(weights);self.b.append(biases)
            width=max(seq[-1].out_features for seq in layer_lists)
            self.head_w=torch.zeros(K,self.mode_count+1,width,dtype=torch.float32)
            self.head_b=torch.zeros(K,self.mode_count+1,dtype=torch.float32)
            for j,net in enumerate(nets):
                n=net.coeff_head.in_features
                self.head_w[j,:self.mode_count,:n]=net.coeff_head.weight
                self.head_w[j,self.mode_count,:n]=net.energy_head.weight.reshape(-1)
                self.head_b[j,:self.mode_count]=net.coeff_head.bias
                self.head_b[j,self.mode_count]=net.energy_head.bias.reshape(())

    @torch.inference_mode()
    def predict(self,features):
        ng,ns,nf=np.shape(features);K=len(self.keys)
        if ng==0:return np.empty((0,ns,self.mode_count),'float32'),np.empty((0,ns),'float32')
        x=torch.from_numpy(np.ascontiguousarray(features,dtype=np.float32))
        x=x[:,self.lookup].permute(1,0,2,3).contiguous().reshape(K,ng*self.max_states,nf)
        x=(x-self.x_mean[:,None,:])/self.x_std[:,None,:]
        for w,b in zip(self.W,self.b):
            x=torch.bmm(x,w.transpose(1,2))+b[:,None,:]
            if self.act=='tanh':x=torch.tanh(x)
            elif self.act=='relu':x=torch.relu(x)
            elif self.act=='silu':x=torch.nn.functional.silu(x)
        out=(torch.bmm(x,self.head_w.transpose(1,2))+self.head_b[:,None,:]).reshape(K,ng,self.max_states,self.mode_count+1)
        coeff=out[...,:self.mode_count]*self.a_std[:,None,None,:]+self.a_mean[:,None,None,:]
        if self.clip:coeff=coeff.clamp(min=self.clip_lo[:,None,None,:],max=self.clip_hi[:,None,None,:])
        energy=out[...,self.mode_count]*self.e_std[:,None,None]+self.e_mean[:,None,None]
        a=torch.empty((ng,ns,self.mode_count),dtype=torch.float32)
        e=torch.empty((ng,ns),dtype=torch.float32)
        a[:,self.scatter_states]=coeff[self.scatter_k,:,self.scatter_local].permute(1,0,2)
        e[:,self.scatter_states]=energy[self.scatter_k,:,self.scatter_local].permute(1,0)
        return a.numpy(),e.numpy()


def benchmark(func,features,repeats=12,warmups=4):
    for _ in range(warmups):func(features)
    vals=[]
    for _ in range(repeats):
        t=perf_counter();func(features);vals.append((perf_counter()-t)*1000)
    return float(np.median(vals))


def select_backend(bundles,labels,val_features,cfg):
    from .models import Bundle
    seq=SequentialExperts(bundles,labels,clip=cfg['network']['clip_predicted_coefficients'])
    pack=PackedExperts(bundles,labels,clip=cfg['network']['clip_predicted_coefficients'])
    # Equivalence audit on validation only; never inspect test set for selection.
    left=seq.predict(val_features);right=pack.predict(val_features)
    if not all(np.allclose(a,b,rtol=5e-5,atol=1e-5) for a,b in zip(left,right)):
        diff=max(float(abs(a-b).max()) for a,b in zip(left,right))
        raise AssertionError(f'Packed expert forward mismatch: {diff}')
    audits=[]
    for threads in sorted(set(cfg['runtime']['infer_thread_candidates'])):
        torch.set_num_threads(threads)
        for name,pred in [('sequential',seq),('packed',pack)]:
            ms=benchmark(pred.predict,val_features,repeats=cfg['runtime']['benchmark_repetitions'])
            audits.append(dict(backend=name,threads=threads,median_forward_ms=ms))
    best=min(audits,key=lambda row:row['median_forward_ms'])
    torch.set_num_threads(best['threads'])
    return (pack if best['backend']=='packed' else seq),audits,best
