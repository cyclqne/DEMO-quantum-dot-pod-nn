"""Global and original-size-policy local CPU MLP training on the same split."""
import numpy as np
import copy
import torch
from torch.utils.data import DataLoader,TensorDataset
from .models import EigenpairMLP,Bundle,create_scalers,model_predict
from .clustering import expert_width


def sign_loss(true,pred):
    den=(true*true).sum(dim=1)+1e-12
    same=((true-pred)**2).sum(dim=1)/den
    flipped=((true+pred)**2).sum(dim=1)/den
    return torch.minimum(same,flipped).mean()


def active_coeff_loss(true,pred,cfg):
    n=cfg['network'];eps=1e-12
    frac=true.square()/(true.square().sum(dim=1,keepdim=True)+eps)
    sorted_f,idx=torch.sort(frac,dim=1,descending=True)
    previous=torch.cumsum(sorted_f,dim=1)-sorted_f
    active_sorted=(previous<float(n['active_coeff_energy_fraction'])).to(frac.dtype)
    mask=torch.zeros_like(frac).scatter(1,idx,active_sorted)
    weights=mask*torch.rsqrt(frac+float(n['active_coeff_floor']))
    weights=1.+float(n['active_coeff_multiplier'])*weights/(weights.mean(dim=1,keepdim=True)+eps)
    denom=(weights*true.square()).sum(dim=1)+eps
    same=(weights*(true-pred).square()).sum(dim=1)/denom
    flipped=(weights*(true+pred).square()).sum(dim=1)/denom
    return torch.minimum(same,flipped).mean()


def pinn_loss(a,e,hred):
    left=torch.bmm(hred,a.unsqueeze(-1)).squeeze(-1)
    right=e[:,None]*a
    return ((left-right).square().sum(dim=1)/(right.square().sum(dim=1)+1e-12)).mean()


def energy_loss(true,pred):
    return (abs(true-pred)/(abs(true)+1e-12)).mean()


def validation_wave_metrics(model, scalers, features, true_coeff, true_wave_sqnorm,
                            true_energies, clip=True):
    """Validation-only sign-invariant full-space L2 error, cheaply in POD space.

    For an orthonormal POD basis Phi, ||psi - Phi a||^2 =
      ||psi||^2 + ||a||^2 - 2 <Phi^T psi, a>.
    This is the actual full-space metric (up to float32 roundoff), *not*
    merely the coefficient loss. No validation data is used for gradients.
    """
    with torch.inference_mode():
        pred_z, zenergy = model(features)
        pred = pred_z * torch.as_tensor(scalers['a_std']) + torch.as_tensor(scalers['a_mean'])
        if clip:
            pred = pred.clamp(min=torch.as_tensor(scalers['clip_lo']),
                              max=torch.as_tensor(scalers['clip_hi']))
        e = zenergy * torch.as_tensor(scalers['e_std']) + torch.as_tensor(scalers['e_mean'])
        norm = true_wave_sqnorm.clamp_min(1e-12)
        err2 = (norm + pred.square().sum(dim=1) - 2*(pred*true_coeff).sum(dim=1).abs())/norm
        wave_l2 = torch.sqrt(err2.clamp_min(0)).mean().item()
        energy_rel = (true_energies-e).abs().div(true_energies.abs().clamp_min(1e-12)).mean().item()
    return float(wave_l2), float(energy_rel)


def train_bundle(features,coeff,energies,psi,hred,train_ids,selected_states,phi,cfg,width,tag='model',verbose=True,
                 validation_ids=None):
    """Uses exclusively train geometry IDs. Validation/test are never fitted."""
    n=cfg['network'];ns=coeff.shape[1]
    states=np.asarray(selected_states,dtype=int)
    X=features[np.ix_(train_ids,states)].reshape(-1,features.shape[-1]).astype('float32')
    Y=coeff[np.ix_(train_ids,states)].reshape(-1,coeff.shape[-1]).astype('float32')
    E=energies[np.ix_(train_ids,states)].reshape(-1).astype('float32')
    G=np.broadcast_to(np.asarray(train_ids)[:,None],(len(train_ids),len(states))).reshape(-1)
    scalers=create_scalers(X,Y,E,cfg)
    Xz=((X-scalers['x_mean'])/scalers['x_std']).astype('float32')
    model=EigenpairMLP(X.shape[-1],Y.shape[-1],width,int(n['hidden_layers']),n['activation'])
    t=torch.tensor
    xt=t(Xz); yt=t(Y);et=t(E);git=t(G,dtype=torch.long)
    am=t(scalers['a_mean']);ast=t(scalers['a_std'])
    em=t(scalers['e_mean']);est=t(scalers['e_std'])
    h=t(hred.astype('float32'))
    p=t(psi[np.ix_(train_ids,states)].reshape(-1,psi.shape[-1]).astype('float32'))
    phi_t=t(phi.T.astype('float32'))
    validation_every = int(n.get('validation_every', 0))
    validate = validation_ids is not None and len(validation_ids) and validation_every>0
    if validate:
        vx = features[np.ix_(validation_ids,states)].reshape(-1,features.shape[-1]).astype('float32')
        vy = coeff[np.ix_(validation_ids,states)].reshape(-1,coeff.shape[-1]).astype('float32')
        ve = energies[np.ix_(validation_ids,states)].reshape(-1).astype('float32')
        vpsi = psi[np.ix_(validation_ids,states)].reshape(-1,psi.shape[-1]).astype('float32')
        vx_t = t((vx-scalers['x_mean'])/scalers['x_std'])
        vy_t = t(vy)
        ve_t = t(ve)
        vwave_norm = t(np.sum(vpsi*vpsi,axis=1))
    restore_best = bool(n.get('restore_best_validation', False)) and validate
    patience = int(n.get('validation_patience', 0))
    delta = float(n.get('validation_min_delta', 0.0))
    history=[]
    best_overall_wave=float('inf')
    best_overall_state=None
    for stage,epochs,lr in [('coeff',int(n['fast_epochs']),float(n['learning_rate'])),
                            ('full',int(n['finetune_epochs']),float(n['finetune_learning_rate']))]:
        if epochs==0:continue
        ds=TensorDataset(xt,yt,et,git,p)
        loader=DataLoader(ds,batch_size=int(n['batch_size']),shuffle=True)
        opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=float(n.get('weight_decay',1e-4)))
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.05)
                     if n.get('lr_schedule','constant')=='cosine' else None)
        if n.get('lr_schedule','constant') not in ('constant','cosine'):
            raise ValueError('network.lr_schedule must be constant or cosine')
        best_wave=float('inf'); best_state=None; checks_without_improvement=0
        for epoch in range(epochs):
            model.train();total=0.;count=0
            for xb,yb,eb,gb,psib in loader:
                opt.zero_grad(set_to_none=True)
                zc,ze=model(xb)
                a=zc*ast+am
                e=ze*est+em
                active=active_coeff_loss(yb,a,cfg)
                # Optional gauge-supervised objective. Labels are sign-oriented
                # relative to modes selected from TRAIN geometries only.
                # The original sign-invariant physics/reconstruction loss stays.
                gauge_weight = float(n.get('lambda_gauge_supervision', 0.0)) if n.get('gauge_align_coefficients', False) else 0.0
                gauge_mse = (((a-yb).square().sum(dim=1)) /
                             (yb.square().sum(dim=1)+1e-12)).mean() if gauge_weight>0 else a.new_zeros(())
                reg=zc.square().mean()
                if stage=='coeff':
                    warm=int(n['pinn_warmup_epochs']);ramp=int(n['pinn_ramp_epochs'])
                    scale=max(0.,min(1.,(epoch-warm)/max(1,ramp)))
                    loss=(float(n['lambda_coeff'])*sign_loss(yb,a)+
                          float(n['lambda_active_coeff'])*active+
                          float(n['lambda_energy'])*energy_loss(eb,e)+
                          gauge_weight*gauge_mse+
                          float(n['lambda_reg'])*reg)
                    if scale>0:
                        loss=loss+float(n['lambda_pinn'])*scale*pinn_loss(a,e,h[gb])
                else:
                    wave=a@phi_t
                    true_d=psib.square();pred_d=wave.square()
                    density=((pred_d-true_d).square().sum(dim=1)/(true_d.square().sum(dim=1)+1e-12)).mean()
                    loss=(float(n['lambda_finetune_wave'])*sign_loss(psib,wave)+
                          float(n['lambda_finetune_density'])*density+
                          float(n['lambda_finetune_active_coeff'])*active+
                          float(n['lambda_finetune_energy'])*energy_loss(eb,e)+
                          float(n['lambda_finetune_pinn'])*pinn_loss(a,e,h[gb])+
                          gauge_weight*gauge_mse+
                          float(n['lambda_reg'])*reg)
                if not torch.isfinite(loss):raise FloatingPointError(f'{tag} nonfinite loss at {stage} epoch {epoch}')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),float(n['grad_clip']))
                opt.step();total+=float(loss.detach());count+=1
            mean=total/max(count,1)
            if scheduler is not None:scheduler.step()
            val_wave=float('nan');val_energy=float('nan')
            if validate and ((epoch+1)%validation_every==0 or epoch+1==epochs):
                model.eval()
                val_wave,val_energy=validation_wave_metrics(
                    model,scalers,vx_t,vy_t,vwave_norm,ve_t,
                    clip=bool(n.get('clip_predicted_coefficients',True)))
                if val_wave < best_wave - delta:
                    best_wave = val_wave
                    checks_without_improvement = 0
                    if restore_best:best_state = copy.deepcopy(model.state_dict())
                else:
                    checks_without_improvement += 1
                if restore_best and val_wave < best_overall_wave - delta:
                    best_overall_wave = val_wave
                    best_overall_state = copy.deepcopy(model.state_dict())
            history.append(dict(model=tag,stage=stage,epoch=epoch+1,loss=mean,
                                validation_wave_rel_l2=val_wave,
                                validation_energy_rel_abs=val_energy,
                                learning_rate=float(opt.param_groups[0]['lr'])))
            if verbose and (epoch==0 or epoch+1==epochs or (epoch+1)%max(1,epochs//4)==0):
                tail = f', val wave L2 {val_wave:.4f}' if np.isfinite(val_wave) else ''
                print(f'    {tag} {stage} {epoch+1}/{epochs}, loss {mean:.4e}{tail}',flush=True)
            if patience>0 and checks_without_improvement>=patience:
                if verbose: print(f'    {tag}: validation early stop after {epoch+1} {stage} epochs',flush=True)
                break
        if restore_best and best_state is not None:
            model.load_state_dict(best_state)
    # Do not allow the fine-tuning stage to overwrite a better validation
    # checkpoint from coefficient training.
    if restore_best and best_overall_state is not None:
        model.load_state_dict(best_overall_state)
    model.eval()
    return Bundle(model,scalers),history


def train_models(data,features,coeff,red,phi,split,labels,cfg,verbose=True):
    ns=coeff.shape[1];train=split['train'];network=cfg['network']
    results={}; histories=[]
    if verbose:print('  Global model',flush=True)
    results['global'],hist=train_bundle(features,coeff,data['energies'],data['psi'],red,train,
                        np.arange(ns),phi,cfg,int(network['global_width']),tag='global',verbose=verbose,
                        validation_ids=split['validation'])
    histories+=hist
    locals_={}
    for label in sorted(set(labels)):
        states=np.flatnonzero(labels==label)
        if verbose:print(f'  Local expert {label}: states {states+1}, width {expert_width(len(states),cfg)}',flush=True)
        locals_[int(label)],hist=train_bundle(features,coeff,data['energies'],data['psi'],red,train,
                        states,phi,cfg,expert_width(len(states),cfg),tag=f'expert_{label}',verbose=verbose,
                        validation_ids=split['validation'])
        histories+=hist
    results['local']=locals_
    return results,histories
