"""Self-contained QD DNS→POD→signature KMeans→global/local NN comparison."""
from pathlib import Path
from datetime import datetime
import argparse, hashlib, json
import numpy as np
import pandas as pd
import torch
from .config import load_config,config_digest
from .physics import generate_dns
from .gauge import align_pod_gauge
from .pod import split_geometries,construct_pod,reduced_hamiltonians
from .features import build_features
from .clustering import state_mode_signatures,choose_validated_state_clusters,cluster_state_groups
from .training import train_models
from .models import save_bundle,load_bundle
from .inference import select_backend
from .evaluation import evaluate,benchmark_online
from .plotting import make_plots, make_reconstruction_figures, make_training_figures


def log(message):print(message,flush=True)


def write_rows(path,rows):
    pd.DataFrame(rows).to_csv(path,index=False)


def run(config_path,output=None,resume=False,plots=True):
    cfg=load_config(config_path)
    np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    torch.set_num_threads(int(cfg['runtime']['cpu_threads']))
    root=Path(output) if output else Path('results')/cfg['name']/datetime.now().strftime('%Y%m%d_%H%M%S')
    for sub in ('metrics','models','data','figures'): (root/sub).mkdir(parents=True,exist_ok=True)
    statepath=root/'config_digest.txt';digest=config_digest(cfg)
    if statepath.exists() and statepath.read_text()!=digest:
        raise ValueError('Configuration differs from this run directory. Choose a new --output.')
    statepath.write_text(digest)
    (root/'config.yaml').write_text(Path(config_path).read_text(encoding='utf-8'))
    dnsfile=root/'data'/'dns.npz'
    if resume and dnsfile.exists():
        log('[1/8] Loading cached DNS data')
        with np.load(dnsfile) as f:data={k:f[k] for k in f.files}
    else:
        log('[1/8] Solving 2D Schrödinger DNS (four Gaussian dots)')
        data=generate_dns(cfg)
        np.savez_compressed(dnsfile,**data)
    split=split_geometries(len(data['params']),cfg)
    np.savez(root/'data'/'geometry_split.npz',**split)
    log(f"[2/8] Train/val/test geometries: {*(len(split[x]) for x in ('train','validation','test')),}")
    podfile=root/'data'/'pod.npz'
    if resume and podfile.exists():
        with np.load(podfile) as f:phi=f['phi'];coeff=f['coeff'];sv=f['singular_values']
    else:
        phi,coeff,sv=construct_pod(data,split,cfg)
        np.savez_compressed(podfile,phi=phi,coeff=coeff,singular_values=sv)
    log(f'[3/8] Train-only POD basis = {phi.shape}, coefficients = {coeff.shape}')
    # Training-only POD mode anchors reduce arbitrary sign changes in the
    # supervised regression targets; the actual physical states do not change.
    if cfg['network'].get('gauge_align_coefficients', False):
        coeff, oriented_psi, gauge_audit = align_pod_gauge(coeff, data['psi'], split['train'])
        data['psi'] = oriented_psi
        write_rows(root/'metrics'/'gauge_anchor_modes.csv', gauge_audit)
        log('      Applied train-only POD coefficient sign anchors')
    redfile=root/'data'/'reduced.npz'
    if resume and redfile.exists():
        with np.load(redfile) as f:red=f['red'];features=f['features']
    else:
        red=reduced_hamiltonians(data,phi,cfg)
        features=build_features(data,red,cfg)
        np.savez_compressed(redfile,red=red,features=features)
    log(f'[4/8] Features = {features.shape}, projected Hamiltonians = {red.shape}')
    tr=state_mode_signatures(coeff,split['train'],cfg)
    va=state_mode_signatures(coeff,split['validation'],cfg)
    labels,scores=choose_validated_state_clusters(tr,va,cfg)
    groups=cluster_state_groups(labels,tr)
    write_rows(root/'metrics'/'cluster_validation_scores.csv',scores)
    write_rows(root/'metrics'/'expert_state_groups.csv',groups)
    np.savez(root/'data'/'clustering.npz',labels=labels,train_signatures=tr,val_signatures=va)
    log(f'[5/8] Research-rule KMeans chose {len(np.unique(labels))} experts; sizes {np.bincount(labels).tolist()}')
    saved=root/'models'/'global.pt'
    if resume and saved.exists():
        log('[6/8] Loading existing trained global/local checkpoints')
        global_net=load_bundle(saved)
        experts={int(label):load_bundle(root/'models'/f'expert_{label}.pt') for label in np.unique(labels)}
        models={'global':global_net,'local':experts}
    else:
        log('[6/8] Training global model and research-width local experts')
        models,hist=train_models(data,features,coeff,red,phi,split,labels,cfg)
        save_bundle(models['global'],saved)
        for label,b in models['local'].items():save_bundle(b,root/'models'/f'expert_{label}.pt')
        write_rows(root/'metrics'/'training_history.csv',hist)
    log('[7/8] Validating packed inference and choosing CPU threads/backend')
    input_val=features[split['validation'][:min(2,len(split['validation']))]]
    local_predictor,audits,choice=select_backend(models['local'],labels,input_val,cfg)
    write_rows(root/'metrics'/'cpu_inference_tuning.csv',audits)
    parameters=[dict(model='Global POD-NN',parameters=sum(t.numel() for t in models['global'].model.parameters()))]
    for k,b in models['local'].items():
        parameters.append(dict(model=f'Local expert {k}',parameters=sum(t.numel() for t in b.model.parameters())))
    parameters.append(dict(model='Local POD-NN (all experts)',parameters=sum(row['parameters'] for row in parameters[1:])))
    write_rows(root/'metrics'/'parameter_counts.csv',parameters)
    log(f'    Backend = {choice}')
    log('[8/8] Held-out accuracy + complete online timing')
    accuracy,per_state=evaluate(data,features,phi,red,models,labels,split,cfg,local_predictor)
    timings=benchmark_online(data,features,phi,red,models,split['test'],cfg,local_predictor)
    write_rows(root/'metrics'/'summary.csv',accuracy)
    write_rows(root/'metrics'/'per_state_errors.csv',per_state)
    write_rows(root/'metrics'/'timings.csv',timings)
    import statistics
    sm={r['model']:r for r in accuracy}
    time_med=pd.DataFrame(timings).groupby('model')['total_online_ms'].median().to_dict()
    # Outcome criteria are *diagnostics*, never model-selection rules.
    target_rows=[
        dict(goal='Local mean wave relative L2 below 0.20',
             observed=float(sm['Local POD-NN']['mean_wave_rel_l2']),threshold=0.20,
             passes=bool(sm['Local POD-NN']['mean_wave_rel_l2']<0.20)),
        dict(goal='Local mean relative energy error below 0.02',
             observed=float(sm['Local POD-NN']['mean_energy_rel_abs']),threshold=0.02,
             passes=bool(sm['Local POD-NN']['mean_energy_rel_abs']<0.02)),
        dict(goal='Local/global total latency at most 1.20',
             observed=float(time_med['Local POD-NN']/time_med['Global POD-NN']),threshold=1.20,
             passes=bool(time_med['Local POD-NN']/time_med['Global POD-NN']<=1.20)),
        dict(goal='Local wave error lower than global wave error',
             observed=float(sm['Local POD-NN']['mean_wave_rel_l2']/max(sm['Global POD-NN']['mean_wave_rel_l2'],1e-12)),
             threshold=1.0,
             passes=bool(sm['Local POD-NN']['mean_wave_rel_l2']<sm['Global POD-NN']['mean_wave_rel_l2'])),
    ]
    write_rows(root/'metrics'/'goal_check.csv',target_rows)
    details=dict(config=cfg,run_output=str(root),geometry_counts={k:len(v) for k,v in split.items()},
                 selected_experts=len(set(labels)),expert_state_counts=np.bincount(labels).tolist(),
                 accuracy=accuracy,median_total_online_ms=time_med,goal_check=target_rows,inference_backend=choice,
                 parameter_counts=parameters,
                 notes='This demo does not guarantee that local accuracy beats global accuracy.')
    (root/'run_summary.json').write_text(json.dumps(details,indent=2,default=float))
    if plots:
        make_plots(root)
        make_reconstruction_figures(root, data, phi, red, models, labels, split, cfg, local_predictor)
        if (root/'metrics'/'training_history.csv').exists():
            make_training_figures(root)
    log('\nModel accuracy summary:')
    log(pd.DataFrame(accuracy).to_string(index=False))
    log('\nMedian total online runtime (ms):')
    log(pd.Series(time_med).to_string())
    log(f'\nSaved all outputs in {root.resolve()}')
    return root


def main():
    ap=argparse.ArgumentParser(description='Self-contained 40-state, 80-mode four-QD POD-NN demo')
    ap.add_argument('--config',default='configs/default_40state_tanh.yaml')
    ap.add_argument('--output',default=None)
    ap.add_argument('--resume',action='store_true',help='Reuse caches/checkpoints in the same output folder')
    ap.add_argument('--no-plots',action='store_true')
    args=ap.parse_args()
    run(args.config,args.output,args.resume,not args.no_plots)
