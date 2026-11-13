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


def goal_diagnostics(accuracy, timings):
    """Informative thresholds only; never used to choose a network model."""
    sm={r['model']:r for r in accuracy}
    time_med=pd.DataFrame(timings).groupby('model')['total_online_ms'].median().to_dict()
    target_rows=[
        dict(goal='Local mean squared wave LSE below 4%',
             observed=float(sm['Local POD-NN']['mean_wave_lse_pct']),threshold=4.0,
             passes=bool(sm['Local POD-NN']['mean_wave_lse_pct']<4.0)),
        dict(goal='Local mean relative energy error below 2%',
             observed=float(sm['Local POD-NN']['mean_energy_rel_pct']),threshold=2.0,
             passes=bool(sm['Local POD-NN']['mean_energy_rel_pct']<2.0)),
        dict(goal='Local/global total latency at most 1.20',
             observed=float(time_med['Local POD-NN']/time_med['Global POD-NN']),threshold=1.20,
             passes=bool(time_med['Local POD-NN']/time_med['Global POD-NN']<=1.20)),
        dict(goal='Local wave error lower than global wave error',
             observed=float(sm['Local POD-NN']['mean_wave_lse_pct']/max(sm['Global POD-NN']['mean_wave_lse_pct'],1e-12)),
             threshold=1.0,
             passes=bool(sm['Local POD-NN']['mean_wave_lse_pct']<sm['Global POD-NN']['mean_wave_lse_pct'])),
    ]
    return target_rows


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
    write_rows(root/'metrics'/'pod_timing_breakdown.csv',[r for r in timings if r['model']=='POD-Galerkin'])
    time_med=pd.DataFrame(timings).groupby('model')['total_online_ms'].median().to_dict()
    target_rows=goal_diagnostics(accuracy,timings)
    write_rows(root/'metrics'/'goal_check.csv',target_rows)
    details=dict(config=cfg,run_output=str(root),geometry_counts={k:len(v) for k,v in split.items()},
                 selected_experts=len(set(labels)),expert_state_counts=np.bincount(labels).tolist(),
                 accuracy=accuracy,median_total_online_ms=time_med,goal_check=target_rows,inference_backend=choice,
                 pod_timing_medians_ms=pd.DataFrame(timings).query('model == "POD-Galerkin"')[
                     ['pod_assembly_ms','pod_eigensolve_only_ms','pod_reconstruction_ms','pod_full_online_ms']].median().to_dict(),
                 wavefunction_error_display='sign-aligned squared LSE ratio × 100 percent',
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



def replot_saved_run(config_path, output):
    """Refresh accuracy/timing figures and signed wavefunctions from saved models.

    Does not generate DNS, build POD, retrain NNs, or alter checkpoints.
    Re-evaluates held-out accuracy and benchmarks two POD timing definitions
    using identical saved test geometries and checkpoint predictions.
    """
    from .inference import SequentialExperts
    cfg = load_config(config_path)
    root = Path(output)
    expected = config_digest(cfg)
    digest_file = root/'config_digest.txt'
    if not digest_file.exists() or digest_file.read_text().strip() != expected:
        raise ValueError('Saved run is missing or config differs; use its original --config')
    required = [root/'data'/'dns.npz',root/'data'/'pod.npz',
                root/'data'/'reduced.npz',root/'data'/'geometry_split.npz',
                root/'data'/'clustering.npz',root/'models'/'global.pt']
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError('Incomplete saved run: '+', '.join(missing))
    torch.set_num_threads(int(cfg['runtime']['cpu_threads']))
    with np.load(root/'data'/'dns.npz') as f:data={k:f[k] for k in f.files}
    with np.load(root/'data'/'pod.npz') as f:phi=f['phi']
    with np.load(root/'data'/'reduced.npz') as f:red=f['red'];features=f['features']
    with np.load(root/'data'/'geometry_split.npz') as f:split={k:f[k] for k in f.files}
    with np.load(root/'data'/'clustering.npz') as f:labels=f['labels']
    global_net=load_bundle(root/'models'/'global.pt')
    experts={int(k):load_bundle(root/'models'/f'expert_{k}.pt') for k in np.unique(labels)}
    models={'global':global_net,'local':experts}
    predictor=SequentialExperts(experts,labels,clip=cfg['network']['clip_predicted_coefficients'])
    log('Recomputing held-out LSE percentages and both POD timing definitions without retraining')
    accuracy,per_state=evaluate(data,features,phi,red,models,labels,split,cfg,predictor)
    timing_rows=benchmark_online(data,features,phi,red,models,split['test'],cfg,predictor)
    write_rows(root/'metrics'/'summary.csv',accuracy)
    write_rows(root/'metrics'/'per_state_errors.csv',per_state)
    write_rows(root/'metrics'/'timings.csv',timing_rows)
    write_rows(root/'metrics'/'pod_timing_breakdown.csv',
               [r for r in timing_rows if r['model']=='POD-Galerkin'])
    goals=goal_diagnostics(accuracy,timing_rows)
    write_rows(root/'metrics'/'goal_check.csv',goals)
    summary_path=root/'run_summary.json'
    run_info=json.loads(summary_path.read_text()) if summary_path.exists() else {}
    run_info.update(accuracy=accuracy,goal_check=goals,
        median_total_online_ms=pd.DataFrame(timing_rows).groupby('model')['total_online_ms'].median().to_dict(),
        pod_timing_medians_ms=pd.DataFrame(timing_rows).query('model == "POD-Galerkin"')[
            ['pod_assembly_ms','pod_eigensolve_only_ms','pod_reconstruction_ms','pod_full_online_ms']].median().to_dict(),
        wavefunction_error_display='sign-aligned squared LSE ratio × 100 percent')
    summary_path.write_text(json.dumps(run_info,indent=2,default=float))
    log('Generating accuracy, timing and signed-wavefunction figures from saved weights')
    make_plots(root)
    make_reconstruction_figures(root,data,phi,red,models,labels,split,cfg,predictor)
    if (root/'metrics'/'training_history.csv').exists():
        make_training_figures(root)
    log(f'Updated figures saved under {(root/"figures").resolve()}')
    return root


def main():
    ap=argparse.ArgumentParser(description='Self-contained 40-state, 80-mode four-QD POD-NN demo')
    ap.add_argument('--config',default='configs/demo_parameters.yaml',
                    help='YAML parameter file (default: configs/demo_parameters.yaml)')
    ap.add_argument('--output',default=None)
    ap.add_argument('--resume',action='store_true',help='Reuse caches/checkpoints in the same output folder')
    ap.add_argument('--no-plots',action='store_true')
    ap.add_argument('--plots-only',action='store_true',help='Recompute accuracy/timing figures and wavefunction plots from saved models; no retraining')
    args=ap.parse_args()
    if args.plots_only:
        if not args.output:
            ap.error('--plots-only requires --output pointing to an existing run folder')
        replot_saved_run(args.config,args.output)
    else:
        run(args.config,args.output,args.resume,not args.no_plots)
