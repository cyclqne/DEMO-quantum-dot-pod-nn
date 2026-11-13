"""Configuration loader and validation; no Windows-specific file paths."""
from pathlib import Path
import json
import yaml


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    p, pod, feats, split = cfg['physics'], cfg['pod'], cfg['features'], cfg['splits']
    if p['grid'] < 9 or p['states'] < 2 or p['states'] >= p['grid']**2 - 1:
        raise ValueError('Invalid grid or eigenstate count')
    if not (p['states'] <= pod['modes'] <= p['grid']**2):
        raise ValueError('POD modes must be >= retained states and <= grid points')
    if feats['spectral_eigenvalues'] > pod['modes']:
        raise ValueError('Cannot compute more reduced eigenvalues than POD modes')
    if abs(sum(split.values()) - 1.) > 1e-9:
        raise ValueError('Train+validation+test must sum to 1')
    if p['random_options_per_well'] < 2:
        raise ValueError('At least two random options per well are required')
    n = cfg['network']
    if int(n.get('validation_every', 0)) < 0 or int(n.get('validation_patience', 0)) < 0:
        raise ValueError('Validation interval/patience cannot be negative')
    if n.get('restore_best_validation', False) and int(n.get('validation_every', 0)) <= 0:
        raise ValueError('Validation checkpoint selection requires validation_every>0')
    if n.get('lr_schedule', 'constant') not in ('constant', 'cosine'):
        raise ValueError('Invalid network learning-rate schedule')
    if n.get('gauge_align_coefficients', False) and float(n.get('lambda_gauge_supervision', 0)) < 0:
        raise ValueError('Gauge supervision weight must be nonnegative')
    r = cfg['runtime']
    if 'reconstruction_states' in r and any(int(st)<1 or int(st)>int(p['states']) for st in r['reconstruction_states']):
        raise ValueError('Reconstruction state indices must be between 1 and physics.states')
    c = cfg['clustering']
    if c['n_init'] < 1 or not c['candidate_k']:
        raise ValueError('Invalid KMeans settings')
    if any(k < 2 for k in c['candidate_k']):
        raise ValueError('KMeans candidate K must be >=2')
    fixed_k=c.get('fixed_k')
    if fixed_k is not None:
        if isinstance(fixed_k,bool) or not isinstance(fixed_k,int) or not 2 <= fixed_k <= p['states']:
            raise ValueError('clustering.fixed_k must be null or an integer between 2 and physics.states')
    threshold=n.get('expert_width_state_thresholds',{})
    if threshold:
        a,b,d=(int(threshold.get(x,default)) for x,default in
               (('small_max',3),('large_min',7),('very_large_min',10)))
        if not (1<=a<b<d):
            raise ValueError('Expert width thresholds must satisfy 1 <= small_max < large_min < very_large_min')
    return cfg


def config_digest(cfg):
    import hashlib
    return hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
