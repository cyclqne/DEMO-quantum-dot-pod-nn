"""Honest diagnostic plots; no claimed speedup when measurements do not support it."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_plots(out):
    out=Path(out);figs=out/'figures';figs.mkdir(exist_ok=True)
    summary=pd.read_csv(out/'metrics'/'summary.csv')
    timings=pd.read_csv(out/'metrics'/'timings.csv')
    groups=pd.read_csv(out/'metrics'/'expert_state_groups.csv')
    cluster=pd.read_csv(out/'metrics'/'cluster_validation_scores.csv')
    errors=pd.read_csv(out/'metrics'/'per_state_errors.csv')
    order=['Global POD-NN','Local POD-NN','POD-Galerkin']
    for col,label,outfile in [('mean_wave_rel_l2','Mean sign-invariant wave L2 error','wavefunction_accuracy.png'),
                              ('mean_density_rel_l2','Mean relative density L2 error','density_accuracy.png'),
                              ('mean_energy_rel_abs','Mean relative energy error','energy_accuracy.png')]:
        fig,ax=plt.subplots(figsize=(8,4.8))
        selected=summary.set_index('model').loc[order]
        ax.bar(order,selected[col],color=['#57789d','#398d8a','#9a9f9e'])
        ax.set_ylabel(label);ax.set_title('Held-out accuracy (lower is better)')
        ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(figs/outfile,dpi=180);plt.close(fig)
    names=['DNS (solve only)','POD-Galerkin','Global POD-NN','Local POD-NN']
    m=timings.groupby('model')['total_online_ms'].median()
    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.bar(names,m.reindex(names),color=['#999999','#a2a760','#57789d','#398d8a'])
    ax.set_ylabel('Median time per geometry (ms)');ax.set_title('Online cost; DNS = solve only')
    ax.tick_params(axis='x',labelrotation=10);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(figs/'total_cpu_latency.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5))
    only=timings[timings.model.isin(order[:2])]
    comps=only.groupby('model')[['feature_ms','forward_ms','reconstruction_ms']].median().reindex(order[:2])
    bottom=np.zeros(2)
    for name,color in [('feature_ms','#a1b4c9'),('forward_ms','#57789d'),('reconstruction_ms','#55a57d')]:
        ax.bar(order[:2],comps[name],bottom=bottom,label=name.replace('_ms','').replace('_',' ').title(),color=color)
        bottom+=comps[name].values
    ax.set_ylabel('Median component times (ms)');ax.set_title('NN online latency breakdown');ax.legend()
    fig.tight_layout();fig.savefig(figs/'online_latency_breakdown.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,4.5))
    ax.plot(cluster.k,cluster.total_score,'o-')
    ax.scatter(cluster.loc[cluster.selected,'k'],cluster.loc[cluster.selected,'total_score'],
               s=100,marker='*',label='Selected',color='tab:red')
    ax.set(xlabel='KMeans candidate K',ylabel='Original validation score',title='Research-rule state clustering')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(figs/'kmeans_selection.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.scatter(groups.state,groups.expert,c=groups.expert,cmap='viridis',s=46)
    ax.set(xlabel='Eigenstate index',ylabel='Local expert ID',title='Expert assignments (train/validation only)')
    fig.tight_layout();fig.savefig(figs/'expert_state_groups.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5))
    means=errors.groupby(['model','state'])['wave_rel_l2'].mean().unstack(0)
    for method in order:ax.plot(means.index,means[method],label=method)
    ax.set(xlabel='Eigenstate index',ylabel='Mean sign-invariant wave L2',title='Held-out per-state errors')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(figs/'per_state_wave_error.png',dpi=180);plt.close(fig)
    make_timing_detail_figures(out, timings)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def make_timing_detail_figures(out, timings):
    """Plot per-geometry online cost for *every* method and relative overhead.

    DNS is clearly marked solve-only, while POD and both NNs include their
    required online computation. X labels are actual held-out geometry IDs.
    """
    out = Path(out)
    figures = out/'figures'
    all_methods = ['DNS (solve only)', 'POD-Galerkin', 'Global POD-NN', 'Local POD-NN']
    reduced = ['POD-Galerkin', 'Global POD-NN', 'Local POD-NN']
    neural = ['Global POD-NN', 'Local POD-NN']
    colors = {'DNS (solve only)':'#a66532', 'POD-Galerkin':'#6a8d53',
              'Global POD-NN':'#406ca5', 'Local POD-NN':'#188d90'}
    t = timings.pivot(index='geometry', columns='model', values='total_online_ms').sort_index()
    x = t.index.to_numpy()

    for group, filename, title, log_axis in [
        (all_methods,'online_time_by_geometry_all_methods.png',
         'Online time for each held-out geometry (DNS: solve only)',True),
        (reduced,'online_time_by_geometry_reduced_methods.png',
         'Full online time: POD-Galerkin vs global/local NNs',False),
        (neural,'online_time_by_geometry_neural_networks.png',
         'Global vs local NN total online time per geometry',False),
    ]:
        fig, ax = plt.subplots(figsize=(11,5.2))
        for name in group:
            ax.plot(x, t[name].to_numpy(), marker='o', markersize=3.2, linewidth=1.4,
                    label=name, color=colors[name])
        if log_axis:ax.set_yscale('log')
        ax.set(xlabel='Held-out geometry ID',ylabel='Milliseconds per geometry',title=title)
        ax.grid(alpha=.23, which='both');ax.legend(fontsize=9)
        _save(fig,figures/filename)

    fig, ax = plt.subplots(figsize=(10.5,4.5))
    ratio = t['Local POD-NN'].to_numpy()/np.maximum(t['Global POD-NN'].to_numpy(),1e-9)
    ax.plot(x,ratio,'o-',linewidth=1.5,markersize=3.5,color=colors['Local POD-NN'])
    ax.axhline(1.,linestyle='--',color='#444444',label='Same online time')
    ax.axhline(1.2,linestyle=':',color='#aa6222',label='20% slower than global')
    ax.set(xlabel='Held-out geometry ID',ylabel='Local/global full online-time ratio',
           title='Local inference overhead on each test geometry')
    ax.grid(alpha=.2);ax.legend()
    _save(fig,figures/'local_global_latency_ratio_by_geometry.png')

    f = timings[timings.model.isin(neural)].pivot(index='geometry',columns='model',values='forward_ms').sort_index()
    fig,ax=plt.subplots(figsize=(10.5,4.5))
    for name in neural:
        ax.plot(f.index,f[name],marker='o',markersize=3.2,label=name,linewidth=1.4,color=colors[name])
    ax.set(xlabel='Held-out geometry ID', ylabel='Forward inference (ms)',
           title='Network forward time (separate from online feature construction)')
    ax.grid(alpha=.2);ax.legend()
    _save(fig,figures/'nn_forward_time_by_geometry.png')

    fig, ax = plt.subplots(figsize=(9,4.6))
    values = [t[name].to_numpy() for name in reduced]
    ax.boxplot(values,showfliers=True)
    ax.set_xticks(np.arange(1,len(reduced)+1),reduced)
    ax.set(ylabel='Full online time (ms)',title='Held-out geometry latency distributions')
    ax.grid(axis='y',alpha=.2)
    _save(fig,figures/'online_time_distribution_reduced.png')


def make_reconstruction_figures(out, data, phi, red, models, labels, split, cfg, local_predictor):
    """Reconstruct wavefunctions/densities on fixed held-out geometries.

    Color range is SHARED across the four methods for each field; predicted
    wavefunction signs are aligned to DNS *for display only*. No hidden test
    selection or oracle features are used by the neural predictors.
    """
    from .evaluation import predict_global
    from .pod import pod_galerkin

    out = Path(out); figdir=out/'figures'; figdir.mkdir(parents=True,exist_ok=True)
    ngrid=int(cfg['physics']['grid']); length=float(cfg['physics']['length_nm'])
    test=split['test']; selected_geometry_count=min(len(test),int(cfg['runtime'].get('reconstruction_geometries',2)))
    selected_geometries=list(test[:selected_geometry_count])
    selected_states=cfg['runtime'].get('reconstruction_states',[1,cfg['physics']['states']])
    selected_states=sorted(set(int(s)-1 for s in selected_states))
    for geom_id in selected_geometries:
        one_features = None
        # The same precomputed reduced-operator features that enter accuracy
        # evaluation are used; no hidden DNS states enter either predictor.
        from .features import build_features
        one_features=build_features({'params':data['params'][geom_id:geom_id+1]},
                                    red[geom_id:geom_id+1],cfg)
        global_coeff,_=predict_global(models['global'],one_features,cfg)
        local_coeff,_=local_predictor.predict(one_features)
        _,gal=pod_galerkin({'energies':data['energies'][geom_id:geom_id+1]},phi,
                            red[geom_id:geom_id+1])
        global_wave=np.einsum('sm,pm->sp',global_coeff[0],phi,optimize=True)
        local_wave=np.einsum('sm,pm->sp',local_coeff[0],phi,optimize=True)
        gt=data['psi'][geom_id]
        names=['DNS','POD-Galerkin','Global POD-NN','Local POD-NN']
        all_waves=[gt,gal[0],global_wave,local_wave]
        for state in selected_states:
            if not (0<=state<gt.shape[0]):continue
            reference=gt[state].astype('float64')
            fields=[reference]
            for waves in all_waves[1:]:
                v=waves[state].astype('float64')
                if np.dot(reference,v)<0:v=-v
                fields.append(v)
            vmax=max(float(np.max(abs(w))) for w in fields)
            density_max=max(float(np.max(w*w)) for w in fields)
            fig,axes=plt.subplots(2,4,figsize=(15,7.7),sharex=True,sharey=True)
            for j,(name,wave) in enumerate(zip(names,fields)):
                arr=wave.reshape((ngrid,ngrid),order='F')
                im_wave=axes[0,j].imshow(arr,origin='lower',extent=[0,length,0,length],
                                          cmap='RdBu_r',vmin=-vmax,vmax=vmax,interpolation='nearest')
                im_density=axes[1,j].imshow(arr*arr,origin='lower',extent=[0,length,0,length],
                                             cmap='viridis',vmin=0,vmax=density_max,interpolation='nearest')
                error=min(np.linalg.norm(reference-wave),np.linalg.norm(reference+wave))/max(np.linalg.norm(reference),1e-12)
                axes[0,j].set_title(name + (f'  | L2={error:.3f}' if j else '  | reference'),fontsize=11)
                axes[1,j].set_xlabel('x (nm)')
                if j==0:
                    axes[0,j].set_ylabel('y (nm)   |   wavefunction')
                    axes[1,j].set_ylabel('y (nm)   |   density |ψ|²')
            fig.suptitle(f'Geometry {int(geom_id)} · eigenstate {state+1} · same color ranges by row',fontsize=14)
            fig.colorbar(im_wave,ax=list(axes[0,:]),fraction=.012,pad=.012,label='Signed wavefunction amplitude')
            fig.colorbar(im_density,ax=list(axes[1,:]),fraction=.012,pad=.012,label='Density |ψ|²')
            # avoid tight_layout warning for shared per-row colorbars
            fig.subplots_adjust(left=.055,right=.91,top=.87,bottom=.08,hspace=.17,wspace=.10)
            fig.savefig(figdir/f'reconstruction_geometry_{geom_id:04d}_state_{state+1:02d}.png',dpi=180)
            plt.close(fig)


def make_training_figures(out):
    """Loss and direct held-out-validation trends, split by model and stage."""
    out=Path(out);f=out/'metrics'/'training_history.csv'
    if not f.exists():return
    df=pd.read_csv(f)
    if df.empty:return
    figdir=out/'figures'
    for metric,filename,title,yscale in [
        ('loss','training_loss_by_model.png','Stage-wise optimization objective','log'),
        ('validation_wave_rel_l2','validation_wave_error_by_model.png',
         'Actual sign-invariant validation wave L2','linear'),
        ('validation_energy_rel_abs','validation_energy_error_by_model.png',
         'Relative validation energy error','linear')]:
        if metric not in df:continue
        fig,axes=plt.subplots(1,2,figsize=(13,4.5))
        for k,stage in enumerate(('coeff','full')):
            part=df[df.stage==stage]
            for name,group in part.groupby('model'):
                v=group[metric].to_numpy(dtype=float)
                valid=np.isfinite(v) & (v>0 if yscale=='log' else np.ones(len(v),dtype=bool))
                if valid.any():axes[k].plot(group.epoch.to_numpy()[valid],v[valid],label=name,linewidth=1.4)
            axes[k].set(xlabel='Epoch',ylabel=metric.replace('_',' '),title=stage+' stage')
            axes[k].grid(alpha=.2)
            if yscale=='log':axes[k].set_yscale('log')
            if axes[k].lines:axes[k].legend(fontsize=8)
        fig.suptitle(title)
        _save(fig,figdir/filename)
