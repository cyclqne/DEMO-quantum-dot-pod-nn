"""Accuracy, wavefunction and honest CPU timing visualizations."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Subdued, legible colors: retained blue/teal identity for the neural models,
# warm olive for POD, and a muted clay brown for DNS.
PALETTE = {'DNS (solve only)':'#ae8169', 'POD-Galerkin':'#898f68',
           'Global POD-NN':'#577a98', 'Local POD-NN':'#348887'}
NN_METHODS = ['Global POD-NN', 'Local POD-NN']
# Both POD timing definitions are retained in numeric output. Only the reduced
# eigensolve-only definition is displayed in the one POD timing comparison.
POD_FIELDS = {'full_online': ('total_online_ms','POD full online (assembly + eigensolve + reconstruction)'),
              'eigensolve_only': ('pod_eigensolve_only_ms','POD reduced eigensolve only')}
POD_TIMING_FIGURE = 'online_time_by_geometry_reduced_methods_pod_eigensolve_only.png'

# Exact names from earlier releases. Refreshing an existing run will not leave
# obsolete (or misleading full-online) POD comparisons in the figures folder.
RETIRED_POD_TIMING_FIGURES = (
    'online_time_by_geometry_all_methods_pod_full_online.png',
    'online_time_by_geometry_all_methods_pod_eigensolve_only.png',
    'online_time_by_geometry_reduced_methods_pod_full_online.png',
    'online_time_by_geometry_all_methods.png',
    'online_time_by_geometry_reduced_methods.png',
    'online_time_distribution_reduced.png',
    'online_time_distribution_reduced_pod_full_online.png',
    'online_time_distribution_reduced_pod_eigensolve_only.png',
    'pod_vs_nn_timing_by_geometry_pod_full_online.png',
    'pod_vs_nn_timing_by_geometry_pod_eigensolve_only.png',
    'pod_to_nn_time_ratio_by_geometry_pod_full_online.png',
    'pod_to_nn_time_ratio_by_geometry_pod_eigensolve_only.png',
    'pod_timing_components_by_geometry.png',
    'pod_timing_components_stacked_by_geometry.png',
    'total_cpu_latency.png',
    'online_time_by_geometry_all_methods_pod_eigensolve_only.png.bak',
)


def remove_retired_pod_plots(figures):
    figures = Path(figures)
    for filename in RETIRED_POD_TIMING_FIGURES:
        (figures / filename).unlink(missing_ok=True)



def _save(fig,path):
    fig.savefig(path,dpi=195,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def make_plots(out):
    out=Path(out)
    figs=out/'figures';figs.mkdir(exist_ok=True)
    # Remove legacy outputs when regenerating an existing run. These files are
    # intentionally no longer produced, but older versions may have left them.
    obsolete = (
        'wavefunction_accuracy.png',
        'per_state_wave_error.png',
        'wave_lse_percent_by_geometry.png',
        'online_time_by_geometry_neural_networks.png',
        'online_time_by_geometry_nueral_networks.png',
        'local_global_latency_ratio_by_geometry.png',
        'nn_forward_time_by_geometry.png',
    )
    for filename in obsolete:
        (figs / filename).unlink(missing_ok=True)
    remove_retired_pod_plots(figs)
    summary=pd.read_csv(out/'metrics'/'summary.csv')
    timings=pd.read_csv(out/'metrics'/'timings.csv')
    groups=pd.read_csv(out/'metrics'/'expert_state_groups.csv')
    cluster=pd.read_csv(out/'metrics'/'cluster_validation_scores.csv')
    errors=pd.read_csv(out/'metrics'/'per_state_errors.csv')
    # All accuracy visualizations compare the two trained NNs, not POD.
    selected=summary.set_index('model').loc[NN_METHODS]
    for col,label,outfile in [('mean_density_lse_pct','Density LSE (%)','density_accuracy.png'),
                              ('mean_energy_rel_pct','Mean relative energy error (%)','energy_accuracy.png')]:
        fig,ax=plt.subplots(figsize=(7.8,4.6))
        ax.bar(NN_METHODS,selected[col].to_numpy(),
               color=[PALETTE[n] for n in NN_METHODS],width=.56)
        ax.set(ylabel=label,title='Held-out NN accuracy (lower is better)')
        ax.grid(axis='y',alpha=.2,color='#888888')
        ax.set_axisbelow(True);fig.tight_layout();_save(fig,figs/outfile)

    fig,ax=plt.subplots(figsize=(8.8,4.7))
    components=timings[timings.model.isin(NN_METHODS)].groupby('model')[
        ['feature_ms','forward_ms','reconstruction_ms']].median().reindex(NN_METHODS)
    bottom=np.zeros(2)
    for col,display,color in [('feature_ms','Operator and features','#bdc5bf'),
                              ('forward_ms','Neural inference','#577a98'),
                              ('reconstruction_ms','Wavefunction reconstruction','#a5ad78')]:
        ax.bar(NN_METHODS,components[col].to_numpy(),bottom=bottom,
               label=display,color=color,width=.56)
        bottom+=components[col].to_numpy()
    ax.set(ylabel='Median component time (ms)',title='Neural network online latency breakdown')
    ax.legend(loc='upper right',fontsize=9);ax.grid(axis='y',alpha=.16)
    fig.tight_layout();_save(fig,figs/'online_latency_breakdown.png')

    fig,ax=plt.subplots(figsize=(7.7,4.5))
    ax.plot(cluster.k,cluster.total_score,'o-',color=PALETTE['Global POD-NN'],lw=1.6)
    chosen=cluster.loc[cluster.selected]
    ax.scatter(chosen.k,chosen.total_score,s=110,marker='*',label='Selected',color='#b2755e',zorder=3)
    ax.set(xlabel='Number of KMeans experts',ylabel='Validation score',title='State clustering selection')
    ax.legend();ax.grid(alpha=.18);fig.tight_layout();_save(fig,figs/'kmeans_selection.png')

    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.scatter(groups.state,groups.expert,c=groups.expert,cmap='cividis',s=48)
    ax.set(xlabel='Eigenstate index',ylabel='Local expert ID',title='Expert assignments')
    ax.grid(axis='x',alpha=.12);fig.tight_layout();_save(fig,figs/'expert_state_groups.png')

    err_nn=errors[errors.model.isin(NN_METHODS)]
    means=err_nn.groupby(['model','state'])['wave_lse_pct'].mean().unstack('model')
    fig,ax=plt.subplots(figsize=(10,4.8))
    for model in NN_METHODS:
        ax.plot(means.index,means[model],label=model,color=PALETTE[model],lw=1.8)
    ax.set(xlabel='Eigenstate index',ylabel='Mean wavefunction LSE (%)',
           title='Held-out wavefunction error by eigenstate (sign aligned)')
    ax.grid(alpha=.18);ax.legend();fig.tight_layout()
    _save(fig,figs/'per_state_wave_lse_percent.png')

    timing_nn=timings[timings.model.isin(NN_METHODS)]
    merged=err_nn.groupby(['model','geometry'],as_index=False)['wave_lse_pct'].mean().merge(
        timing_nn[['model','geometry','total_online_ms']],on=['model','geometry'],validate='one_to_one')
    fig,ax=plt.subplots(figsize=(8.2,5.1))
    for model in NN_METHODS:
        subset=merged[merged.model==model]
        ax.scatter(subset.total_online_ms,subset.wave_lse_pct,s=35,alpha=.75,
                   color=PALETTE[model],label=model,edgecolor='white',linewidth=.35)
    ax.set(xlabel='Total NN online time (ms/geometry)',ylabel='Mean wavefunction LSE (%)',
           title='Held-out accuracy versus online latency')
    ax.legend();ax.grid(alpha=.16);fig.tight_layout()
    _save(fig,figs/'accuracy_vs_online_latency_by_geometry.png')
    make_timing_detail_figures(out,timings)


def _timing_matrix(timings, pod_variant):
    """Return one per-geometry timing matrix with the requested POD definition."""
    field,_=POD_FIELDS[pod_variant]
    df=timings[['geometry','model','total_online_ms', 'pod_eigensolve_only_ms']].copy()
    if pod_variant=='eigensolve_only':
        mask=df.model.eq('POD-Galerkin')
        if df.loc[mask,'pod_eigensolve_only_ms'].isna().any():
            raise ValueError('POD eigensolve-only timings are missing; rerun timing benchmark')
        df.loc[mask,'total_online_ms']=df.loc[mask,'pod_eigensolve_only_ms']
    return df.pivot(index='geometry',columns='model',values='total_online_ms').sort_index()


def make_timing_detail_figures(out, timings):
    """Generate exactly ONE POD timing comparison with reduced eigensolve only.

    Shows per-geometry reduced POD eigensolve time versus the *complete* online
    time of both NNs. POD assembly and reconstruction are excluded from the
    plotted POD series, but are still measured and written to CSV separately.
    """
    out = Path(out)
    figures = out / 'figures'
    figures.mkdir(exist_ok=True)
    remove_retired_pod_plots(figures)
    series = _timing_matrix(timings, 'eigensolve_only')
    x = series.index.to_numpy()
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for method in ('POD-Galerkin', *NN_METHODS):
        display_name = ('POD-Galerkin (reduced eigensolve only)'
                        if method == 'POD-Galerkin' else method)
        ax.plot(x, series[method], marker='o', markersize=3,
                linewidth=1.5, label=display_name, color=PALETTE[method])
    ax.set(xlabel='Held-out geometry ID', ylabel='Time (ms per geometry)',
           title='Per-geometry CPU timing: reduced POD eigensolve vs neural networks')
    ax.grid(alpha=.18)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, figures / POD_TIMING_FIGURE)


def _wavefunction_modal_sum_figure(figdir, geometry_id, state, grid, length,
                                   ref, pod, global_wave, local_wave, num_modes):
    """Actual signed wavefunction (not probability density) in the x-y domain.

    Each approximation is the full sum a_j * eta_j. All panels share one
    zero-centered color scale; signs were matched to DNS solely for display.
    """
    names = ['DNS reference', 'POD-Galerkin', 'Global POD-NN', 'Local POD-NN']
    waves = [ref, pod, global_wave, local_wave]
    vmax = max(float(np.max(np.abs(w))) for w in waves)
    vmax = max(vmax, 1e-12)
    nx = grid
    # A physically meaningful horizontal section: the row containing the
    # largest absolute DNS amplitude. This choice affects plotting only.
    ix_max = int(np.argmax(np.abs(ref)))
    yrow = ix_max % nx  # Fortran-order flattened (y,x) grid
    # Flattening is Fortran-order, so y index varies fastest.
    xs = np.arange(nx) * length / nx
    y_nm = yrow * length / nx
    fig = plt.figure(figsize=(16, 10.8))
    gs = fig.add_gridspec(3,4, height_ratios=[1.0, .79, .75], hspace=.39, wspace=.20,
                          left=.06, right=.91, bottom=.07, top=.87)
    first_images = []
    for j,(name,wave) in enumerate(zip(names,waves)):
        field = wave.reshape((nx,nx),order='F')
        ax=fig.add_subplot(gs[0,j])
        im=ax.imshow(field,origin='lower',extent=[0,length,0,length],
                     cmap='RdBu_r',vmin=-vmax,vmax=vmax,interpolation='nearest')
        ax.axhline(y_nm,color='k',linewidth=.75,linestyle=':',alpha=.75)
        ax.set_title(name,fontsize=11.5)
        ax.set_xlabel('x (nm)')
        if j==0:ax.set_ylabel('y (nm)')
        first_images.append(im)
        if j>=2:
            ax=fig.add_subplot(gs[1,j])
            diff=field-ref.reshape((nx,nx),order='F')
            ax.imshow(diff,origin='lower',extent=[0,length,0,length],
                      cmap='PuOr',vmin=-2*vmax,vmax=2*vmax,interpolation='nearest')
            ax.set_title('NN signed residual vs DNS',fontsize=10)
            ax.set_xlabel('x (nm)')
        elif j==1:
            ax=fig.add_subplot(gs[1,j]);ax.axis('off')
            ax.text(.5,.55,'POD modal reconstruction\nfrom reduced eigenvectors',
                    ha='center',va='center',fontsize=11)
        else:
            ax=fig.add_subplot(gs[1,0]);ax.axis('off')
            ax.text(0.03,.92,'Wavefunction construction',fontsize=11,weight='bold',va='top')
            ax.text(0.03,.74,r'$\psi_\mathrm{POD}=\sum_{j=1}^{M}a_j^\mathrm{POD}\eta_j$',fontsize=13,va='top')
            ax.text(0.03,.40,r'$\widehat{\psi}_\mathrm{NN}=\sum_{j=1}^{M}\hat a_j^\mathrm{NN}\eta_j$',fontsize=13,va='top')
            ax.text(0.03,.07,f'M = {num_modes} modes',fontsize=11,va='bottom')
    # One centerline for all methods; 1D real wavefunction, not |psi|^2.
    line_ax=fig.add_subplot(gs[2,:]);colors=['#333333','#898f68','#577a98','#348887']
    for wave,name,col in zip(waves,names,colors):
        field=wave.reshape((nx,nx),order='F')
        line_ax.plot(xs,field[yrow,:],label=name,color=col,linewidth=2 if name=='Local POD-NN' else 1.5)
    line_ax.axhline(0,color='gray',lw=.75)
    line_ax.set(xlabel='x (nm)',ylabel='Signed wavefunction amplitude',
                title=f'Horizontal wavefunction profile at y = {y_nm:.2f} nm (dotted lines above)')
    line_ax.grid(alpha=.2);line_ax.legend(ncol=4,fontsize=9,loc='upper right')
    # Colorbar for row-one heatmaps, separate from the residual colors.
    cb_ax=fig.add_axes([.924,.59,.012,.27])
    fig.colorbar(first_images[0],cax=cb_ax,label='Signed ψ amplitude')
    fig.suptitle(f'Geometry {geometry_id} | eigenstate {state+1}: signed wavefunction from POD modal sums\n'
                 'DNS reference vs POD and learned modal coefficients (NN residuals shown)',fontsize=14)
    path=figdir/f'wavefunction_sum_a_eta_geometry_{geometry_id:04d}_state_{state+1:02d}.png'
    fig.savefig(path,dpi=180)
    plt.close(fig)



def _modal_terms_figure(figdir, geometry_id, state, phi, grid, length,
                        pod_a, local_a, pod_sign, local_sign):
    """Show actual terms a_j * eta_j and their full signed sum for POD/local NN.

    Select the four most influential modes under the COMBINED POD/local
    coefficient magnitudes, then aggregate the remaining M-4 contributions.
    Each column uses one shared symmetric color scale across both methods.
    Display signs are consistently applied to all modal terms and totals.
    """
    from .wavefunction import modal_sum

    modes=phi.shape[1]
    top=min(4,modes)
    highest=np.argsort(np.abs(pod_a)+np.abs(local_a))[-top:][::-1]
    methods=[('POD-Galerkin',pod_a*pod_sign),('Local POD-NN',local_a*local_sign)]
    maps=[]
    titles=[rf'$a_{{{j+1}}}\eta_{{{j+1}}}(x,y)$' for j in highest]
    titles += [f'Remaining {modes-top} modes',f'Full sum ({modes} modes)']
    for _,a in methods:
        selected=[a[int(j)]*phi[:,int(j)] for j in highest]
        total=modal_sum(phi,a)
        other=total-np.sum(selected,axis=0)
        maps.append(selected+[other,total])
    ncol=top+2
    fig,axes=plt.subplots(2,ncol,figsize=(3.0*ncol,6.1),sharex=True,sharey=True)
    for j in range(ncol):
        lim=max(float(np.max(np.abs(maps[0][j]))),float(np.max(np.abs(maps[1][j]))),1e-12)
        for i,(name,_) in enumerate(methods):
            ax=axes[i,j]
            image=maps[i][j].reshape((grid,grid),order='F')
            ax.imshow(image,origin='lower',extent=[0,length,0,length],
                      cmap='RdBu_r',vmin=-lim,vmax=lim,interpolation='nearest')
            if i==0:ax.set_title(titles[j],fontsize=10)
            if j==0:ax.set_ylabel(name+'\ny (nm)',fontsize=10)
            if i==1:ax.set_xlabel('x (nm)')
    fig.suptitle(f'Geometry {geometry_id} | eigenstate {state+1} | signed wavefunction modal decomposition\n'
                 r'Left: individual $a_j\eta_j$. Right: $\sum_{j=1}^{M}a_j\eta_j$. '
                 '(Each column shares its color scale across methods.)',fontsize=12)
    fig.subplots_adjust(left=.075,right=.99,top=.82,bottom=.09,wspace=.12,hspace=.18)
    fig.savefig(figdir/f'wavefunction_modal_contributions_geometry_{geometry_id:04d}_state_{state+1:02d}.png',dpi=170)
    plt.close(fig)


def _wavefunction_surface_figure(figdir,geometry_id,state,grid,length,ref,pod,global_wave,local_wave):
    """Three-dimensional surfaces of the *signed* real wavefunction amplitude."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    names=['DNS reference','POD: Σ a η','Global NN: Σ â η','Local NN: Σ â η']
    waves=[ref,pod,global_wave,local_wave]
    vlim=max(max(float(np.max(np.abs(w))) for w in waves),1e-12)
    x=np.arange(grid)*length/grid
    X,Y=np.meshgrid(x,x,indexing='xy')
    fig=plt.figure(figsize=(16,4.9))
    cmap=plt.get_cmap('RdBu_r')
    for k,(name,wave) in enumerate(zip(names,waves)):
        ax=fig.add_subplot(1,4,k+1,projection='3d')
        Z=wave.reshape(grid,grid,order='F')
        ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap=cmap,
                        vmin=-vlim,vmax=vlim,linewidth=0,antialiased=True)
        ax.set(xlabel='x (nm)',ylabel='y (nm)',zlabel='ψ(x,y)',zlim=(-vlim,vlim))
        ax.set_title(name,fontsize=11)
        ax.view_init(elev=32,azim=-62)
    fig.suptitle(f'Geometry {geometry_id} | eigenstate {state+1}: signed wavefunction ψ(x,y)',fontsize=14)
    fig.subplots_adjust(left=.02,right=.98,top=.87,bottom=.05,wspace=.12)
    fig.savefig(figdir/f'wavefunction_3d_geometry_{geometry_id:04d}_state_{state+1:02d}.png',dpi=160)
    plt.close(fig)


def make_reconstruction_figures(out, data, phi, red, models, labels, split, cfg, local_predictor):
    """DNS and reconstructed ψ(x,y) = sum_j a_j * η_j(x,y) on held-out cases.

    The POD coefficients are eigenvectors of the reduced Hamiltonian, while
    both neural coefficient arrays come from trained model outputs. All three
    methods share the *same* train-only POD spatial basis. DNS is used only as
    a plotted reference and to choose a display sign; it never enters a model.
    """
    from .evaluation import predict_global
    from .features import build_features
    from .wavefunction import modal_sum, pod_galerkin_amplitudes, align_display_sign

    out = Path(out); figdir=out/'figures'; figdir.mkdir(parents=True,exist_ok=True)
    diagdir=out/'data'/'wavefunction_reconstructions';diagdir.mkdir(parents=True,exist_ok=True)
    grid=int(cfg['physics']['grid']);length=float(cfg['physics']['length_nm'])
    num_modes=int(phi.shape[1]);nstates=int(cfg['physics']['states'])
    test=split['test'];nselected=min(len(test),int(cfg['runtime'].get('reconstruction_geometries',2)))
    selected_geometry_ids=list(test[:nselected])
    selected_states=sorted(set(int(x)-1 for x in cfg['runtime'].get('reconstruction_states',[1,nstates])))
    for geom_id in selected_geometry_ids:
        one_features=build_features({'params':data['params'][geom_id:geom_id+1]},
                                    red[geom_id:geom_id+1],cfg)
        global_coeff,_=predict_global(models['global'],one_features,cfg)
        local_coeff,_=local_predictor.predict(one_features)
        pod_coeff=pod_galerkin_amplitudes(red[geom_id], nstates)
        # The ENTIRE signed wavefunction comes from sum(a[j]*eta[j](x,y)):
        psi_pod=modal_sum(phi,pod_coeff)
        psi_global=modal_sum(phi,global_coeff[0])
        psi_local=modal_sum(phi,local_coeff[0])
        gt=data['psi'][geom_id]
        names=['DNS', 'POD-Galerkin', 'Global POD-NN', 'Local POD-NN']
        for state in selected_states:
            if not (0 <= state < gt.shape[0]): continue
            reference=np.asarray(gt[state],dtype='float64')
            p_raw=np.asarray(psi_pod[state],dtype='float64')
            g_raw=np.asarray(psi_global[state],dtype='float64')
            l_raw=np.asarray(psi_local[state],dtype='float64')
            p,psign=align_display_sign(reference,p_raw)
            g,gsign=align_display_sign(reference,g_raw)
            l,lsign=align_display_sign(reference,l_raw)
            fields=[reference,p,g,l]
            vmax=max(float(np.max(np.abs(w))) for w in fields)
            density_max=max(float(np.max(w*w)) for w in fields)
            vmax=max(vmax,1e-12);density_max=max(density_max,1e-12)
            # Reserve a dedicated left-hand column for the two colorbars.
            # Never attach colorbars to auto-resized axes: that can overlap
            # the right-most reconstruction panels in narrow exports.
            fig=plt.figure(figsize=(16.6,7.8))
            grid_spec=fig.add_gridspec(2,5,width_ratios=[.06,1,1,1,1],
                left=.085,right=.984,top=.83,bottom=.10,hspace=.28,wspace=.14)
            bar_wave=fig.add_subplot(grid_spec[0,0])
            bar_density=fig.add_subplot(grid_spec[1,0])
            axes=np.empty((2,4),dtype=object)
            for ii in range(2):
                for jj in range(4):
                    axes[ii,jj]=fig.add_subplot(grid_spec[ii,jj+1])
            field_titles=['DNS reference',
                          r'POD: $\sum_j a_j^{\mathrm{POD}}\eta_j$',
                          r'Global NN: $\sum_j \hat a_j^{\mathrm{G}}\eta_j$',
                          r'Local NN: $\sum_j \hat a_j^{\mathrm{L}}\eta_j$']
            for j,(name,wave) in enumerate(zip(field_titles,fields)):
                arr=wave.reshape((grid,grid),order='F')
                im_wave=axes[0,j].imshow(arr,origin='lower',extent=[0,length,0,length],
                                          cmap='RdBu_r',vmin=-vmax,vmax=vmax,interpolation='nearest')
                im_density=axes[1,j].imshow(arr*arr,origin='lower',extent=[0,length,0,length],
                                             cmap='viridis',vmin=0,vmax=density_max,interpolation='nearest')
                # Numeric accuracy annotations are NN-only. POD-Galerkin remains
                # a spatial reconstruction baseline, not an accuracy plot series.
                if j == 0:
                    caption = '\nreference'
                elif j == 1:
                    caption = ''
                else:
                    from .evaluation import percent_lse
                    error = float(percent_lse(reference,wave,sign_invariant=True))
                    caption = f'\nLSE={error:.2f}%'
                axes[0,j].set_title(name + caption, fontsize=10)
                axes[1,j].set_xlabel('x (nm)')
                if j==0:
                    axes[0,j].set_ylabel('y (nm) | signed wavefunction ψ')
                    axes[1,j].set_ylabel('y (nm) | density |ψ|²')
            fig.suptitle(f'Geometry {int(geom_id)} · eigenstate {state+1} · {num_modes} POD modes\n'
                         'Signed ψ reconstructed as the full modal sum a × η; shared color scales by row',fontsize=13)
            cb_wave=fig.colorbar(im_wave,cax=bar_wave)
            cb_density=fig.colorbar(im_density,cax=bar_density)
            for cb in (cb_wave,cb_density):
                cb.ax.yaxis.set_ticks_position('left')
                cb.ax.yaxis.set_label_position('left')
                cb.ax.tick_params(labelsize=8,pad=2)
            cb_wave.set_label('Signed ψ',fontsize=9,labelpad=4)
            cb_density.set_label('Density |ψ|²',fontsize=9,labelpad=4)
            fig.savefig(figdir/f'reconstruction_geometry_{geom_id:04d}_state_{state+1:02d}.png',dpi=180)
            plt.close(fig)

            # Store the unmodified raw coefficient and reconstructed fields so
            # an external researcher can recompute each modal sum independently.
            np.savez_compressed(diagdir/f'wavefunction_geometry_{geom_id:04d}_state_{state+1:02d}.npz',
                geometry_id=int(geom_id),state_index=int(state+1),grid=grid,
                n_modes=num_modes,psi_dns=reference.astype('float32'),
                a_pod=pod_coeff[state].astype('float32'),
                a_global=global_coeff[0,state].astype('float32'),
                a_local=local_coeff[0,state].astype('float32'),
                psi_pod=p_raw.astype('float32'),psi_global=g_raw.astype('float32'),
                psi_local=l_raw.astype('float32'),
                display_sign_pod=psign,display_sign_global=gsign,display_sign_local=lsign)

            _wavefunction_modal_sum_figure(figdir,int(geom_id),state,grid,length,
                                           reference,p,g,l,num_modes)
            _modal_terms_figure(figdir,int(geom_id),state,phi,grid,length,
                                pod_coeff[state],local_coeff[0,state],psign,lsign)
            if cfg['runtime'].get('reconstruction_3d',True):
                _wavefunction_surface_figure(figdir,int(geom_id),state,grid,length,reference,p,g,l)


def make_training_figures(out):
    """Loss and direct held-out-validation trends, split by model and stage."""
    out=Path(out);f=out/'metrics'/'training_history.csv'
    if not f.exists():return
    df=pd.read_csv(f)
    if df.empty:return
    figdir=out/'figures'
    for metric,filename,title,yscale in [
        ('loss','training_loss_by_model.png','Stage-wise optimization objective','log'),
        ('validation_wave_lse_pct','validation_wave_error_by_model.png',
         'Validation wavefunction LSE (%) — sign aligned','linear'),
        ('validation_energy_rel_abs','validation_energy_error_by_model.png',
         'Relative validation energy error (%)','linear')]:
        if metric not in df:continue
        fig,axes=plt.subplots(1,2,figsize=(13,4.5))
        for k,stage in enumerate(('coeff','full')):
            part=df[df.stage==stage]
            for name,group in part.groupby('model'):
                v=group[metric].to_numpy(dtype=float)
                if metric=='validation_energy_rel_abs': v=100.*v
                valid=np.isfinite(v) & (v>0 if yscale=='log' else np.ones(len(v),dtype=bool))
                if valid.any():axes[k].plot(group.epoch.to_numpy()[valid],v[valid],label=name,linewidth=1.4)
            ylabel = ('Wavefunction LSE (%)' if metric=='validation_wave_lse_pct' else
                      'Relative energy error (%)' if metric=='validation_energy_rel_abs' else
                      'Training objective')
            axes[k].set(xlabel='Epoch',ylabel=ylabel,title=stage+' stage')
            axes[k].grid(alpha=.2)
            if yscale=='log':axes[k].set_yscale('log')
            if axes[k].lines:axes[k].legend(fontsize=8)
        fig.suptitle(title)
        _save(fig,figdir/filename)
