"""Audit and publish all paired manuscript-I/Q outcomes, including failures."""
import math
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator,FormatStrFormatter
import manuscript_iq_study_v1 as study
import paper_uav_charts as ledger

ROOT,read,write,sha=study.ROOT,study.read,study.write,study.sha
DEST=ROOT/'final_final_charts/manuscript_iq_pilot'
METHODS={'proposed':'Proposed','iq_only':'IQ-only','paper_iq':'Manuscript-IQ'}
COLORS=['#4C72B0','#EFAA54','#66B366'];MARKERS=['o','s','v'];LINES=['-','--','-.']


def source(case,ds,method):
    if method=='proposed':return ROOT/read(study.OUT/'protocol.json')['original_Proposed'][ds]
    key='gate_select16' if method=='iq_only' else 'paper_iq'
    return study.OUT/method/key/'cases/baseline'/(ds.lower()+'_seed0')/study.study.PROPOSED


def audit(arm,old):
    for name in ['plan.json','requests.json','initial20/final.pt','reference200/final.pt','feature_cache/features.pt']:
        assert sha(arm.parent/name)==sha(old.parent/name),name
    assert read(arm.parents[1]/'settings.json')==read(old.parents[1]/'settings.json')
    ts=[read(p) for p in sorted((arm/'transactions').glob('*.json'))]
    works={r['transaction']:r for r in [read(p) for p in (arm/'work_attempts').glob('*.json')] if 'transaction' in r}
    for t in ts:
        r=t['radio'];eps=np.array(r['component_epsilon']);v=np.array(r['block_variance']);x=np.array(r['transceiver_x'])
        assert np.allclose(v.sum(1),r['component_mse'],rtol=1e-10,atol=1e-30)
        assert np.all(v.sum(1)<=eps**2*(1+2e-6)+1e-30)
        weights=np.array(r['normalized_curvature']);dist=weights@v.sum(0)
        assert np.allclose(dist,r['distortion'],rtol=1e-10,atol=1e-30)
        assert np.all(dist<=np.array(r['normalized_distortion_limits'])*(1+2e-6)+1e-30)
        assert math.isclose(sum(r['user_RF_energy_j']),r['energy_bound_j'],rel_tol=1e-10,abs_tol=1e-20)
        assert r['energy_bound_j']<=.1*(1+1e-6) and r['max_power_w']<=.32*(1+1e-6)
        assert r['max_user_energy_j']<=.5*(1+1e-6)
        assert r['uplink_s']<=1.31072+1e-12 and r['frame_count']*128/1e6<=.01+1e-12
        assert np.allclose(v[v>0],(128/(2*1e6*x))[v>0],rtol=1e-10)
        assert all(b<=a*(1+1e-8) for a,b in zip(r['BCD_objectives'],r['BCD_objectives'][1:]))
        curvature=works[t['sequence']]['compute']['manuscript_curvature']
        assert not curvature['certified'] and curvature['hvp_calls'] in [2,4]
        assert sum(works[t['sequence']]['compute']['FU_control_scalar_ops'])>0
    return dict(passed=True,transactions=len(ts),inputs_shared=True,all_physical_and_surrogate_constraints_verified=True,
                all_noise_samples_retained=True,curvature_cost_charged=True)


def curves(arm):
    initial=read(arm.parent/'initial20/rounds.json')
    out=[dict(round=r['round'],evaluation=r['evaluation']) for r in initial if r.get('evaluation')]
    for p in sorted((arm/'evaluations').glob('*.json')):
        r=read(p);out.append(dict(round=r['completed_FL_rounds'],evaluation=r['evaluation']))
    if (arm/'service_completion.json').exists():
        r=read(arm/'service_completion.json');out.append(dict(round=r['completed_FL_rounds'],evaluation=r['evaluation']))
    return out


def canvas(ylabel):
    fig,axes=plt.subplots(1,2,figsize=(7.4,3.3615))
    fig.subplots_adjust(left=.125,right=.985,bottom=.24,top=.84,wspace=.45)
    for i,(ax,ds) in enumerate(zip(axes,study.published.DATASETS)):
        ax.set_ylabel(ylabel,fontsize=14);ax.tick_params(labelsize=12)
        ax.grid(True,axis='y',ls=':',lw=.5,color='#cccccc');ax.set_axisbelow(True)
        fig.text([.29,.80][i],.02,f'({chr(97+i)}) {ds}',ha='center',fontsize=14)
    return fig,axes


def finish(fig,name,legend=True):
    if legend:
        handles,labels=fig.axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,ncol=3,loc='upper center',bbox_to_anchor=(.53,1.0),frameon=False,fontsize=12)
    for ext in ['pdf','png','svg']:fig.savefig(DEST/(name+'.'+ext),dpi=350)
    plt.close(fig)


def main():
    protocol=study.register();DEST.mkdir(parents=True,exist_ok=True)
    assert read(study.OUT/'radio_checks.json')['passed']
    shutil.copyfile(study.OUT/'radio_checks.json',DEST/'radio_checks.json')
    ledger.source=source;ledger.METHODS=METHODS
    rows=[];checks=[];series=[]
    for ds in study.published.DATASETS:
        for method,label in METHODS.items():
            arm=source('baseline',ds,method);outcome=read(arm/'outcome.json')
            if outcome['status']!='completed':
                rows.append(dict(dataset=ds,method=method,label=label,status=outcome['status'],source=str(arm.relative_to(ROOT))))
                continue
            row=ledger.ledger('baseline',ds,method)
            tx=[read(p) for p in sorted((arm/'transactions').glob('*.json'))]
            row.update(FL_rounds=read(arm/'manifest.json')['completed_FL_rounds'],
                uplink_s=sum(t['radio']['uplink_s'] for t in tx),
                compute_s=sum(t['timing']['compute_s'] for t in tx),
                downlink_s=sum(t['timing']['downlink_s'] for t in tx),
                RF_energy_j=sum(t['radio']['energy_bound_j'] for t in tx),
                optimizer_wall_s=sum(t['radio'].get('optimizer_wall_s',0) for t in tx))
            works=[read(p) for p in (arm/'work_attempts').glob('*.json')]
            compute=sum(float(ledger.physical.compute_cost(w['compute']['device_macs'])[0].sum()) for w in works)
            row['manuscript_scope_energy_j']=compute+row['RF_energy_j']/.35+study.radio.HOVER*(row['compute_s']+row['uplink_s'])
            row['manuscript_scope_delay_s']=row['compute_s']+row['uplink_s']
            rows.append(row)
            if method!='proposed':checks.append(dict(dataset=ds,method=method,**audit(arm,source('baseline',ds,'proposed'))))
            series.append(dict(dataset=ds,method=method,points=curves(arm)))
    changes=[]
    for r in rows:
        if r['status']!='completed' or r['method']=='proposed':continue
        ref=next(p for p in rows if p['dataset']==r['dataset'] and p['method']=='proposed')
        changes.append(dict(dataset=r['dataset'],method=r['method'],mR_change_pp=r['retained_mR_percent']-ref['retained_mR_percent'],
            KL_change=r['forget_KL']-ref['forget_KL'],KL_reduction_percent=100*(1-r['forget_KL']/ref['forget_KL']),
            energy_reduction_percent=100*(1-r['total_energy_j']/ref['total_energy_j']),delay_reduction_percent=100*(1-r['delay_s']/ref['delay_s'])))
    write(DEST/'metrics.json',rows);ledger.write_csv(DEST/'metrics.csv',rows)
    write(DEST/'changes.json',changes);write(DEST/'curve_data.json',series)
    write(DEST/'protocol.json',protocol);shutil.copyfile(study.OUT/'preflight.json',DEST/'preflight.json')
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'pdf.fonttype':42,'svg.fonttype':'none'})
    for name,metric,ylabel in [('01_round_mr_1x2','mR','Retained mR (%)'),('02_round_kl_1x2','KL_retrain_to_candidate','Forget-set KL')]:
        fig,axes=canvas(ylabel)
        for ax,ds in zip(axes,study.published.DATASETS):
            for i,(method,label) in enumerate(METHODS.items()):
                record=next((s for s in series if (s['dataset'],s['method'])==(ds,method)),None)
                if record is None:continue
                ps=[p for p in record['points'] if metric in p['evaluation'].get('retained' if metric=='mR' else 'forget',{})]
                x=[p['round'] for p in ps];y=[p['evaluation']['retained' if metric=='mR' else 'forget'][metric]*(100 if metric=='mR' else 1) for p in ps]
                ax.plot(x,y,color=COLORS[i],ls=LINES[i],lw=1.6,label=label,marker=MARKERS[i],markevery=[],ms=5,mfc='white')
                markers={p['round']:p for p in ps if p['round']%25==0}
                ax.plot(list(markers),[p['evaluation']['retained' if metric=='mR' else 'forget'][metric]*(100 if metric=='mR' else 1) for p in markers.values()],ls='none',marker=MARKERS[i],ms=5,mfc='white',mec=COLORS[i])
            ax.set_xlim(0,200);ax.set_xticks([0,50,100,150,200]);ax.set_xlabel('FL round',fontsize=14)
            if metric!='mR':ax.set_ylim(bottom=0)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5));ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f' if metric=='mR' else '%.3f'))
            ticks=ax.get_yticks();ax.set_ylim(ticks[0],ticks[-1]);ax.set_yticks(ticks)
        finish(fig,name)
    for name,key,scale,label in [('03_energy_1x2','total_energy_j',.001,'Total energy (kJ)'),('04_delay_1x2','delay_s',1,'Delay (s)')]:
        fig,axes=canvas(label)
        for ax,ds in zip(axes,study.published.DATASETS):
            for i,(m,lab) in enumerate(METHODS.items()):
                r=next(r for r in rows if r['dataset']==ds and r['method']==m)
                if r['status']=='completed':ax.bar(i,r[key]*scale,color=COLORS[i],edgecolor='#222222',lw=.7,width=.55,hatch=['//','\\','..'][i],label=lab)
            ax.set_xticks([]);ax.set_xlim(-.6,2.6);ax.set_ylim(bottom=0)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ticks=ax.get_yticks();ax.set_ylim(0,ticks[-1]);ax.set_yticks(ticks[ticks>=0])
        finish(fig,name)
    text='# Manuscript I/Q pilot\n\nSame 200 FL rounds and 45 requests × 5 FU steps; paired development seed 0. Existing Proposed and its published charts are unchanged.\n\n'
    text+='IQ-only keeps the current P-KS learning policy and changes to the manuscript I/Q transceiver and empirical task-distortion design. Manuscript-IQ also uses the manuscript cosine-forgetting/embedding-consistency objective, uniform FL sampling, and mini-batch-size aggregation weights.\n\n'
    text+='The manuscript omits numerical curvature bounds, loss-perturbation budgets, expansion grid and initialization. The operational choices are recorded in protocol.json, before observing the candidate results. HVP-based curvature is an empirical surrogate, not a certified Hessian bound or a forgetting guarantee. Both I/Q streams count toward the SAME block power limit. BCD solves are local, not globally optimal.\n\n'
    text+='Primary energy/delay use the existing active FL/FU scope including downlink, circuits and all executed curvature costs; travel/collection/common preflight are excluded for every method. The narrower manuscript formula is exported separately in metrics.csv; mixing these scopes would be misleading. Solver wall time is reported separately and is not calibrated to UAV energy. All KL values compare with the same reporting-only retained retraining reference200.\n\n'
    text+='Curvature and update-norm control information is assumed available to the head; its signaling payload is not modeled. The simulated curvature probes access the local training objectives jointly, so this pilot is not a fully decentralized curvature-estimation protocol.\n\n'
    text+='| Dataset | Method | Status | mR (%) | Loss | Forget KL | Energy (kJ) | Delay (s) | Uplink (s) |\n|---|---|---|---:|---:|---:|---:|---:|---:|\n'
    for r in rows:
        if r['status']=='completed':text+=f"| {r['dataset']} | {r['label']} | completed | {r['retained_mR_percent']:.3f} | {r['retained_loss']:.5f} | {r['forget_KL']:.6f} | {r['total_energy_j']/1000:.3f} | {r['delay_s']:.3f} | {r['uplink_s']:.3f} |\n"
        else:text+=f"| {r['dataset']} | {r['label']} | {r['status']} | — | — | — | — | — | — |\n"
    text+='\nThe radio-only change leaves model quality close to Proposed in this seed. The full manuscript changes the learning/forgetting tradeoff: lower reference KL accompanies lower retained mR. It does not dominate Proposed in energy or delay under the declared constants and BCD initialization. The two interventions do not isolate the individual effects of the FU objective, FL sampler and aggregation weights.\n\n'
    text+='Reproduce with `manuscript_iq_study_v1.py run --variant iq_only`, `manuscript_iq_study_v1.py run --variant paper_iq`, then `manuscript_iq_report_v1.py` in `tools/uav_convergence_v2`, using the registered local source experiments.\n'
    (DEST/'README.md').write_text(text)
    names=['01_round_mr_1x2','02_round_kl_1x2','03_energy_1x2','04_delay_1x2']
    (DEST/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{max-width:1100px;margin:30px auto;font:16px system-ui}img{width:100%}</style><h1>Manuscript I/Q pilot</h1><p><a href="README.md">Results and conditions</a></p>'+''.join(f'<a href="{n}.pdf"><img src="{n}.png"></a>' for n in names))
    write(DEST/'verification.json',dict(passed=True,arms=checks,source_sha256={str(Path(f).relative_to(ROOT)):sha(Path(f)) for f in [__file__,study.__file__,study.radio.__file__]},input_sha256=ledger.SOURCES))
    write(study.OUT/'finished.json',dict(passed=True,new_arms=len(checks),compared_arms=len(rows),all_outcomes_reported=True))
    print(rows);print(changes)


if __name__=='__main__':main()
