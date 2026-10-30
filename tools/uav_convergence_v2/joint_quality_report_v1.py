"""Publish every development setting, failures, costs and fixed-reference results."""
import csv
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
import joint_quality_control_v1 as study
import joint_quality_audit_v1 as audit
import forgetting_quality_report_v1 as style

ROOT,read,write,sha=study.ROOT,study.read,study.write,study.sha
DEST=ROOT/'final_final_charts/joint_quality_control'
FAMILIES={'first':study.OUT,'context':ROOT/'results/joint_quality_context_v2_20261030',
    'focused':ROOT/'results/joint_quality_focused_v3_20261030','loss':ROOT/'results/joint_quality_loss_v4_20261030'}
PLOT=[]


def collect():
    original=[];physical=[];screens=[];paired=[];provenance={}
    for family,path in FAMILIES.items():
        protocol=read(path/'protocol.json');count=len(protocol['variants'])
        shutil.copyfile(path/'protocol.json',DEST/'data'/f'{family}_protocol.json')
        for ds in study.audit.DATASETS:
            files=sorted((path/'screen'/ds.lower()).glob('*/finished.json'));assert len(files)==count
            for file in files:
                row=read(file);variant=row['variant'];row.update(family=family,mode='exact_screen',
                    checkpoint_sha256=sha(file.parent/'final.pt'),source=str(file.relative_to(ROOT)))
                row['configuration']=json.dumps(row['configuration'],sort_keys=True);screens.append(row)
                provenance[str(file.relative_to(ROOT))]=sha(file)
                assert row['FL_rounds']==200 and row['FU_steps']==225
                if variant=='control':assert row['exact_control_reproduced']
    for ds in study.audit.DATASETS:
        for r in read(study.audit.OUT/ds.lower()/'summary.json'):
            m=r['metrics'];cost=read(study.audit.current.source('baseline',ds,r['method'])/'energy_and_delay.json')
            original.append(dict(dataset=ds,method=r['method'],label=r['label'],
                retained_mR=100*m['retained']['mR'],retained_loss=m['retained']['loss'],
                common_KL=m['forget']['common_KL'],matched_KL=m['forget']['matched_KL'],
                worst_KL=r['worst_request_common_KL'],worst_matched_KL=r['worst_request_matched_KL'],
                energy_kJ=cost['total_energy_j']/1000,active_delay_s=cost['delay_s']))
        summary=read(audit.OUT/ds.lower()/'summary.json')
        for r in summary['physical']:
            row={k:v for k,v in r.items() if k not in ['verification','cost']}
            row.update(energy_kJ=r['cost']['total_energy_j']/1000,active_delay_s=r['cost']['delay_s'],
                FL_rounds=r['verification']['FL_rounds'],FU_steps=r['verification']['FU_steps'],
                network_cap_overrun_transactions=r['cost']['network_cap_overrun_transactions'])
            if row['variant']==audit.VARIANT:
                row['matched_KL']=summary['paired']['physical_fu']['metrics']['forget']['matched_KL']
                row['worst_matched_KL']=summary['paired']['physical_fu']['worst_matched_KL']
            physical.append(row)
            shutil.copyfile(audit.OUT/ds.lower()/'physical'/r['stage']/r['variant']/'request_kl.json',
                DEST/'data'/f'{ds}_{r["stage"]}_{r["variant"]}_requests.json')
        for name,r in summary['paired'].items():
            m=r['metrics'];paired.append(dict(dataset=ds,condition=name,retained_mR=100*m['retained']['mR'],
                forget_mR=100*m['forget']['mR'],common_KL=m['forget']['common_KL'],matched_KL=m['forget']['matched_KL'],
                association_margin=m['forget']['association_margin'],worst_common_KL=r['worst_common_KL'],
                worst_matched_KL=r['worst_matched_KL'],checkpoint_sha256=r['checkpoint_sha256']))
        shutil.copyfile(audit.OUT/ds.lower()/'summary.json',DEST/'data'/f'{ds}_audit.json')
    for p in ROOT.glob('tools/uav_convergence_v2/joint_quality_*.py'):
        provenance[str(p.relative_to(ROOT))]=sha(p)
    for file in audit.PHYSICAL.glob('*/*/protocol.json'):
        shutil.copyfile(file,DEST/'data'/f'physical_{file.parts[-3]}_{file.parts[-2]}_protocol.json')
    shutil.copyfile(audit.OUT/'protocol.json',DEST/'data/audit_protocol.json')
    style.table(DEST/'data/all_122_exact_screens.csv',screens)
    style.table(DEST/'data/original_physical.csv',original)
    style.table(DEST/'data/all_physical_candidates.csv',physical)
    style.table(DEST/'data/paired_controls.csv',paired)
    write(DEST/'data/provenance.json',provenance)
    return original,physical,screens,paired


def values(original,physical,ds):
    old=[next(r for r in original if r['dataset']==ds and r['method']==m) for m in study.audit.METHODS]
    new=next(r for r in physical if r['dataset']==ds and r['variant']==audit.VARIANT)
    return old,new


def comparison(original,physical,key,ylabel,name,factor=1.):
    allvalues=[r[key]*factor for r in original]+[r[key]*factor for r in physical if r['variant']==audit.VARIANT]
    legend=[Patch(facecolor='white',edgecolor='#444444',hatch='//',label='Original'),
            Patch(facecolor='#4C72B0',edgecolor='#444444',label='Controlled')]
    fig,axes=style.layout(ylabel,allvalues,legend)
    for ax,ds in zip(axes,study.audit.DATASETS):
        old,new=values(original,physical,ds)
        if key=='retained_mR':
            ticks=np.arange(15.,18.01,.5) if ds=='CapERA' else np.arange(13.,14.01,.2)
            ax.set_ylim(ticks[0],ticks[-1]);ax.set_yticks(ticks);ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        for j,r in enumerate(old):
            ax.bar(j-.15 if j==0 else j,r[key]*factor,width=.27 if j==0 else .54,
                color=style.COLORS[j],alpha=.65,edgecolor='#222222',lw=.65,hatch='//')
            PLOT.append(dict(figure=name,dataset=ds,method=r['label'],condition='original',metric=key,value=r[key],scale=factor))
        ax.bar(.15,new[key]*factor,width=.27,color=style.COLORS[0],edgecolor='#222222',lw=.65)
        PLOT.append(dict(figure=name,dataset=ds,method='Proposed',condition=audit.VARIANT,metric=key,value=new[key],scale=factor))
    style.save(fig,name)


def controls(paired):
    name='32_paired_fu_control_kl_1x2';rows=[r for r in paired if r['condition']!='physical_fu']
    legend=[Patch(facecolor='#a8bbd7',edgecolor='#222',hatch='//',label='No FU'),
            Patch(facecolor='#4C72B0',edgecolor='#222',label='FU')]
    fig,axes=style.layout(r'Forget KL ($\times 10^{-3}$)',[r['common_KL']*1000 for r in rows],legend)
    for ax,ds in zip(axes,study.audit.DATASETS):
        ax.set_xlim(-.5,1.5);ax.set_xticks([0,1],['Common','Matched'])
        for j,key in enumerate(['common_KL','matched_KL']):
            for i,condition in enumerate(['exact_no_fu','exact_fu']):
                r=next(r for r in rows if r['dataset']==ds and r['condition']==condition)
                ax.bar(j+(i-.5)*.3,r[key]*1000,width=.28,color=['#a8bbd7','#4C72B0'][i],edgecolor='#222',lw=.65,hatch='//' if i==0 else '')
                PLOT.append(dict(figure=name,dataset=ds,condition=condition,metric=key,value=r[key],scale=1000))
    style.save(fig,name)


def scatter(original,physical,screens):
    name='33_development_tradeoff_1x2';fig,axes=style.layout('Retained mR (%)',[0,20])
    for ax,ds in zip(axes,study.audit.DATASETS):
        rows=[r for r in screens if r['dataset']==ds]
        ax.scatter([r['common_KL']*1000 for r in rows],[r['retained_mR'] for r in rows],s=13,
            color='#afb6c0',alpha=.75,linewidths=.3,edgecolors='white',label='Exact screens',zorder=2)
        old,new=values(original,physical,ds)
        ax.scatter(old[0]['common_KL']*1000,old[0]['retained_mR'],s=45,marker='o',facecolor='white',edgecolor='#4C72B0',lw=1.4,label='Original',zorder=4)
        ax.scatter(new['common_KL']*1000,new['retained_mR'],s=90,marker='*',color='#4C72B0',label='Controlled',zorder=5)
        for row in rows:PLOT.append(dict(figure=name,dataset=ds,condition=row['family']+'/'+row['variant'],metric='retained_mR',value=row['retained_mR'],x=row['common_KL']*1000))
        xt,dec=style.nice_range([r['common_KL']*1000 for r in rows]);ax.set_xlim(xt[0],xt[-1]);ax.set_xticks(xt);ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        yt,dec=style.nice_range([r['retained_mR'] for r in rows],False);ax.set_ylim(yt[0],yt[-1]);ax.set_yticks(yt);ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_xlabel(r'Forget KL ($\times 10^{-3}$)',fontsize=14,labelpad=2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,loc='lower center',bbox_to_anchor=(.537,.885),fontsize=12,frameon=False,columnspacing=1.1)
    style.save(fig,name)


def documentation(original,physical,screens,paired):
    lines=['# Joint mR/KL control — development results','',
        '**Result: the shared round100 FL switch retains mR leadership over the three unchanged comparators on both datasets and lowers KL. It does not achieve joint leadership on fixed-reference KL and mR.**','',
        '[Combined six-figure PDF](joint_quality_control_all.pdf) · [Gallery](index.html) · [All122 exact-screen outcomes](data/all_122_exact_screens.csv) · [Every physical candidate](data/all_physical_candidates.csv)','',
        'Current published Proposed remains P-KS. `Controlled` denotes the separate `uniform_after100` candidate: use the original priority16 sampler through round100, then uniform16 through200. The original gated FU rule, parallel AirComp, data, request schedule, temperature and power caps are unchanged. This is a fixed training schedule, not an automatic online KL controller.','',
        '| Dataset | Configuration | Retained mR (%) | Loss | Common KL | Worst-request KL | Matched KL | Energy (kJ) | Active delay (s) |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    improvements=[]
    for ds in study.audit.DATASETS:
        old,new=values(original,physical,ds)
        for r in [old[0],dict(new,label='Controlled'),*old[1:]]:
            lines.append(f"| {ds} | {r['label']} | {r['retained_mR']:.5f} | {r['retained_loss']:.5f} | {r['common_KL']:.7f} | {r['worst_KL']:.7f} | {r['matched_KL']:.7f} | {r['energy_kJ']:.3f} | {r['active_delay_s']:.3f} |")
        assert new['retained_mR']>=old[0]['retained_mR']-1e-9
        assert new['retained_mR']>max(r['retained_mR'] for r in old[1:])
        improvements.append(dict(dataset=ds,mR_change_pp=new['retained_mR']-old[0]['retained_mR'],
            common_KL_reduction_percent=100*(1-new['common_KL']/old[0]['common_KL']),
            worst_KL_reduction_percent=100*(1-new['worst_KL']/old[0]['worst_KL']),
            matched_KL_reduction_percent=100*(1-new['matched_KL']/old[0]['matched_KL']),
            loss_change=new['retained_loss']-old[0]['retained_loss'],
            leads_three_baselines_mR=True,leads_three_baselines_common_KL=new['common_KL']<min(r['common_KL'] for r in old[1:]),
            meets_common_KL_001=new['common_KL']<=.01))
    write(DEST/'data/improvements.json',improvements)
    lines+=['','The CapERA common KL remains slightly above0.01; its worst-request KL also exceeds0.01. AirScape satisfies both. Training loss increases versus original P-KS on both datasets, despite preserving/improving retained mR. Small mR gains are descriptive seed0 outcomes, not statistical significance.','',
        'The common reference is the same fixed retained-only200-round reference for every method. Matched KL uses each learning policy’s own retained-only retraining reference; the new schedule has a fresh200-round matched reference. It is a different counterfactual, not a replacement for common KL. AirScape controlled matched KL is best among the three unchanged baselines; CapERA remains behind ISCC-P2/AFL-OTA.','',
        '## FU intervention check','',
        '| Dataset | Exact condition | Forget mR (%) | Common KL | Matched KL |',
        '|---|---|---:|---:|---:|']
    for r in paired:
        if r['condition']=='physical_fu':continue
        lines.append(f"| {r['dataset']} | {r['condition']} | {r['forget_mR']:.5f} | {r['common_KL']:.7f} | {r['matched_KL']:.7f} |")
    lines+=['','FU-on still has larger common and matched KL than its paired exact no-FU control. It also has higher forget-set mR. Thus the improvement versus original P-KS supports reducing FL-induced distribution drift; it does **not** establish a positive forgetting effect. C9 bounds transmission perturbation, not equivalence to retraining. No-FU request commits are bookkeeping, not completed deletion.','',
        '## Scope and complete search','',
        'Four sequential exploratory families contain61 declared configurations including repeated controls, each on both datasets:122 exact-aggregation runs. All outcomes, including worse mR/KL and failed settings, are exported. Exact controls reproduce original exact-P-KS tensors bitwise. Exact screens do not supply physical energy/delay or power-feasibility claims.','',
        'Three shortlisted settings are physically rerun on both datasets, using the original AirComp/head/power optimization and actual noisy received updates. `bound100` lost CapERA mR leadership; `uniform_after100` is preferred here because it gives the largest verified KL reduction while maintaining the original mR level on both datasets. The third candidate, `rank0.5`, tests a symmetric CE/hard-negative margin mixture and is retained in the table below. These candidates are alternatives, not additional reference baselines.','',
        '| Dataset | Physical candidate | Retained mR (%) | Common KL | Worst-request KL |',
        '|---|---|---:|---:|---:|']
    for r in physical:lines.append(f"| {r['dataset']} | {r['variant']} | {r['retained_mR']:.5f} | {r['common_KL']:.7f} | {r['worst_KL']:.7f} |")
    lines+=['','The physical-v1 factory attempt failed before training while constructing an unused proxy-objective factory. Its script and local log are retained; physical-v2 restores the original factory for that unused objective and supplies the original two seed adapters. No failed attempt is silently overwritten.','',
        'All candidates use the same200 FL rounds,45 requests and225 prescribed FU cycles, existing known-forget-cohort proxy20, native video captions, full gallery and frozen pretrained encoders with trainable rank10 adapters. No evaluation reference, test caption or evaluation probability is used as a training teacher/gradient. References retrain adapters from initialization, not backbone pretraining from scratch.','',
        'This is adaptive development on the previously inspected seed0 cohorts. Shared settings across datasets do not make this an independent held-out confirmation. Further seeds and untouched evaluation cohorts are required before claiming a general lead. No per-dataset model selection, changed comparator, temperature-only calibration, favorable-query filtering or added training rounds is used.','',
        'Active energy uses the unchanged communication+computation+hover ledger; shared travel/collection/preflight are excluded. All physical runs retain0.1J network RF and0.32W device caps. Extra gradient/teacher work is charged through executed device MAC counts. Inherited FL_gradient_calls metadata counts nominal participating clients, not additional gradient passes for bounded variants; use work MAC ledgers for costs. Simulated energy is not a hardware measurement.','',
        '## Figures','',
        '| File | Meaning |','|---|---|',
        '|28_controlled_retained_mr_1x2|Actual final retained mR; original and controlled Proposed against unchanged comparators.|',
        '|29_controlled_common_kl_1x2|Primary fixed-reference forget KL; lower is better.|',
        '|30_controlled_matched_kl_1x2|Secondary policy-matched KL; references differ by learning policy.|',
        '|31_controlled_worst_kl_1x2|Maximum common-reference request KL over all45 requests.|',
        '|32_paired_fu_control_kl_1x2|Exact FU/no-FU causal diagnostic under the new FL schedule.|',
        '|33_development_tradeoff_1x2|All exact-screen candidates; two physical Proposed points identified separately.|','',
        '## Reproduction','',
        'Run the four `joint_quality_*` screen drivers with `.venv/bin/python`; use the physical-v2 driver for first/focused/context candidates and the loss-v4 physical action for loss candidates. Run `joint_quality_audit_v1.py`, then `joint_quality_report_v1.py` and `joint_quality_check_v1.py`. GPU drivers preserve interrupted trials rather than overwrite them. Protocol hashes, plot arrays, physical outcomes and causal controls are in `data/`; raw checkpoints and licensed feature caches remain local.','',
        'Conceptual motivation only: [CLIPErase (ACL2025)](https://aclanthology.org/2025.acl-long.1469/) for visual/text forgetting-retention-consistency terms and [SLUG (ICML2025)](https://proceedings.mlr.press/v267/cai25f.html) for VLM gradient considerations. The new controls are custom adaptations, not full reproductions of either paper.']
    (DEST/'README.md').write_text('\n'.join(lines)+'\n')
    names=[a['figure'] for a in style.AUDITS]
    captions=['Retained retrieval performance of the original and controlled learning schedules.',
        'Forget-set divergence from a common retained-only retraining reference.',
        'Forget-set divergence from policy-matched retained-only retraining references.',
        'Worst-request divergence from the common retraining reference.',
        'Effect of FU on forget-set divergence under exact aggregation.',
        'Retained retrieval performance and forget-set divergence across exploratory training controls.']
    (DEST/'captions.tex').write_text('\n\n'.join('% '+n+'\n\\caption{'+c+'}'for n,c in zip(names,captions))+'\n')
    body=''.join(f'<article><a href="{n}.pdf"><img src="{n}.png" alt="{c}"></a><p>{c}</p></article>' for n,c in zip(names,captions))
    (DEST/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Joint quality controls</title><style>body{font:16px Georgia;margin:30px auto;max-width:1000px;color:#26334a}img{width:100%}article{margin:38px 0}a{color:#345e9b}</style><h1>Joint quality controls</h1><p>The shared FL schedule preserves mR leadership against the three unchanged comparators and reduces KL. It does not lead common KL. Seed0 exploratory results; FU-on/off does not establish a forgetting benefit.</p><p><a href="README.md">Methods, complete results and limits</a> · <a href="joint_quality_control_all.pdf">All six figures</a></p>'+body+'</html>\n')


if __name__=='__main__':
    (DEST/'data').mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'pdf.fonttype':42,'svg.fonttype':'none'})
    style.DEST=DEST
    original,physical,screens,paired=collect()
    assert len(screens)==122 and len(physical)==6
    with PdfPages(DEST/'joint_quality_control_all.pdf') as pdf:
        style.COLLECTION=pdf
        comparison(original,physical,'retained_mR','Retained mR (%)','28_controlled_retained_mr_1x2')
        comparison(original,physical,'common_KL',r'Forget KL ($\times 10^{-3}$)','29_controlled_common_kl_1x2',1000)
        comparison(original,physical,'matched_KL',r'Forget KL ($\times 10^{-3}$)','30_controlled_matched_kl_1x2',1000)
        comparison(original,physical,'worst_KL',r'Worst-request KL ($\times 10^{-3}$)','31_controlled_worst_kl_1x2',1000)
        controls(paired);scatter(original,physical,screens)
    style.table(DEST/'data/plot_values.csv',PLOT);write(DEST/'data/geometry.json',style.AUDITS)
    documentation(original,physical,screens,paired)
    print(json.dumps(dict(figures=len(style.AUDITS),screens=len(screens),physical=len(physical),output=str(DEST))))
