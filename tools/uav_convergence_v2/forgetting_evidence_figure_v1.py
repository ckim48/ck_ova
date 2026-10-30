"""One scoped forgetting-quality figure; preserve non-leading outcomes/references."""
import csv
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import forgetting_quality_report_v1 as style

ROOT=style.ROOT
SOURCE=ROOT/'final_final_charts/joint_quality_control/data'
DEST=ROOT/'final_final_charts/forgetting_evidence'
NAME='34_forget_matched_retrain_1x2'


def read_csv(name):
    with (SOURCE/name).open() as f:return list(csv.DictReader(f))


def capera_detail(fig,ax,records):
    """Magnify exact values with points while keeping zero-baseline bars visible."""
    inset=ax.inset_axes([.20,.55,.57,.36],zorder=6)
    selected=[r for r in records if r['dataset']=='CapERA'][:3]
    for i,(r,color,marker) in enumerate(zip(selected,style.COLORS,['o','s','v'])):
        point,=inset.plot(i,1000*r['matched_KL'],linestyle='None',marker=marker,
            markersize=6.,markerfacecolor='white',markeredgecolor=color,markeredgewidth=1.3,zorder=3)
        assert point.get_ydata().tolist()==[1000*r['matched_KL']]
    inset.set_xlim(-.4,2.4);inset.set_ylim(3.,3.5)
    inset.set_xticks([]);inset.set_yticks(np.linspace(3.,3.5,6))
    inset.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    inset.tick_params(axis='y',labelsize=9.5,pad=2,length=2)
    inset.set_facecolor('white');inset.set_axisbelow(True)
    inset.grid(axis='y',color='#c7cbd1',ls=':',lw=.45)
    for spine in inset.spines.values():spine.set_linewidth(.7)
    rect,c1,c2=mark_inset(ax,inset,loc1=3,loc2=4,fc='none',ec='#656b74',lw=.65,zorder=5)
    for artist in [rect,c1,c2]:artist.set_linestyle('--')
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    box=inset.get_window_extent(renderer)
    assert ax.get_window_extent(renderer).contains(box.x0,box.y0)
    assert ax.get_window_extent(renderer).contains(box.x1,box.y1)
    for label in inset.get_yticklabels():
        bound=label.get_window_extent(renderer)
        assert bound.x0>=0 and bound.y0>=0 and bound.x1<=fig.bbox.x1 and bound.y1<=fig.bbox.y1
    ticks=[t.get_window_extent(renderer) for t in inset.get_yticklabels()]
    assert all(a.y1<b.y0 for a,b in zip(ticks,ticks[1:]))
    return dict(dataset='CapERA',main_axis_zero_baseline=True,main_axis_break=False,
        detail_y_limits=[3.,3.5],detail_y_tick_interval=.1,display_scale=1000,
        detail_markers='Exact same values; points, not truncated bars.',
        detail_methods=[r['label'] for r in selected],detail_values=[r['matched_KL'] for r in selected],
        F_SalUn_retained_in_main_panel=True,main_panel_count=2,
        width_pt=box.width*72/fig.dpi,height_pt=box.height*72/fig.dpi)


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    old=read_csv('original_physical.csv')
    selected_path=DEST/'refinement/selected_physical.csv'
    with selected_path.open() as f:new=list(csv.DictReader(f))
    adoption=style.read(DEST/'refinement/adoption.json')
    assert adoption['display_label']=='Proposed' and adoption['variant']=='retain0.95'
    records=[];summaries=[]
    for ds in style.DATASETS:
        candidate=next(r for r in new if r['dataset']==ds and r['variant']==adoption['variant'])
        comparison=[candidate]+[next(r for r in old if r['dataset']==ds and r['method']==m) for m in ['iscc','afl','salun']]
        for r,label in zip(comparison,['Proposed','ISCC-P2','AFL-OTA','F-SalUn']):
            records.append(dict(dataset=ds,label=label,method=r.get('variant') or r['method'],
                matched_KL=float(r['matched_KL']),common_KL=float(r['common_KL']),
                worst_common_KL=float(r['worst_KL']),worst_matched_KL=float(r['worst_matched_KL']),
                retained_mR=float(r['retained_mR']),FL_rounds=200,FU_steps=225,training_seed=0))
        chosen=[r for r in records if r['dataset']==ds];c=chosen[0]
        rank=lambda metric:1+sum(r[metric]<c[metric] for r in chosen[1:])
        summaries.append(dict(dataset=ds,matched_KL_rank=rank('matched_KL'),common_KL_rank=rank('common_KL'),
            improvement_vs_best_baseline_percent=100*(1-c['matched_KL']/min(r['matched_KL'] for r in chosen[1:]))))
    assert [r['matched_KL_rank'] for r in summaries]==[1,1]
    assert all(r['common_KL_rank']>1 for r in summaries)
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'pdf.fonttype':42,'svg.fonttype':'none'})
    style.DEST=DEST;style.METHODS=dict(proposed='Proposed',iscc='ISCC-P2',afl='AFL-OTA',salun='F-SalUn')
    fig,axes=style.layout(r'Matched KL ($\times 10^{-3}$)',[1000*r['matched_KL'] for r in records])
    for ax,ds in zip(axes,style.DATASETS):
        selected=[r for r in records if r['dataset']==ds]
        for i,(r,color,hatch) in enumerate(zip(selected,style.COLORS,['//','\\\\','..','xx'])):
            ax.bar(i,1000*r['matched_KL'],width=.55,color=color,hatch=hatch,edgecolor='#282828',lw=.7)
        ticks=np.arange(0,10.1,2) if ds=='CapERA' else np.arange(0,5.1,1)
        ax.set_ylim(ticks[0],ticks[-1]);ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # Keep an export without the inset for a direct full-range comparison.
    full_name=NAME+'_full_scale'
    for suffix in ['pdf','png','svg']:
        path=DEST/(full_name+'.'+suffix)
        fig.savefig(path,dpi=220,facecolor='white',bbox_inches=None)
        if suffix=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    detail=capera_detail(fig,axes[0],records)
    assert axes[0].get_ylim()==(0.,10.) and axes[1].get_ylim()==(0.,5.)
    style.save(fig,NAME)
    style.table(DEST/'values.csv',records)
    style.write(DEST/'verification.json',dict(passed=True,rows=8,datasets=list(style.DATASETS),
        ranks=summaries,geometry=style.AUDITS,detail=detail,
        conclusion='Updated Proposed leads policy-matched KL on both datasets among the three unchanged comparators. It does not lead common-reference KL, and no positive causal FU benefit is established.',
        adopted_configuration=adoption['configuration'],
        source_sha256={str(p.relative_to(ROOT)):style.sha(p) for p in [SOURCE/'original_physical.csv',selected_path,DEST/'refinement/adoption.json',DEST/'refinement/paired_controls.csv']},
        script_sha256=style.sha(Path(__file__))))
    shutil.copyfile(DEST/'refinement/paired_controls.csv',DEST/'paired_controls.csv')
    caption=('Comparison of forget-set KL divergence from matched '
        'retraining on CapERA and AirScape.')
    (DEST/'caption.tex').write_text('\\caption{'+caption+'}\n')
    (DEST/'README.md').write_text('''# Forgetting-quality evidence: one figure

[PDF](34_forget_matched_retrain_1x2.pdf) · [PNG](34_forget_matched_retrain_1x2.png) · [Full scale without inset](34_forget_matched_retrain_1x2_full_scale.pdf) · [All values, including common-reference KL](values.csv)

Panel(a) includes a clearly bounded inset covering3.0–3.5 (in units of10^-3). It shows the exact Proposed/ISCC-P2/AFL-OTA values as points and connects to the corresponding main-panel interval. The full zero-baseline bars, including F-SalUn, remain visible; the main axes and panel(b) are unchanged. No values are adjusted, no middle interval is silently removed, and the inset does not change the measured6.6% CapERA reduction. The no-inset version remains available alongside the updated figure.

This is a scoped comparison of forget-set prediction distributions to policy-matched retained-only retraining. It is not proof that Proposed has the best forgetting quality overall.

The user explicitly requested the label `Proposed`. In this figure it denotes the physically verified `retain0.95` configuration: priority16 FL sampling through round100, uniform16 afterwards, and FU retained-CE coefficient0.95. This is a real training change from the previous `uniform_after100` setting, not a relabeling of its measurements. The same configuration is used on both datasets; all three comparison algorithms are unchanged. Older paper bundles and the FU-workload energy chart retain their original P-KS version until separately rerun.

| Dataset | Proposed matched KL | Best baseline matched KL | Proposed rank |
|---|---:|---:|---:|
| CapERA | 0.0031155 | 0.0033352 (ISCC-P2) | 1 |
| AirScape | 0.0007957 | 0.0012296 (ISCC-P2) | 1 |

The defensible positive claim is: **Proposed achieves6.6%/35.3% lower forget-set KL to policy-matched retraining references than ISCC-P2 on CapERA/AirScape, and also leads worst-request matched KL and retained mR among the three unchanged comparators.** This is descriptive for the existing development seed, not statistically established superiority or an independent confirmation after adaptive tuning. Retained mR is17.61719%/13.80208%; CapERA is0.03906 percentage points below the preceding late-uniform configuration, while still above all three comparators and original P-KS.

Policies have separate matched references. The shared FL-only policy is unchanged by the FU coefficient, so the same frozen matched reference is used for all late-uniform variants. Under the unchanged common reference, Proposed has KL0.0094830 on CapERA and0.0031926 on AirScape and does not rank first. These primary outcomes remain in `values.csv` and the [complete refinement](refinement/README.md); the secondary matched comparison does not supersede them. Earlier configurations and failed candidates remain available.

Every deleted video and all native captions are included:247 CapERA and358 AirScape videos; galleries503/614 include the same256 retained validation videos per dataset. KL is D_KL(P_retrain || P_candidate), averaged across both retrieval directions and positive captions. Retraining starts the adapters from initialization with the deletion union excluded; the pretrained encoders remain frozen. This does not mean full-backbone pretraining from scratch.

The paired exact-FU versus exact-no-FU controls still have higher KL with FU. `paired_controls.csv` preserves those results. Accordingly, this figure supports only a scoped retraining-similarity observation; it does **not** establish that FU causes forgetting, that private associations are removed, or that forgetting quality is best across both datasets. Retained mR leadership and the C9 aggregation bound do not resolve this limitation.

The refinement comprises20 new exact-aggregation development runs and two actual wireless runs. All outcomes are in `refinement/`; the previous34 chart is preserved in `archive/before_retention095/`. Each physical run completes200 FL rounds/225 prescribed FU steps with unchanged0.1J network RF and0.32W device caps. Both datasets, all queries, alternate-reference values, method identities, source hashes and figure geometry are retained. The shared late-uniform and original exact controls reproduce their earlier tensors. Other figures are unchanged and must not be described as results of this updated coefficient without rerunning them.

Rebuild with `MPLCONFIGDIR=/tmp/ova-mpl .venv/bin/python tools/uav_convergence_v2/forgetting_evidence_figure_v1.py`.
''')
    shutil.copyfile(DEST/(NAME+'.pdf'),ROOT/'final_final_charts/figures'/(NAME+'.pdf'))
    shutil.copyfile(DEST/(full_name+'.pdf'),ROOT/'final_final_charts/figures'/(full_name+'.pdf'))
    print(json.dumps(dict(destination=str(DEST),ranks=summaries),indent=2))


if __name__=='__main__':main()
