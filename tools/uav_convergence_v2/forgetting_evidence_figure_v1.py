"""One scoped forgetting-quality figure; preserve non-leading outcomes/references."""
import csv
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import forgetting_quality_report_v1 as style

ROOT=style.ROOT
SOURCE=ROOT/'final_final_charts/joint_quality_control/data'
DEST=ROOT/'final_final_charts/forgetting_evidence'
NAME='34_forget_matched_retrain_1x2'


def read_csv(name):
    with (SOURCE/name).open() as f:return list(csv.DictReader(f))


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    old=read_csv('original_physical.csv');new=read_csv('all_physical_candidates.csv')
    records=[];summaries=[]
    for ds in style.DATASETS:
        candidate=next(r for r in new if r['dataset']==ds and r['variant']=='uniform_after100')
        comparison=[candidate]+[next(r for r in old if r['dataset']==ds and r['method']==m) for m in ['iscc','afl','salun']]
        for r,label in zip(comparison,['Controlled','ISCC-P2','AFL-OTA','F-SalUn']):
            records.append(dict(dataset=ds,label=label,method=r.get('variant') or r['method'],
                matched_KL=float(r['matched_KL']),common_KL=float(r['common_KL']),
                worst_common_KL=float(r['worst_KL']),worst_matched_KL=float(r['worst_matched_KL']),
                retained_mR=float(r['retained_mR']),FL_rounds=200,FU_steps=225,training_seed=0))
        chosen=[r for r in records if r['dataset']==ds];c=chosen[0]
        rank=lambda metric:1+sum(r[metric]<c[metric] for r in chosen[1:])
        summaries.append(dict(dataset=ds,matched_KL_rank=rank('matched_KL'),common_KL_rank=rank('common_KL'),
            improvement_vs_best_baseline_percent=100*(1-c['matched_KL']/min(r['matched_KL'] for r in chosen[1:]))))
    assert [r['matched_KL_rank'] for r in summaries]==[3,1]
    assert all(r['common_KL_rank']>1 for r in summaries)
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'pdf.fonttype':42,'svg.fonttype':'none'})
    style.DEST=DEST;style.METHODS=dict(controlled='Controlled',iscc='ISCC-P2',afl='AFL-OTA',salun='F-SalUn')
    fig,axes=style.layout(r'Matched KL ($\times 10^{-3}$)',[1000*r['matched_KL'] for r in records])
    for ax,ds in zip(axes,style.DATASETS):
        selected=[r for r in records if r['dataset']==ds]
        for i,(r,color,hatch) in enumerate(zip(selected,style.COLORS,['//','\\\\','..','xx'])):
            ax.bar(i,1000*r['matched_KL'],width=.55,color=color,hatch=hatch,edgecolor='#282828',lw=.7)
        ticks=np.arange(0,10.1,2) if ds=='CapERA' else np.arange(0,5.1,1)
        ax.set_ylim(ticks[0],ticks[-1]);ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    style.save(fig,NAME)
    style.table(DEST/'values.csv',records)
    style.write(DEST/'verification.json',dict(passed=True,rows=8,datasets=list(style.DATASETS),
        ranks=summaries,geometry=style.AUDITS,
        conclusion='Controlled leads only AirScape policy-matched KL among the three unchanged comparison methods. Not a joint/common-reference forgetting lead; no established causal FU benefit.',
        source_sha256={str((SOURCE/f).relative_to(ROOT)):style.sha(SOURCE/f) for f in ['original_physical.csv','all_physical_candidates.csv','paired_controls.csv']},
        script_sha256=style.sha(Path(__file__))))
    shutil.copyfile(SOURCE/'paired_controls.csv',DEST/'paired_controls.csv')
    caption=(r'Forget-set KL divergence from policy-matched retained-only retraining references '
        r'after 200 FL rounds and 225 prescribed FU steps. Controlled uses the proposed framework '
        r'with uniform FL sampling after round 100; each reference follows the corresponding learning policy. '
        r'Lower values indicate closer prediction distributions. Results are from one development seed.')
    (DEST/'caption.tex').write_text('\\caption{'+caption+'}\n')
    (DEST/'README.md').write_text('''# Forgetting-quality evidence: one figure

[PDF](34_forget_matched_retrain_1x2.pdf) · [PNG](34_forget_matched_retrain_1x2.png) · [All values, including common-reference KL](values.csv)

This is a scoped comparison of forget-set prediction distributions to policy-matched retained-only retraining. It is not proof that Proposed has the best forgetting quality overall.

`Controlled` is the new `uniform_after100` candidate, not the original P-KS used in the existing paper figures. The blue label explicitly preserves that distinction. All three comparison algorithms are unchanged. This figure does not silently replace Proposed or relabel a candidate as the original method.

| Dataset | Controlled matched KL | Best baseline matched KL | Controlled rank |
|---|---:|---:|---:|
| CapERA | 0.0034600 | 0.0033352 (ISCC-P2) | 3 |
| AirScape | 0.0008931 | 0.0012296 (ISCC-P2) | 1 |

The defensible positive claim is: **On AirScape, the controlled schedule achieves 27.4% lower forget-set KL to its policy-matched retraining reference than ISCC-P2, the best comparator under this metric.** This is descriptive for the existing development seed, not statistically established superiority or an independent confirmation after adaptive tuning.

CapERA is also plotted, including its non-leading result. Policies have separate matched references. Under the unchanged common reference, Controlled has KL0.0100569 on CapERA and0.0034039 on AirScape and ranks fourth on both. These primary outcomes remain in `values.csv` and the [complete comparison](../joint_quality_control/README.md); the secondary matched comparison does not supersede them.

Every deleted video and all native captions are included:247 CapERA and358 AirScape videos; galleries503/614 include the same256 retained validation videos per dataset. KL is D_KL(P_retrain || P_candidate), averaged across both retrieval directions and positive captions. Retraining starts the adapters from initialization with the deletion union excluded; the pretrained encoders remain frozen. This does not mean full-backbone pretraining from scratch.

The paired exact-FU versus exact-no-FU controls still have higher KL with FU. `paired_controls.csv` preserves those results. Accordingly, this figure supports only a scoped retraining-similarity observation; it does **not** establish that FU causes forgetting, that private associations are removed, or that forgetting quality is best across both datasets. Retained mR leadership and the C9 aggregation bound do not resolve this limitation.

No new training or selection of favorable queries was performed for this chart. It presents the already evaluated shared candidate,200 FL rounds/225 prescribed FU steps, seed0, and the unchanged comparators. Both datasets, alternate-reference values, method identities, source hashes and figure geometry are retained. The existing main Proposed figures are unchanged.

Rebuild with `MPLCONFIGDIR=/tmp/ova-mpl .venv/bin/python tools/uav_convergence_v2/forgetting_evidence_figure_v1.py`.
''')
    print(json.dumps(dict(destination=str(DEST),ranks=summaries),indent=2))


if __name__=='__main__':main()
