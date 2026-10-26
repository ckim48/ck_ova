"""Scientific summary of separate experimental scopes; no new experiment."""
import base64,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path('/home/wnlab/Documents/Codex/2026-10-20/files-pasted-by-the-user-standalone/outputs/ova_CK');f=r/'results/overnight_energy_v2'
original={x['method']:x for x in json.loads((r/'results/energy_quality_v1/energy_quality_report.json').read_text())['methods']}
burst={x['method'].split('/')[-1]:x for x in json.loads((f/'burst_comparison_v31/results.json').read_text())['methods']}
pilot={x['mode']:x for x in json.loads((f/'inverse_quality_v30/summary.json').read_text())['selected']}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,3,figsize=(11.8,4.2),gridspec_kw={'wspace':.4})
panels=[('Original protocol\n100 completed rounds',['Mixed','Hybrid\n6-bit','Hybrid\n8-bit'],[original[k]['energy_bound_j'] for k in ['coupled_mixed_aircomp','coupled_hybrid_q6','coupled_hybrid']],['#287c8e','#cb7329','#cb7329'],'Energy upper bound (J)'),
        ('Changed request tolerance\n100 completed rounds',['Mixed','Hybrid'],[burst[k]['energy_bound_j'] for k in ['mixed_aircomp','hybrid']],['#287c8e','#cb7329'],'Energy upper bound (J)'),
        ('New communication design\nOne fixed FU61 update',['Mixed','Time\nseparation','Hybrid'],[1000*pilot[k]['energy_bound_j'] for k in ['mixed_aircomp','phase_grid','hybrid']],['#287c8e','#7962a7','#cb7329'],'Energy upper bound (mJ)')]
for ax,(title,names,values,colors,ylabel) in zip(axes,panels):
    bars=ax.bar(names,values,color=colors,width=.62);ax.set_title(title,fontsize=11,pad=12);ax.set_ylabel(ylabel);ax.set_ylim(0,max(values)*1.25);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,value+max(values)*.035,f'{value:.3f}' if max(values)>1 else f'{value:.5f}',ha='center',va='bottom',fontsize=10)
fig.subplots_adjust(left=.07,right=.985,top=.78,bottom=.28)
fig.suptitle('Lowest modeled communication energy remains Mixed AirComp',fontsize=14,y=.97)
fig.text(.07,.115,'Panels use different protocols or scopes and must be compared separately. Full time-separated training is still running.',fontsize=9)
fig.text(.07,.06,'RF/control accounting included; extra solver/codec compute and flight energy unmeasured. No quality-equivalence or deletion guarantee.',fontsize=8.6)
path=f/'final_energy_summary.png';fig.savefig(path,dpi=170);fig.savefig(f/'final_energy_summary.svg');plt.close(fig)
print(base64.b64encode(path.read_bytes()).decode())
