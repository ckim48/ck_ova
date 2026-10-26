"""Completed-run modeled energy, with final forgetting evaluation pending."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True)
root=p.parse_args().project.resolve();out=root/'results/overnight_energy_v2/precision_completion_v1'
source=out/'summary.json';data=json.loads(source.read_text())
assert data['status']=='q6_training_complete_final_reference_quality_pending'
rows={x['method']:x for x in data['completed_methods']}
plt.rcParams.update({'font.size':9,'axes.labelsize':9,'axes.titlesize':10,
                     'svg.fonttype':'none','figure.dpi':160})
fig,axes=plt.subplots(1,2,figsize=(9.2,4.0));fig.subplots_adjust(left=.075,right=.985,bottom=.25,top=.85,wspace=.31)
names=['coupled_mixed_aircomp','coupled_hybrid_q6','coupled_hybrid','coupled_fixed_head']
labels=['Mixed','Hybrid 6b','Hybrid 8b','Fixed 8b'];x=np.arange(4)
learning=np.array([rows[n]['learning_only_energy_bound_j'] for n in names]);request=np.array([rows[n]['request_round_energy_bound_j'] for n in names])
axes[0].bar(x,learning,color='#b6c5d4',label='95 learning-only rounds')
axes[0].bar(x,request,bottom=learning,color='#354f70',label='5 request rounds')
for i,n in enumerate(names):axes[0].text(i,rows[n]['energy_bound_j']+.006,f"{rows[n]['energy_bound_j']:.3f}",ha='center',fontsize=9)
axes[0].set(xticks=x,xticklabels=labels,ylabel='Cumulative communication bound (J)',ylim=(0,.40),title='(a) All 100 rounds completed')
axes[0].legend(loc='upper left',fontsize=8,frameon=False)
rounds=[51,61,71,81,91];x=np.arange(5);width=.25
for offset,name,label,color in [(-1,'coupled_hybrid','Hybrid 8b','#687aa0'),(0,'coupled_hybrid_q6','Hybrid 6b','#ba6655'),(1,'coupled_mixed_aircomp','Mixed','#3b7b72')]:
    rr={r['round']:r for r in data['request_rounds'] if r['method']==name}
    axes[1].bar(x+offset*width,[1000*rr[k]['energy_bound_j'] for k in rounds],width,label=label,color=color)
axes[1].set(xticks=x,xticklabels=rounds,ylabel='Per-request communication bound (mJ)',xlabel='Request round',title='(b) Energy in each request round')
axes[1].legend(loc='upper left',fontsize=8,frameon=False)
for ax in axes:
    ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
fig.suptitle('6-bit Hybrid: 10.94% less modeled energy than 8-bit Hybrid',fontsize=12,y=.98)
fig.text(.075,.10,'One seed, identical canonical schedule. Mixed remains lower in energy.\nFinal retained-retraining KL / disagreement / MIA evaluation is pending.',fontsize=8.5)
fig.text(.075,.025,'Modeled upper bounds include fixed overhead; flight and local computation are excluded. Failed/incomplete and ideal links are omitted.',fontsize=7.6,color='#444444')
for suffix in ['png','svg']:fig.savefig(out/f'energy_summary.{suffix}')
plt.close(fig)
(out/'figure_provenance.json').write_text(json.dumps({'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
print('SAVED',out/'energy_summary.png')
