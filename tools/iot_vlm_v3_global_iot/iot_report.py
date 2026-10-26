"""Paper figures from completed arms only; all modeling details live in JSON."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from iot_energy import method_energy, PARAMETERS


def generate(root):
    root = Path(root)
    protocol = json.loads((root/'protocol.json').read_text())
    workspace = root.parents[1]
    geometry = json.loads((workspace/protocol['geometry_directory']/'mission_spec.json').read_text())
    methods = list(protocol['plotting']['labels'])
    rows, missing = [], []
    for dataset in protocol['datasets']:
        folder = root/(dataset.lower()+'_seed0')
        for method in methods:
            manifest = folder/method/'manifest.json'
            if not manifest.exists() or json.loads(manifest.read_text())['status'] != 'completed_200FL_and_requests':
                missing.append(dict(dataset=dataset, method=method))
                continue
            rows.append(dict(dataset=dataset, **method_energy(folder, method, geometry)))
    target = root/'figures'
    target.mkdir(exist_ok=True)
    (target/'metrics.json').write_text(json.dumps(dict(rows=rows, missing=missing, energy_model=PARAMETERS), indent=2, allow_nan=False))
    if rows:
        with (target/'metrics.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    if missing:
        return dict(status='incomplete', missing=missing)
    plt.rcParams.update({'font.family':'serif', 'font.serif':['Times New Roman','DejaVu Serif'],
                         'font.size':9, 'axes.labelsize':9, 'axes.titlesize':10,
                         'pdf.fonttype':42, 'ps.fonttype':42, 'axes.linewidth':.65,
                         'xtick.labelsize':8, 'ytick.labelsize':8})
    colors = ['#167D8D','#4978A8','#D69B48','#8A769F']
    for key, label, scale, name in [('loss','Retained loss',1.,'loss'),
                                   ('total_energy_j','Total energy (kJ)',1000.,'total_energy'),
                                   ('total_delay_s','Total delay (s)',1.,'total_delay')]:
        fig, axes = plt.subplots(1,2,figsize=(7.16,2.7),layout='constrained')
        for ax, dataset in zip(axes,protocol['datasets']):
            subset = {r['method']:r for r in rows if r['dataset']==dataset}
            values = np.array([subset[m][key]/scale for m in methods])
            bars = ax.bar(np.arange(4),values,width=.6,color=colors,edgecolor='white',linewidth=.5)
            ax.set_xticks(range(4),[protocol['plotting']['labels'][m] for m in methods])
            ax.set_ylabel(label); ax.set_title(dataset,pad=7)
            ax.set_ylim(0,float(values.max())*1.17)
            ax.spines[['top','right']].set_visible(False)
            ax.set_axisbelow(True); ax.yaxis.grid(True,color='#E6E9ED',linewidth=.55)
            ax.tick_params(direction='out',length=3,width=.6)
            ax.bar_label(bars,labels=[f'{v:.5f}' if key=='loss' else f'{v:.3f}' for v in values],padding=3,fontsize=7,color='black')
        for suffix in ['png','pdf','svg']:
            fig.savefig(target/f'{name}_1x2.{suffix}',dpi=450,bbox_inches='tight')
        plt.close(fig)
    return dict(status='completed', figures=str(target), rows=len(rows))


if __name__ == '__main__':
    import sys
    print(generate(sys.argv[1]))
