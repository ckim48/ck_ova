"""Select measured 3/9-step bars; retain the complete 1/3/5/7/9 workload sweep."""
import copy
import csv
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import paper_typography_v1 as typography
import tgcn_additions_report as report

ROOT,DEST,read,write,sha=report.ROOT,report.DEST,report.read,report.write,report.sha
NAME='11_fu_steps_energy_1x2'
LEVELS=[3,9]


def main():
    metrics_path=DEST/'data/fu_workload_metrics.json';series_path=DEST/'data/tgcn_plot_series.json'
    manifest_path=DEST/'data/tgcn_manifest.json'
    rows=read(metrics_path);series=read(series_path);manifest=read(manifest_path)
    measurement_hashes={str(p.relative_to(ROOT)):sha(p) for p in [metrics_path,metrics_path.with_suffix('.csv')]}
    assert len(rows)==40 and sorted({r['fu_steps_per_request'] for r in rows})==[1,3,5,7,9]
    index=next(i for i,f in enumerate(manifest['figures']) if f['name']==NAME)
    figure=copy.deepcopy(manifest['figures'][index])
    other={str((DEST/f['file']).relative_to(ROOT)):sha(DEST/f['file']) for entry in manifest['figures'] if entry['name']!=NAME for f in entry['files']}
    for ext in ['pdf','png','svg']:
        p=ROOT/'final_final_charts/figures'/('all_method_rank_boxplot_1x2.'+ext);other[str(p.relative_to(ROOT))]=sha(p)
    archive=DEST/'archive/fu_steps_1_9_20261030'
    if not (archive/'original.json').exists():
        assert figure['steps_per_request']==[1,9]
        archive.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(DEST/(NAME+'.pdf'),archive/(NAME+'.pdf'))
        write(archive/'original.json',dict(figure=figure,series=[s for s in series if s['figure']==NAME],
            archived_pdf_sha256=sha(archive/(NAME+'.pdf'))))
        (archive/'README.md').write_text('Previous paired 1-step/9-step figure. Its measurements remain in ../../data/fu_workload_metrics.csv. The current figure uses actual 3-step/9-step results; no data were relabeled or interpolated.\n')
    old_archive=read(archive/'original.json');assert old_archive['figure']['steps_per_request']==[1,9]
    assert sha(archive/(NAME+'.pdf'))==old_archive['archived_pdf_sha256']
    exported=[]
    for s in series:
        if s['figure']!=NAME:continue
        actual=[next(r for r in rows if (r['dataset'],r['method'],r['fu_steps_per_request'])==(s['dataset'],s['method'],k)) for k in LEVELS]
        for row in actual:
            assert row['origin']=='fresh_execution' and row['FL_rounds']==200 and row['FU_cycles']==45*row['fu_steps_per_request']
            source=ROOT/row['source'];check=read(source/'verification.json')
            assert check['passed'] and check['FU_steps']==row['FU_cycles']
            requests=read(source/'request_records.json');assert len(requests)==45
            assert all(r['steps']==r['required_steps']==row['fu_steps_per_request'] and r['completed_s'] is not None for r in requests)
        s['x']=LEVELS[:];s['y']=[r['total_energy_j']*.001 for r in actual]
        exported.extend(actual)
    assert len(exported)==16
    figure.update(steps_per_request=LEVELS,title='Total energy: paired 3-step and 9-step FU workloads',
        display_revision='measured_3_and_9_steps_v2')
    report.setup();report.prior.chart.plt.rcParams.update({'savefig.bbox':None,'hatch.linewidth':.55})
    result=typography.render(DEST,figure,series)
    for ext in ['pdf','png','svg']:
        p=DEST/(NAME+'.'+ext)
        if ext=='svg':p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
        shutil.copyfile(p,ROOT/'final_final_charts/figures'/('fu_steps_energy_1x2.'+ext))
    result['files']=[dict(file=NAME+'.'+ext,sha256=sha(DEST/(NAME+'.'+ext))) for ext in ['png','pdf','svg']]
    manifest['figures'][index]=result
    sources=[Path(__file__),Path(typography.__file__),Path(report.__file__),Path(__file__).with_name('match_rank_fu_layout_v1.py'),Path(__file__).with_name('tgcn_additions_check.py')]
    manifest['source_sha256'].update({str(p.relative_to(ROOT)):sha(p) for p in sources})
    write(series_path,series);write(manifest_path,manifest)
    for p in [DEST/'TGCN_ADDITIONS.md',DEST/'index.html',ROOT/'final_final_charts/figures/README.md']:
        text=p.read_text().replace('paired 1-step and 9-step','paired 3-step and 9-step')
        text=text.replace('1 versus 9 outer FU steps per request, the endpoints of the completed sweep',
            '3 versus 9 outer FU steps per request, two measured workloads from the completed sweep')
        p.write_text(text)
    caption=('Effect of the number of FU steps per request (3 and 9) on total energy consumption on CapERA and AirScape.')
    (ROOT/'final_final_charts/figures/fu_steps_energy_1x2_caption.txt').write_text(caption+'\n')
    gaps=[]
    for ds in report.DATASETS:
        for k in [1,3,5,7,9]:
            p=next(r for r in rows if (r['dataset'],r['method'],r['fu_steps_per_request'])==(ds,'gate_select16',k))
            a=next(r for r in rows if (r['dataset'],r['method'],r['fu_steps_per_request'])==(ds,'afl',k))
            gaps.append(dict(dataset=ds,steps=k,Proposed_kJ=p['total_energy_j']*.001,AFL_OTA_kJ=a['total_energy_j']*.001,
                reduction_percent=100*(1-p['total_energy_j']/a['total_energy_j'])))
    with (DEST/'data/fu_workload_proposed_afl_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(gaps[0]),lineterminator='\n');w.writeheader();w.writerows(gaps)
    svg=ET.parse(DEST/(NAME+'.svg')).getroot()
    text=' '.join(svg.itertext());assert '3 steps / request' in text and '9 steps / request' in text and '1 step / request' not in text
    panels=svg.findall('.//s:clipPath/s:rect',typography.NS)
    assert len(panels)==2
    for panel in panels:assert np.allclose([float(panel.get('width')),float(panel.get('height'))],[205.128,174.798])
    assert all(sha(ROOT/p)==h for p,h in measurement_hashes.items())
    assert all(sha(ROOT/p)==h for p,h in other.items())
    assert result['axes']==figure['axes']
    write(DEST/'data/fu_bar_workload_selection_v2.json',dict(passed=True,displayed_steps=LEVELS,
        all_measured_steps=[1,3,5,7,9],new_experiments=False,measured_bars=16,
        method_identity='Original P-KS/gate_select16; unchanged comparators. Not the new Controlled candidate.',
        original_1_step_archive=str(archive.relative_to(ROOT)),
        unchanged_measurements=measurement_hashes,unchanged_other_figures=other,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        typography=result['typography'],comparison=gaps))
    notes=DEST/'TGCN_ADDITIONS.md';text=notes.read_text();marker='<!-- fu-workload-3-9 -->'
    text=text.split(marker)[0].rstrip()
    notes.write_text(text+'\n\n'+marker+'\n\nFigure11 now displays the measured 3-step and9-step workloads. At3 steps, Proposed uses12.06%/12.16% less energy than AFL-OTA on CapERA/AirScape; at1 step the reductions are3.50%/3.63%. The full1/3/5/7/9 sweep remains in the metrics files, and the previous1/9 figure is archived at `archive/fu_steps_1_9_20261030/`. These are workload-cost comparisons, not equal-forgetting-quality comparisons; prescribed FU steps alone do not certify deletion. No new training, interpolated2-step results, changed method, or changed numerical measurement is used.\n')
    print(dict(passed=True,displayed_steps=LEVELS,verified_measured_bars=16,aliased_pdf='final_final_charts/figures/fu_steps_energy_1x2.pdf'))


if __name__=='__main__':main()
