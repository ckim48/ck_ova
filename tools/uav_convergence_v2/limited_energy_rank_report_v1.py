"""Publish the full energy-budget sensitivity without modifying the 650 kJ chart."""
import csv
import math
from pathlib import Path
import shutil

import numpy as np
from matplotlib.cbook import boxplot_stats
import limited_energy_retrieval_v1 as inference
import paper_boxplot_refine_v2 as renderer

ROOT,read,write,sha = inference.ROOT,inference.read,inference.write,inference.sha
BASE=ROOT/'final_final_charts/downlink_bandwidth'
DEST=BASE/'limited_energy'
NAME='19_all_method_rank_boxplot_1x2'


def csv_write(path,rows):
    with path.open('w',newline='') as f:
        out=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        out.writeheader();out.writerows(rows)


def main():
    original_files={str(p.relative_to(ROOT)):sha(p) for p in
                    [BASE/(NAME+'.'+ext) for ext in ['pdf','png','svg']]}
    verified=read(inference.OUT/'finished.json');assert verified['passed']
    protocol=inference.checkpoint.register()
    summary=read(inference.OUT/'summary.json')
    renderer.report.setup();renderer.report.prior.chart.plt.rcParams['savefig.bbox']=None
    figures,all_boxes,pairs=[],[],[]
    for budget in protocol['budgets_kJ']:
        folder=DEST/f'{budget}kJ';folder.mkdir(parents=True,exist_ok=True)
        panels,boxes=[],[]
        for ds in inference.DATASETS:
            rows=read(inference.OUT/f'{budget}kJ'/f'{ds.lower()}_query_ranks.json')
            shutil.copyfile(inference.OUT/f'{budget}kJ'/f'{ds.lower()}_query_ranks.json',folder/f'{ds.lower()}_query_ranks.json')
            csv_write(folder/f'{ds.lower()}_query_ranks.csv',rows)
            arrays=np.load(inference.OUT/f'{budget}kJ'/f'{ds.lower()}_scores.npz')
            for index,(method,label) in enumerate(inference.METHODS.items()):
                values=np.array([r[method+'_rank'] for r in rows],dtype=float)
                # Independent rank reconstruction, including the original tie order.
                import torch
                scores=torch.from_numpy(arrays[method])
                order=torch.argsort(scores.T,dim=1,descending=True)
                ranks=torch.argsort(order,dim=1)+1
                count=scores.shape[1]//scores.shape[0]
                actual=[int(ranks[r['gallery_caption_index'],r['gallery_caption_index']//count]) for r in rows]
                assert actual==values.tolist()
                s=boxplot_stats(values,whis=1.5)[0]
                boxes.append(dict(dataset=ds,method=method,label=label,queries=len(values),energy_budget_kJ=budget,
                    minimum=float(min(values)),maximum=float(max(values)),q1=float(s['q1']),median=float(s['med']),q3=float(s['q3']),
                    whisker_low=float(s['whislo']),whisker_high=float(s['whishi']),fliers=s['fliers'].tolist(),
                    color=renderer.report.PALETTE[index][0]))
                metric=next(r for r in summary if (r['dataset'],r['method'],r['budget_kJ'])==(ds,method,budget))
                assert metric['median_GT_rank']==s['med']
                assert metric['energy_kJ']<=budget<metric['next_completion_energy_kJ']
                assert max(values)<700
                if method!='gate_select16':
                    delta=values-np.array([r['gate_select16_rank'] for r in rows])
                    pairs.append(dict(dataset=ds,budget_kJ=budget,comparator=method,queries=len(rows),
                        improved=int(sum(delta>0)),tied=int(sum(delta==0)),worsened=int(sum(delta<0))))
            panels.append(dict(dataset=ds,
                x=dict(kind='categorical',limits=[-.4,3.4],ticks=list(range(4)),labels=list(inference.METHODS.values())),
                y=dict(limits=[0,700],ticks=list(range(0,701,100)),labels=[str(v) for v in range(0,701,100)])))
        figure=dict(name=NAME,kind='rank_boxplot',axes=panels,boxes=boxes,energy_budget_kJ=budget,
            all_retained_queries=True,show_all_fliers=True,y_scale='linear')
        result=renderer.render(folder,figure)
        caption=(f'Ground-truth video-rank distributions of all retained text queries at a common {budget} kJ '
            'active-energy limit. Each method uses its last completed model within the limit; actual completed '
            'FL rounds and FU procedures are reported in progress.csv. Boxes show the median and interquartile '
            'range, whiskers use 1.5 IQR, and every outlier is shown. Lower ranks are better. The same retained '
            'query sets and full galleries are used throughout. Requests follow the original round-triggered '
            'schedule, so released and completed FU counts may differ. This is a retained-utility comparison '
            'under limited energy, not a comparison of equally completed or equally effective unlearning. '
            'Single development seed; no confidence intervals over training runs.\n')
        result['caption']=caption;result['folder']=str(folder.relative_to(ROOT))
        write(folder/'manifest.json',result)
        (folder/(NAME+'_caption.txt')).write_text(caption)
        local=[r for r in summary if r['budget_kJ']==budget]
        write(folder/'progress.json',local);csv_write(folder/'progress.csv',local)
        all_boxes+=boxes;figures.append(result)
    write(DEST/'progress.json',summary);csv_write(DEST/'progress.csv',summary)
    replay_checks=[read(p) for p in sorted((inference.checkpoint.OUT/'checks').glob('*.json'))]
    assert len(replay_checks)==12 and all(r['passed'] for r in replay_checks)
    write(DEST/'checkpoint_replay_protocol.json',protocol)
    write(DEST/'checkpoint_replay_checks.json',dict(passed=True,stages=replay_checks))
    write(DEST/'box_statistics.json',all_boxes);write(DEST/'paired_outcomes.json',pairs)
    csv_write(DEST/'paired_outcomes.csv',pairs)
    text='# Retrieval rank under limited energy\n\n'
    text+='The current 650 kJ figure remains unchanged. The middle 300 kJ case is the main illustrative variant; '
    text+='200 and 400 kJ are published alongside it. Existing aggregate curves were inspected before choosing this '
    text+='exploratory grid; it is not an independent confirmation experiment. All four methods and all retained queries are included.\n\n'
    text+='Exact replays reproduce original final parameters and recorded learning trajectories. Snapshots are the last '
    text+='completed transactions within each budget, including transactions without a previous evaluation. Their actual '
    text+='full-gallery scores were recomputed; no mR-to-rank conversion or interpolation is used.\n\n'
    text+='Energy includes active communication, adapter computation and hovering, with the same common warmup and '
    text+='travel exclusions as the prior study. After 200 FL rounds and all 45 FU procedures, the unchanged retained-only '
    text+='continuation restarts the common 0.0035 learning rate. The round-triggered request schedule means equal energy '
    text+='does not imply equal request exposure. FU procedure completion does not certify forgetting, and KL is not matched here.\n\n'
    for budget in protocol['budgets_kJ']:
        text+=f'## {budget} kJ\n\n[PDF]({budget}kJ/{NAME}.pdf) · [Caption]({budget}kJ/{NAME}_caption.txt)\n\n'
        text+='| Dataset | Method | Energy (kJ) | FL rounds | FU arrived | FU completed | Retained mR (%) | Median rank |\n'
        text+='|---|---|---:|---:|---:|---:|---:|---:|\n'
        for r in summary:
            if r['budget_kJ']==budget:
                text+=f"| {r['dataset']} | {r['label']} | {r['energy_kJ']:.2f} | {r['FL_round']} | {r['FU_arrived']} | {r['FU_completed']} / 45 | {r['retained_mR_percent']:.3f} | {r['median_GT_rank']:g} |\n"
        text+='\n'
    text+='Reproduce checkpoints, inference and figures with the three `limited_energy_*_v1.py` scripts in `tools/uav_convergence_v2`. Raw checkpoints, full score matrices and features remain local.\n'
    (DEST/'README.md').write_text(text)
    html='<!doctype html><meta charset="utf-8"><title>Energy-limited retrieval ranks</title><style>body{font:16px system-ui;max-width:1150px;margin:35px auto;padding:0 20px}img{width:100%}</style><h1>Energy-limited retrieval ranks</h1><p><a href="README.md">Conditions and completed workloads</a></p>'
    for budget in [300,200,400]:
        html+=f'<section><h2>{budget} kJ</h2><a href="{budget}kJ/{NAME}.pdf"><img src="{budget}kJ/{NAME}.png"></a></section>'
    (DEST/'index.html').write_text(html)
    index=BASE/'index.html';page=index.read_text();marker='<!-- limited-energy-ranks -->';end='<!-- /limited-energy-ranks -->'
    if marker in page:
        before,tail=page.split(marker,1);page=before+tail.split(end,1)[1]
    section=marker+f'<section><h2>Retrieval ranks at 300 kJ</h2><a href="limited_energy/300kJ/{NAME}.pdf"><img loading="lazy" src="limited_energy/300kJ/{NAME}.png"></a><p><a href="limited_energy/index.html">200, 300 and 400 kJ comparisons with completed workloads</a></p></section>'+end
    index.write_text(page.replace('<!-- /tgcn-additions -->','<!-- /tgcn-additions -->'+section,1))
    assert all(sha(ROOT/p)==digest for p,digest in original_files.items())
    write(DEST/'verification.json',dict(passed=True,models=24,figures=3,query_counts=[1280,256],
        all_checkpoint_budget_boundaries_verified=True,all_scores_recomputed=True,all_query_ranks_verified=True,
        all_outliers_retained=True,original_650kJ_figure_unchanged=True,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in
            [Path(__file__),Path(inference.__file__),Path(inference.checkpoint.__file__),Path(renderer.__file__)]},
        original_650kJ_sha256=original_files,inference_checks=verified))
    print(dict(passed=True,figures=3,models=24),flush=True)


if __name__=='__main__':main()
