"""All-arm report without selection or hidden failed outcomes."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/overnight_energy_v2/association_objective_v1'
plan=json.loads((OUT/'plan.json').read_text())
status=json.loads((OUT/'status.json').read_text())
assert status['status']=='completed',status
initial=json.loads((OUT/'initial_metrics.json').read_text())
summary=json.loads((OUT/'summary.json').read_text())
names={'retain_only':'Retain only','gradient_ascent':'Gradient ascent','cliperase_paper':'CLIPErase adaptation (weight 3)','cliperase_balanced':'CLIPErase adaptation (balanced)','neutral_balanced':'Neutral-pair adaptation (balanced)'}
fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
curves=[]
for kind in plan['methods']:
    rows=[dict(step=0,evaluation=initial)]+[json.loads(s) for s in (OUT/kind/'steps.jsonl').read_text().splitlines() if '"evaluation"' in s]
    xs=[r['step'] for r in rows]
    for ax,(group,key,label) in zip(axes,[('retain_validation','loss','Retained validation CE'),('forget_monitor','alignment_gap_vs_unrelated','Forget alignment gap'),('retain_validation','accuracy','Retained validation accuracy')]):
        ys=[r['evaluation'][group][key] for r in rows]
        ax.plot(xs,ys,marker='o',ms=3,label=names[kind]);ax.set_xlabel('Ideal-link pooled SGD steps');ax.set_ylabel(label);ax.grid(alpha=.25)
    for r in rows:
        for group,values in r['evaluation'].items():curves.append(dict(method=kind,step=r['step'],group=group,**values))
axes[0].axhline(initial['retain_validation']['loss']+.02,color='k',linestyle=':',lw=1)
axes[1].axhline(initial['forget_monitor']['alignment_gap_vs_unrelated']*.9,color='k',linestyle=':',lw=1)
axes[2].axhline(initial['retain_validation']['accuracy']-.01,color='k',linestyle=':',lw=1)
fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=8)
fig.suptitle('Mechanism pilot: shared checkpoint, random-sample pair suppression, seed 0\nNot matched retraining, not federated radio trajectories',fontsize=11)
fig.savefig(OUT/'convergence.png',dpi=180);fig.savefig(OUT/'convergence.pdf');plt.close(fig)
with (OUT/'convergence.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(curves[0]));writer.writeheader();writer.writerows(curves)
lines=['association_objective_v1 완료 결과','',
 '주의: 24회 이상적 통신 pooled SGD 목적함수 진단. 기존 20-round FL 수치와 직접 비교 불가.',
 '보존 검증512, 삭제 모니터200, seed0. 같은 sample request 집합의 연결 억제 실험이며 재학습 동등성 검증이 아님.',
 '사전 기준: 삭제 alignment gap 10% 감소 + 보존 loss 증가<=0.02 + 보존 정확도 하락<=1%p. 마지막24step 판정.','',
 'method | retained loss | retained accuracy | forget gap reduction | retain loss change | gate']
for s in summary:
    v=s['metrics']['retain_validation'];lines.append(f"{s['method']} | {v['loss']:.6f} | {v['accuracy']*100:.3f}% | {s['forget_alignment_gap_reduction_fraction']*100:.3f}% | {s['retain_loss_change']:+.6f} | {s['promotion_gate_passed']}")
lines.extend(['','새 목적함수 통과 후보: '+', '.join(status['promotion_candidates']),
 '에너지/시간 우위 및 삭제 보장을 이 실험으로 주장하지 않는다. teacher KL은 보존 편차 지표이다.',
 '정규화 cosine 감소는 분류/검색 순위 감소와 같지 않을 수 있으므로 matched-minus-unrelated gap으로 평가했다.',
 '기존 class-name 텍스트는 여러 이미지가 공유한다. 무작위 일부 샘플의 연결을 지우는 것은 전체 클래스/개념 삭제와 다르다.',
 '단일 seed이며 작은 파일럿이다. MIA와 matched retraining KL은 이 결과에 포함되지 않았다.'])
(OUT/'results_ko.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
