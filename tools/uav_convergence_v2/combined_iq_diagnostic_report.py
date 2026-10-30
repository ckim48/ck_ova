"""Full diagnostic report: failed runs, paired tensors, and all radio sweeps."""
import argparse
import html
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import combined_iq_diagnostic_v1 as study
import combined_iq_diagnostic_v2 as followup
import combined_iq_experiment_a_report as parent
import forgetting_quality_v1 as audit
import joint_quality_audit_v1 as references

ROOT,read,write,sha=study.ROOT,study.read,study.write,study.sha
DEST=ROOT/'final_final_charts/experiment_a_diagnostics'
DATASETS=['CapERA','AirScape']
MODES=['iq','combined']
DISPLAY={'iq':'I/Q AirComp','combined':'Combined AirComp'}
COLORS=['#4C72B0','#EFAA54']
PLOT_CASES=['baseline','fu_moderate','fu_strict']


def source(case,ds,mode):
    root=study.previous.OUT if case=='baseline' else (followup.OUT if case=='fu_moderate' else study.OUT)/case
    return root/mode/'cases/baseline'/(ds.lower()+'_seed0')/study.previous.learning.quality.PROPOSED


def all_conditions():
    yield 'baseline','combined'
    yield 'baseline','iq'
    for case,cfg in study.CASES.items():
        for mode in cfg['modes']:yield case,mode
    for mode in ['combined','iq']:yield 'fu_moderate',mode


def evaluate():
    import torch
    torch.set_num_threads(1)
    assert (study.OUT/'finished.json').exists() and (followup.OUT/'finished.json').exists(), 'Campaign must finish before final evaluation.'
    for ds in DATASETS:
        base,data,engine_class,plan,payload,anchor=audit.runtime(ds)
        engine=study.previous.learning.quality.engine_wrapper(engine_class)(data,plan,payload)
        paths=dict(common=anchor.parent/'reference200/final.pt',matched=references.OUT/ds.lower()/'uniform_after100_reference/final.pt')
        refs={}
        for key,p in paths.items():
            _,refs[key]=audit.per_video(engine,base.core.load_theta(p,engine.device),plan)
        for case,mode in all_conditions():
            arm=source(case,ds,mode)
            if not (arm/'outcome.json').exists() or read(arm/'outcome.json')['status']!='completed':continue
            theta=base.core.load_theta(arm/'final.pt',engine.device)
            dest=DEST/'evaluation'/ds.lower()/case/mode
            metrics,_=audit.store_eval(engine,theta,plan,dest,refs)
            write(dest/'source.json',dict(checkpoint=str((arm/'final.pt').relative_to(ROOT)),
                checkpoint_sha256=sha(arm/'final.pt'),references={k:dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for k,p in paths.items()}))
            print(ds,case,mode,100*metrics['retained']['mR'],metrics['forget']['matched_KL'],flush=True)


def verify_physical(arm):
    settings=read(arm.parents[1]/'settings.json')['config']
    for f in (arm/'transactions').glob('*.json'):
        r=read(f)['radio'];v=np.array(r['block_variance']);eps=np.array(r['component_epsilon'])**2
        assert np.all(v.sum(1)<=eps*(1+3e-6)+1e-30)
        target=np.array(r['normalized_distortion_limits'])
        assert np.all(np.array(r['normalized_curvature'])@v.sum(0)<=target*(1+3e-6)+1e-30)
        assert r['energy_bound_j']<=.1*(1+2e-6)
        assert r['max_power_w']<=.32*(1+2e-6)
        assert r['max_user_energy_j']<=settings['available_energy_j']*(1+2e-6)
        assert r['uplink_s']<=settings['round_uplink_deadline_s']+1e-12
        assert np.isclose(v.sum(),r['conditional_rmse']**2,rtol=1e-8,atol=1e-30)


def gather():
    rows=[];parent.ledger.source=source;parent.ledger.METHODS=DISPLAY
    for ds in DATASETS:
        for case,mode in all_conditions():
            arm=source(case,ds,mode);r=dict(dataset=ds,case=case,mode=mode,source=str(arm.relative_to(ROOT)))
            if not (arm/'outcome.json').exists():
                r['status']='not_completed_or_not_attempted';rows.append(r);continue
            outcome=read(arm/'outcome.json');r['status']=outcome['status']
            verify_physical(arm)
            if outcome['status']!='completed':
                r['outcome']=outcome
                if (arm/'latest.json').exists():r['last_completed_FL_round']=read(arm/'latest.json').get('completed_FL_rounds')
                rows.append(r);continue
            cost=parent.ledger.ledger(case,ds,mode)
            metrics=read(DEST/'evaluation'/ds.lower()/case/mode/'metrics.json')
            r.update(mR=100*metrics['retained']['mR'],loss=metrics['retained']['loss'],
                common_KL=metrics['forget']['common_KL'],matched_KL=metrics['forget']['matched_KL'],
                energy_kJ=cost['total_energy_j']/1000,delay_s=cost['delay_s'],
                RF_energy_j=sum(read(p)['radio']['energy_bound_j'] for p in (arm/'transactions').glob('*.json')))
            assert cost['FU_cycles']==225 and read(arm/'verification.json')['FL_rounds']==200
            rows.append(r)
    return rows


def tensor_checks():
    import torch
    result=[]
    def theta(p):return torch.load(p,map_location='cpu',weights_only=False)['theta']
    for ds in DATASETS:
        original=source('baseline',ds,'iq');replay=source('balanced_replay',ds,'iq');mapped=source('mapped_noise_control',ds,'combined')
        assert torch.equal(theta(original/'final.pt'),theta(replay/'final.pt'))
        assert torch.equal(theta(replay/'final.pt'),theta(mapped/'final.pt'))
        rt=[read(p) for p in sorted((replay/'transactions').glob('*.json'))]
        mt=[read(p) for p in sorted((mapped/'transactions').glob('*.json'))]
        assert len(rt)==len(mt)==180
        for a,b in zip(rt,mt):
            ar,br=a['radio'],b['radio']
            assert ar['head']==br['head'] and ar['frame_count']==br['frame_count']
            assert np.allclose(np.sum(ar['block_variance'],axis=0),np.sum(br['block_variance'],axis=0),rtol=1e-12,atol=1e-30)
            assert br['energy_bound_j']<=ar['energy_bound_j']*(1+2e-8)+1e-18
            assert br['max_power_w']<=ar['max_power_w']*(1+2e-8)+1e-18
        result.append(dict(dataset=ds,original_reproduced_bitwise=True,mapped_Combined_equals_IQ_bitwise=True,
            transactions=180,same_head_duration_and_noise_covariance=True,no_greater_Combined_RF_power=True,
            IQ_checkpoint_sha256=sha(replay/'final.pt'),mapped_checkpoint_sha256=sha(mapped/'final.pt')))
    return result


def sensitivity(sweep):
    grouped=[]
    for ds in DATASETS:
        for gain in [0.,-6.,-12.]:
            for ratio in [.1,.25,1.,4.,10.]:
                for mode in MODES:
                    ps=[r for r in sweep if r['dataset']==ds and r['gain_db']==gain and r['tolerance_ratio']==ratio and r['mode']==mode]
                    assert len(ps)==5
                    ok=[r for r in ps if r['status']=='feasible']
                    r=dict(dataset=ds,gain_db=gain,tolerance_ratio=ratio,mode=mode,
                        feasible_snapshots=len(ok),total_snapshots=5,feasible_percent=20*len(ok))
                    if ok:
                        r.update(uplink_phase_energy_j=float(np.mean([p['uplink_phase_energy_j'] for p in ok])),
                            RF_energy_j=float(np.mean([p['RF_energy_j'] for p in ok])),
                            FU_C8_usage_percent=100*float(np.mean([p['FU_C8_usage'] for p in ok])),
                            C9_usage_percent=100*float(np.mean([p['C9_usage'] for p in ok])))
                    grouped.append(r)
    return grouped


def save(fig,axes,name,caption,extra_handles=None):
    if extra_handles:
        for h in extra_handles:axes[0].add_artist(h)
        handles,labels=axes[0].get_legend_handles_labels()
        for h in extra_handles:
            if h.get_label() not in labels:handles.append(h);labels.append(h.get_label())
        by_label=dict(zip(labels,handles))
        labels=[label for label in DISPLAY.values() if label in by_label]+[h.get_label() for h in extra_handles]
        handles=[by_label[label] for label in labels]
        fig.legend(handles,labels,ncol=3,loc='upper center',bbox_to_anchor=(.54,1.01),frameon=False,
            fontsize=11.5,handlelength=1.4,columnspacing=1.1)
        # Parent save would add another legend; remove the manual artists and keep the normal method legend.
        # Use a separate export below so failure marks have an explicit legend.
        files=[]
        fig.canvas.draw()
        for ext in ['pdf','png','svg']:
            p=DEST/(name+'.'+ext);fig.savefig(p,dpi=250,bbox_inches=None,facecolor='white')
            if ext=='svg':p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
            shutil.copyfile(p,ROOT/'final_final_charts/figures'/p.name)
            files.append(dict(file=p.name,sha256=sha(p)))
        plt.close(fig);parent.FIGURES.append(dict(name=name,caption=caption,files=files))
    else:parent.save(fig,axes,name,caption)


def draw_sweep(grouped):
    xs=[.1,.25,1.,4.,10.]
    for num,gain in [(41,0.),(42,-6.)]:
        fig,axes=parent.canvas('Uplink-phase energy (J)');failed_any=False
        for ax,ds in zip(axes,DATASETS):
            panel=[r for r in grouped if r['dataset']==ds and r['gain_db']==gain]
            vals=[r['uplink_phase_energy_j'] for r in panel if r['feasible_snapshots']==5]
            parent.yscale(ax,vals or [0.,1.],zero=True,decimals=0)
            for i,mode in enumerate(MODES):
                points=[next(r for r in panel if r['mode']==mode and r['tolerance_ratio']==x) for x in xs]
                y=[r['uplink_phase_energy_j'] if r['feasible_snapshots']==5 else np.nan for r in points]
                ax.plot(xs,y,color=COLORS[i],ls=['-','--'][i],marker=['o','s'][i],mfc='white',ms=4.8,lw=1.5,label=DISPLAY[mode])
                for x,r in zip(xs,points):
                    if r['feasible_snapshots']<5:
                        failed_any=True;ax.plot(x,.92*ax.get_ylim()[1],marker='x',color=COLORS[i],ms=7,ls='none',clip_on=False)
            ax.set_xscale('log');ax.set_xlim(.1,10.);ax.set_xticks([.1,1.,10.]);ax.set_xticklabels(['0.1','1','10'])
            ax.set_xlabel('FU/FL tolerance ratio',fontsize=14)
        handles=[Line2D([],[],color='#333333',marker='x',ls='none',label='Not all feasible')] if failed_any else None
        save(fig,axes,f'{num}_iq_tolerance_energy_1x2',
            f'Uplink-phase energy versus the FU-to-FL relative error tolerance ratio at channel-power offset {gain:g} dB. Five fixed VLM updates per dataset; energy means plotted only where all five designs are feasible.',handles)
    fig,axes=parent.canvas('Feasible designs (%)')
    for ax,ds in zip(axes,DATASETS):
        for i,mode in enumerate(MODES):
            ps=[next(r for r in grouped if r['dataset']==ds and r['gain_db']==-12 and r['mode']==mode and r['tolerance_ratio']==x) for x in xs]
            ax.plot(xs,[r['feasible_percent'] for r in ps],color=COLORS[i],ls=['-','--'][i],marker=['o','s'][i],mfc='white',ms=4.8,lw=1.5,label=DISPLAY[mode],clip_on=False)
        ax.set_xscale('log');ax.set_xlim(.1,10.);ax.set_xticks([.1,1.,10.]);ax.set_xticklabels(['0.1','1','10'])
        ax.set_ylim(0,100);ax.set_yticks([0,25,50,75,100]);ax.set_xlabel('FU/FL tolerance ratio',fontsize=14)
    save(fig,axes,'43_iq_tolerance_feasibility_1x2','Fraction of feasible radio designs under a -12 dB channel-power offset on five fixed VLM updates per dataset. Solver failures are not proofs of infeasibility.')


def draw_endpoints(rows):
    for num,key,label,scale,decimals in [(44,'matched_KL',r'Forget-set KL ($\times 10^{-3}$)',1000,2),(45,'mR','Retained mR (%)',1,1),(46,'energy_kJ','Total energy (kJ)',1,0)]:
        fig,axes=parent.canvas(label);failed=False
        for ax,ds in zip(axes,DATASETS):
            values=[]
            for i,mode in enumerate(MODES):
                for j,case in enumerate(PLOT_CASES):
                    r=next(r for r in rows if r['dataset']==ds and r['case']==case and r['mode']==mode)
                    x=j+[-.18,.18][i]
                    if r['status']=='completed':
                        val=r[key]*scale;values.append(val)
                        ax.bar(x,val,width=.32,color=COLORS[i],hatch=['//','\\\\'][i],edgecolor='#333333',lw=.6,label=DISPLAY[mode] if j==0 else None)
                    else:
                        failed=True;ax.plot(x,0,marker='x',color=COLORS[i],ms=7,ls='none',clip_on=False)
            parent.yscale(ax,values or [0.,1.],zero=True,decimals=decimals)
            ax.set_xlim(-.6,2.6);ax.set_xticks(range(3));ax.set_xticklabels(['Balanced','Tighter\nFU','Strict\nFU'],fontsize=10.5)
        handles=[Line2D([],[],color='#333333',marker='x',ls='none',label='Incomplete run')] if failed else None
        save(fig,axes,f'{num}_iq_diagnostic_{key}_1x2',f'Comparison of {label} after the same200 FL rounds and225 FU cycles. Balanced, tighter FU and strict FU use relative tolerance ratios1,0.25 and0.1, respectively. Incomplete runs have no final performance bar.',handles)


def report():
    DEST.mkdir(parents=True,exist_ok=True);study.register()
    assert (study.OUT/'finished.json').exists() and (study.OUT/'snapshot_sweep/finished.json').exists()
    rows=gather();checks=tensor_checks();sweep=read(study.OUT/'snapshot_sweep/results.json');grouped=sensitivity(sweep)
    proofs=read(study.OUT/'snapshot_sweep/constructive_checks.json')
    assert len(sweep)==300 and all(r['same_aggregate_MSE'] and r['no_greater_RF_energy'] for r in proofs)
    activity=[]
    for case,root in [('baseline',study.previous.OUT),*[(c,study.OUT/c) for c in study.CASES]]:
        activity += [dict(case=case,**r) for r in study.binding_rows(root)]
    activity += [dict(case='fu_moderate',**r) for r in study.binding_rows(followup.OUT/'fu_moderate')]
    for name,rs in [('endpoints',rows),('snapshot_sweep',sweep),('sensitivity_summary',grouped),('constraint_activity',activity),('constructive_checks',proofs)]:
        write(DEST/(name+'.json'),rs);parent.ledger.write_csv(DEST/(name+'.csv'),rs)
    write(DEST/'tensor_equivalence.json',checks)
    shutil.copyfile(study.OUT/'protocol.json',DEST/'protocol.json')
    shutil.copyfile(followup.OUT/'protocol.json',DEST/'moderate_followup_protocol.json')
    for name in ['feasibility_bounds.json','bounds_summary.json']:
        shutil.copyfile(study.OUT/'snapshot_sweep'/name,DEST/name)
    waveform=read(study.OUT/'mapped_waveform_check.json');assert waveform['passed']
    shutil.copyfile(study.OUT/'mapped_waveform_check.json',DEST/'mapped_waveform_check.json')
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'pdf.fonttype':42,'svg.fonttype':'none'})
    parent.DEST=DEST;parent.FIGURES=[]
    draw_sweep(grouped);draw_endpoints(rows)
    write(DEST/'verification.json',dict(passed=True,source_sha256=sha(Path(__file__)),
        physical_limits_checked=True,all_conditions_retained=True,tensor_equivalence=checks,
        independent_complex_waveform_checked=True,
        snapshot_designs=300,constructive_checks=len(proofs),figures=parent.FIGURES,
        completed_arms=sum(r['status']=='completed' for r in rows),total_arms=len(rows)))
    (DEST/'captions.tex').write_text('\n\n'.join('% '+f['name']+'\n\\caption{'+f['caption']+'}' for f in parent.FIGURES)+'\n')
    proof=r'''For each block, let v_L=nu/x_L and v_U=nu/x_U be the two I/Q noise variances. Set x_C=nu/(v_L+v_U), omitting inactive branches. Then the Combined noise variance equals v_L+v_U. Weighted Cauchy-Schwarz gives

    x_C ||l_u+u_u||^2 <= x_L ||l_u||^2 + x_U ||u_u||^2.

The channel factor1/|h_u|^2 is common. Thus every device/block has no greater RF energy and, at the same symbol count, no greater transmit power. The final-update noise covariance and the sum-based C9 distortion are unchanged. Both models apply the same final update when their Gaussian noise is coupled. The mapped control uses this construction; it is not a new optimized deployment algorithm.

This argument applies to the present real local updates, perfect phase inversion, AWGN, block-isotropic independent I/Q noise, and a final linear sum. It does not provide separate FL/FU estimates for Combined, nor assert equivalence for nonlinear receiver processing, unknown phase, or different information constraints. Individual C8 limits are additional requirements for I/Q; they are not observable task errors of a one-stream combined estimate.
'''
    (DEST/'sum_only_equivalence.txt').write_text(proof)
    text=['I/Q diagnostic: all outcomes',
        'Original C8 bounds were slack during every joint update while C9 was active. The new full runs tighten the FU relative tolerance by10x while keeping total MSE and C9 targets fixed, and then additionally reduce channel power by12 dB. Both methods retain identical learning objectives, data, power caps and resource budgets.',
        'After the10x-tight I/Q runs could not find a feasible radio design, a disclosed4x-tight follow-up (ratio0.25) was registered for both methods and both datasets. This setting was chosen because all ten registered regional snapshots were feasible at that ratio, before observing its final mR/KL. The earlier incomplete outcomes remain in this report.',
        'Both original I/Q models were reproduced bitwise. A Combined construction with identical final-noise covariance reproduced BOTH I/Q final tensors bitwise, with no greater per-device/block RF power across360 transactions.',
        'The mapped end-to-end run deliberately shares the final Gaussian update, so the tensor equality is a controlled consequence, not an independent quality win. A separate complex-waveform calculation on all ten stored snapshots verifies the same single-stream construction, random channel phases, per-user block-energy inequality, and Monte Carlo noise variance. See mapped_waveform_check.json.',
        proof,
        'The300 frozen-update designs use five predetermined region snapshots per dataset, three channel offsets and five tolerance ratios. These are resource sensitivity results, not new training seeds. All solver failures are retained. They do not establish impossibility of another feasible optimizer solution.',
        'A separate solver-independent audit uses necessary lower bounds on RF energy and required symbol expansion. It can prove some failed grid conditions infeasible under the registered limits; unresolved failures remain only solver failures. See bounds_summary.json and feasibility_bounds.json. No necessary lower bound was violated by a reported feasible point.',
        'Full-run bars contain only completed200 FL round/225 FU cycle endpoints. Missing endpoints are explicitly marked. Same common and policy-matched retained-only references are used for all completed methods. Neither is a training teacher.',
        'Active energy covers computation, transmission, broadcast, circuits and hovering after the common20-round warm start. Fixed travel and the common warm start are excluded, as in experiment A. No flight-energy or hardware-runtime inference is made.',
        'No favorable condition is selected for promotion. These are single-training-seed diagnostics; I/Q superiority is not assumed. Component separation by itself does not use nonlinear FL/FU interactions at the head.',
        'Endpoints:',*map(str,rows)]
    (DEST/'README.txt').write_text('\n\n'.join(text)+'\n')
    ko=['Combined 대 I/Q 추가 진단 결과',
        '기존 실험은 FL/FU 개별 C8 제약이45개 joint round 모두에서 경계까지 활성화되지 않고, 합산 오차 C9 제약이 자원을 결정했다. 따라서 개별 오차 제어의 효용을 확인하기 어려운 조건이었다.',
        'FU/FL 상대 허용 오차 비율을1에서0.25로 바꾸어 추가200 FL round/225 FU step 실험을 수행했다. 총 MSE 예산과 C9 예산은 유지했다. I/Q의 FU 개별 제약이 두 데이터셋 모두45개 joint round 전부에서 활성화되었다.']
    table=[]
    for ds in DATASETS:
        iq=next(r for r in rows if r['dataset']==ds and r['case']=='fu_moderate' and r['mode']=='iq')
        combined=next(r for r in rows if r['dataset']==ds and r['case']=='fu_moderate' and r['mode']=='combined')
        assert iq['status']==combined['status']=='completed'
        kl=100*(iq['matched_KL']/combined['matched_KL']-1)
        energy=100*(iq['energy_kJ']/combined['energy_kJ']-1)
        delay=100*(iq['delay_s']/combined['delay_s']-1)
        ko.append(f'{ds}: I/Q와 Combined mR 차이{iq["mR"]-combined["mR"]:+.5f}%p; I/Q matched KL은{kl:.3f}% 증가, 에너지는{energy:.3f}% 증가, 지연은{delay:.3f}% 증가했다.')
        table.append(f'<tr><td>{ds}</td><td>{iq["mR"]-combined["mR"]:+.5f}%p</td><td>+{kl:.3f}%</td><td>+{energy:.3f}%</td><td>+{delay:.3f}%</td></tr>')
    ko += ['같은 최종 수신 오차를 갖도록 구성한 Combined 대조군은 CapERA와 AirScape 모두 I/Q의 최종 파라미터를 비트 단위까지 동일하게 재현했다. 모든360개 전송에서 per-device/block RF 전력이 I/Q보다 크지 않음을 확인했다. 이 대조군은 수학적 동등성을 검증하는 진단이며 별도의 최적화된 실용 알고리즘을 주장하지 않는다.',
        '동일 최종 잡음을 의도적으로 공유했으므로 모델 일치는 통제 조건의 결과다. 별도의 복소 송수신 파형 계산으로 실제 Combined 단일 신호가 이 잡음 통계와 전력 조건을 구현할 수 있음을10개 실제 업데이트에서 검산했다. 이 결과를 독립적인 성능 승리로 해석하지 않는다.',
        '개별 C8 정확도 요구를 Combined가 충족했다고 해석하면 안 된다. 단일 합산 수신값에서는 FL/FU 별 추정 오차가 정의되지 않는다. 이 검증은 최종적으로 FL+FU를 선형 합산하는 현재 모델의 합산 오차와 모델 성능에 관한 것이다.',
        '현재의 완전한 위상 보상, 독립 Gaussian 잡음, 최종 선형 합산 조건에서는 무선 I/Q 분리만으로 일관된 mR/KL 이득을 확인하지 못했다. 개별 오차 한도를 강화한 것만으로는 모델 성능 개선으로 이어지지 않았다. 단일 development seed 결과로 통계적 우위를 주장하지 않는다.',
        '비율0.1 및 -12dB 스트레스 조건의 미완료 결과도 모두 보존했다. 실제 VLM 업데이트10개에서 무선 설계300개를 비교했으며, no-feasible105개 중81개는 솔버와 별개로 계산한 필요조건 하한에서도 불가능했다. 나머지24개는 이 하한만으로 불가능성을 확정하지 않았다.',
        '41/42: 기본 및-6dB 채널에서의 무선 에너지 민감도. 43:-12dB 조건의 설계 성공 비율. 44/45/46: 동일200라운드의 KL/mR/에너지 비교. X는 미완료를 뜻하며0 성능 또는0 비용으로 해석하지 않는다.',
        '세부 수치:endpoints.csv, snapshot_sweep.csv, constraint_activity.csv. 수학적 조건:sum_only_equivalence.txt. 사전 조건과 추가 실행의 이유:protocol.json, moderate_followup_protocol.json.']
    (DEST/'RESULTS_ko.txt').write_text('\n\n'.join(ko)+'\n')
    cards=''.join('<section><h2>'+html.escape(f['caption'])+'</h2><a href="'+f['name']+'.pdf"><img src="'+f['name']+'.png"></a></section>' for f in parent.FIGURES)
    summary='<p>FU/FL 상대 허용 오차 비율을1에서0.25로 낮춘 조건: I/Q minus Combined. KL·에너지·지연은 증가할수록 불리합니다.</p><table><tr><th>Dataset</th><th>mR 변화</th><th>KL 변화</th><th>에너지 변화</th><th>지연 변화</th></tr>'+''.join(table)+'</table>'
    (DEST/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>I/Q diagnostic</title><style>body{max-width:1100px;margin:30px auto;font:16px system-ui}img{width:100%}pre{white-space:pre-wrap}h2{font-size:18px}section{margin:35px 0}table{border-collapse:collapse}th,td{padding:10px;border-bottom:1px solid #ccc}</style><h1>I/Q diagnostic</h1>'+summary+'<pre>'+html.escape('\n\n'.join(ko))+'</pre>'+cards+'<details><summary>Full protocol and outcomes</summary><pre>'+html.escape('\n\n'.join(text))+'</pre></details>')
    print('verified',len(rows),'arms;',sum(r['status']=='completed' for r in rows),'completed;',len(sweep),'radio points',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['evaluate','report','all'],nargs='?',default='all');a=p.parse_args()
    if a.action in ['evaluate','all']:evaluate()
    if a.action in ['report','all']:report()
