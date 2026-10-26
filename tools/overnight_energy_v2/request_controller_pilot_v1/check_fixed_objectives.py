"""Evaluate fixed, already-used training batches; never tune on these values."""
from run_pilot import ROOT, SOURCE, OUT, write
import json, fcntl, torch
from train import Engine
from dataset import RetrievalData
from objectives import contrastive, relation_kl


def main():
    assert json.loads((OUT/'status.json').read_text())['stage']=='completed'
    lock=(ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((SOURCE/'plan.json').read_text())
    engine=Engine(RetrievalData(),plan)
    initial=torch.load(SOURCE/'proposed/round_020.pt',map_location='cpu',weights_only=True)['theta'].to(engine.device)
    engine.set_teacher(initial)
    batches=[x for x in json.loads((OUT/'baseline_sum/round_021.json').read_text())['local'] if 'forget_batch' in x]
    results={}
    for label in ['initial','baseline_sum','request_qp','retain_only_control']:
        theta=initial if label=='initial' else torch.load(OUT/label/'round_025.pt',map_location='cpu',weights_only=True)['theta'].to(engine.device)
        rows=[]
        with torch.no_grad():
            for rec in batches:
                fi,ri=rec['forget_batch'],rec['retained_batch']
                fc,rc=engine.data.training_caption_ids(fi,21),engine.data.training_caption_ids(ri,21)
                fim,ftx,s=engine.features(theta,fi,fc)
                ix=torch.tensor(fi,device=engine.device)
                fce=contrastive(fim,ftx,s,ix,ix)
                rim,rtx,s=engine.features(theta,ri,rc)
                ix=torch.tensor(ri,device=engine.device)
                rce=contrastive(rim,rtx,s,ix,ix)
                ti,tt=engine.teacher_features(ri,rc)
                relation=relation_kl(rim,rtx,ti,tt,s)
                rows.append(dict(user=rec['user'],beta=rec['beta'],forget_CE=float(fce),retained_CE=float(rce),
                                 retained_relation_KL=float(relation),J=float(-fce+relation)))
        results[label]=dict(requests=rows,weighted_J=sum(x['beta']*x['J'] for x in rows),
                            weighted_forget_CE=sum(x['beta']*x['forget_CE'] for x in rows))
    write(OUT/'fixed_batch_objective_check.json',dict(scope='Fixed round21 training batches/captions at initial and final25 models; descriptive only, no settings selected.',results=results))
    print(json.dumps({k:{n:v for n,v in x.items() if n!='requests'} for k,x in results.items()}),flush=True)


if __name__=='__main__':main()
