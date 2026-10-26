"""Read-only normalized export for one data set and one registered seed."""
from pathlib import Path
import argparse, json, gzip, base64, importlib.util
ROOT=Path('/home/wnlab/Documents/Codex/2026-10-20/files-pasted-by-the-user-standalone/outputs/ova_CK')

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['RSITMD','RSICD'],required=True)
    p.add_argument('--seed',type=int,choices=[0,1,2],required=True);p.add_argument('--base64',action='store_true')
    a=p.parse_args()
    if a.dataset=='RSITMD' and a.seed==0:
        m=load('old_export',ROOT/'tools/overnight_energy_v2/comparison_figures_sequential_cap100_v2/export_results.py')
        data=m.export()
    else:
        m=load('new_export',ROOT/'tools/overnight_energy_v2/comparison_figures_two_datasets_v1/export_rsicd.py')
        if a.seed:
            m.CAMPAIGN=ROOT/f'results/overnight_energy_v2/three_seed_cap100_v1/{a.dataset.lower()}_seed{a.seed}'
        data=m.export()
    data.update(dataset=a.dataset,seed=a.seed)
    value=json.dumps(data,ensure_ascii=True)
    print('EXPORT_START'+base64.b64encode(gzip.compress(value.encode())).decode()+'EXPORT_END' if a.base64 else value)
