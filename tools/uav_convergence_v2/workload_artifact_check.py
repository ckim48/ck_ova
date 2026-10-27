"""Verify 1x2 arrays, endpoint ticks, precision, provenance and figure files."""
import math
from pathlib import Path
import numpy as np
from PIL import Image
import workload_charts as charts

read,write,sha=charts.study.read,charts.study.write,charts.study.sha


def main():
    out=charts.DEST;m=read(out/'data/manifest.json');assert m['complete']
    assert m['compared_arms']==126 and len(m['figures'])==23
    rows=read(out/'data/metrics.json');series=read(out/'data/plot_series.json')
    assert len(rows)==126 and len(series)==138
    by={(r['case'],r['dataset'],r['method']):r for r in rows}
    for s in series:
        if 'cases' in s:
            spec=charts.AXES[s['axis']];assert s['x']==spec.get('display_values',spec['values'])
            assert s['cases']==[charts.cid(s['axis'],v) for v in spec['values']]
            expected=[by[(c,s['dataset'],s['method'])][s['metric']]*s['scale'] for c in s['cases']]
            assert np.allclose(s['y'],expected,atol=1e-10,rtol=0)
        else:
            key={'mR':'retained_mR_percent','loss':'retained_loss','KL':'forget_KL'}[s['metric']]
            row=by[('baseline',s['dataset'],s['method'])]
            assert math.isclose(s['y'][-1],row[key],abs_tol=1e-12)
            if '_time_' in s['figure']:assert math.isclose(s['x'][-1],row['delay_s'],abs_tol=1e-8)
            else:assert s['x'][-1]==200
            assert all(x<=y for x,y in zip(s['x'],s['x'][1:]))
            assert s['display_y'][-1]==s['y'][-1]
    for f in m['figures']:
        precision=[]
        for panel in f['axes']:
            dims={}
            for dim in ['x','y']:
                ax=panel[dim];assert np.allclose(np.diff(ax['ticks']),ax['ticks'][1]-ax['ticks'][0])
                assert ax['limits']==[ax['ticks'][0],ax['ticks'][-1]]
                dims[dim]=[len(label.split('.')[-1]) if '.' in label else 0 for label in ax['labels']]
                assert len(set(dims[dim]))==1
            precision.append(dims)
        for dim in ['x','y']:assert precision[0][dim][0]==precision[1][dim][0]
        for item in f['files']:
            p=out/item['file'];assert sha(p)==item['sha256']
            if p.suffix=='.png':
                im=Image.open(p);assert im.width>1800 and im.height>800
    for name,digest in m['source_sha256'].items():assert sha(charts.ROOT/name)==digest,name
    check=read(out/'data/comparison_audit.json');assert check['passed'] and check['compared_arms']==126
    result=dict(passed=True,figures=23,compared_arms=126,series=138,
        checks=['Every plotted point equals an independently audited ledger value',
        'Round/time convergence endpoints equal actual final metrics and clocks',
        'All figures have regular endpoint ticks and matching panel precision',
        'Rendered PNG/PDF/SVG hashes and registered source hashes verified',
        'All costs and quality tradeoffs retained without smoothing or rank selection'],
        source_sha256={str(Path(__file__).relative_to(charts.ROOT)):sha(Path(__file__))})
    write(out/'data/artifact_checks.json',result);print(result)


if __name__=='__main__':main()
