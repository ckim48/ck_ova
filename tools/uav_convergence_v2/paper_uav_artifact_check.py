"""Check exported figure values against audited arm metrics and rendered axes."""
import math
from pathlib import Path
import numpy as np
from PIL import Image
import paper_uav_charts as plots
from adaptive_fu import ROOT,read,write,sha


def main():
    out=plots.DEST;m=read(out/'data/manifest.json');assert m['complete']
    assert m['compared_arms']==126 and len(m['figures'])==16
    rows=read(out/'data/metrics.json');series=read(out/'data/plot_series.json');assert len(rows)==126 and len(series)==96
    metrics={(r['case'],r['dataset'],r['method']):r for r in rows}
    for s in series:
        n=int(s['figure'].split('_')[0])
        if n<=10:
            axis=list(plots.study.AXES)[(n-1)//2];a=plots.study.AXES[axis]
            assert s['x']==a['values']
            key,scale=('total_energy_j',.001) if n%2 else ('delay_s',1.)
            expected=[metrics[(plots.study.lib.case_id(axis,v),s['dataset'],s['method'])][key]*scale for v in s['x']]
            assert np.allclose(s['y'],expected,rtol=0,atol=1e-10)
        else:
            metric={'mR':'retained_mR_percent','loss':'retained_loss','KL':'forget_KL'}[s['metric']]
            row=metrics[('baseline',s['dataset'],s['method'])]
            assert math.isclose(s['y'][-1],row[metric],abs_tol=1e-12)
            if '_time_' in s['figure']:assert math.isclose(s['x'][-1],row['delay_s'],abs_tol=1e-8)
            else:assert s['x'][-1]==200
            assert all(a<=b for a,b in zip(s['x'],s['x'][1:]))
            assert s['display_y'][-1]==s['y'][-1]
    for f in m['figures']:
        for panel in f['axes']:
            for dim in ['x','y']:
                a=panel[dim];assert np.allclose(np.diff(a['ticks']),a['ticks'][1]-a['ticks'][0])
                assert a['limits']==[a['ticks'][0],a['ticks'][-1]]
        for ext in f['files']:
            path=out/ext['file'];assert sha(path)==ext['sha256']
            if path.suffix=='.png':
                im=Image.open(path);assert im.width>1800 and im.height>800
    for name,d in m['source_sha256'].items():assert sha(ROOT/name)==d,name
    result=dict(passed=True,figures=16,arms=126,series=96,checks=['Exported sweep points equal audited per-arm metrics','Convergence endpoints equal actual final models','Time endpoints equal actual completion clocks','Uniform endpoint ticks and verified figure bytes','No unregistered plot source change'],
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))})
    write(out/'data/artifact_checks.json',result);print(result)


if __name__=='__main__':main()
