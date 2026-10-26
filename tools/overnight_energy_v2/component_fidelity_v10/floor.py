"""Conditional separate-U fidelity certificate, not a deletion-quality test.

For fixed peak amplitude A, the smallest available conservative block bound is
min_s>=0 [sum_u max(rho_u-s*H_u*A,0)+s*sqrt(d/(2N))].
For fixed s, every amplitude can be chosen to cancel its mismatch up to A.
The remaining function is convex piecewise linear, so a breakpoint (or0) is
optimal. Maximum possible U phase length gives an optimistic certificate floor.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import unittest
import numpy as np


def block_floor(rho,channel,peak,noise):
    rho=np.asarray(rho);gain=np.asarray(channel)*np.sqrt(peak)
    assert np.all(rho>=0) and np.all(gain>=0) and noise>0
    candidates=np.r_[0,np.divide(rho[gain>0],gain[gain>0])]
    values=np.maximum(rho[None,:]-candidates[:,None]*gain[None,:],0).sum(axis=1)+noise*candidates
    winner=int(np.argmin(values))
    return float(values[winner]),float(candidates[winner])


class FloorTests(unittest.TestCase):
    def test_single_link(self):
        self.assertAlmostEqual(block_floor([.1],[2],1,.25)[0],.1*.25/2)
        self.assertAlmostEqual(block_floor([.1],[2],1,3)[0],.1)
    def test_zero_and_unreachable(self):
        self.assertEqual(block_floor([0,0],[0,2],1,.2)[0],0)
        self.assertEqual(block_floor([.1],[0],1,.2)[0],.1)
    def test_dense_convex_scan_and_amplitude_recovery(self):
        rho=np.array([.3,.1,.6]);channel=np.array([2,3,1]);peak=.32;noise=.4
        best,s=block_floor(rho,channel,peak,noise)
        grid=np.linspace(0,2,100001)
        values=np.maximum(rho[None,:]-grid[:,None]*channel[None,:]*np.sqrt(peak),0).sum(axis=1)+noise*grid
        self.assertLessEqual(best,float(values.min())+1e-12)
        self.assertLess(float(values.min())-best,1e-4)
        amp=np.minimum(np.sqrt(peak),rho/np.maximum(s*channel,1e-30))
        self.assertAlmostEqual(float(np.abs(s*channel*amp-rho).sum()+s*noise),best)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path)
    p.add_argument('--self-test',action='store_true');args=p.parse_args()
    if args.self_test:
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(FloorTests)
        result=unittest.TextTestRunner().run(suite)
        if not result.wasSuccessful():raise SystemExit(1)
        return
    assert args.project is not None
    root=args.project.resolve();sys.path.insert(0,str(root));from wireless import quantize
    source=root/'results/failed_round_diagnostic';meta=json.loads((source/'metadata.json').read_text());c=meta['config']
    with np.load(source/'updates.npz') as a:l,u,h=[a[k].copy() for k in ['learning','unlearning','channels']]
    b=c['block_dimension'];shape=(len(l),-1,b);padding=((0,0),(0,(-l.shape[1])%b))
    learning=np.pad(l,padding).reshape(shape);unlearning=np.pad(u,padding).reshape(shape)
    rho=np.linalg.norm(unlearning,axis=-1);total=np.full(rho.shape[1],b*c['symbols_per_coordinate'])
    codecs={};deltas={}
    for bits in [6,8]:
        decoded,delta,payload=quantize(unlearning,bits,c['scale_metadata_bits']);deltas[bits]=delta
        codecs[bits]=dict(global_all_client_quantization_rms_bound=float(np.linalg.norm(delta.sum(axis=0))),
                          actual_all_client_quantization_error=float(np.linalg.norm((decoded-unlearning).sum(axis=0))),
                          payload_bits=float(payload.sum()))
    rows=[]
    for head in range(c['users']):
        users=np.delete(np.arange(c['users']),head)
        has_learning=np.any(np.linalg.norm(learning[users],axis=-1)>0,axis=0)
        max_u_symbols=total-b*has_learning
        assert np.all(max_u_symbols>=b)
        floor=[];receiver=[]
        for m,n in enumerate(max_u_symbols):
            value,scale=block_floor(rho[users,m],h[users,head],c['power_limit_w'],np.sqrt(b/(2*n)))
            floor.append(value);receiver.append(scale)
        rms=float(np.linalg.norm(floor))
        rows.append(dict(head=head,optimistic_minimum_conservative_u_rms=rms,
                         per_block_floors=floor,receivers=receiver,max_u_symbols=max_u_symbols.tolist(),
                         codec_reference={bits:{'global_tau':codecs[bits]['global_all_client_quantization_rms_bound'],
                           'head_specific_quantization_bound':float(np.linalg.norm(deltas[bits][users].sum(axis=0))),
                           'floor_over_global_tau':rms/codecs[bits]['global_all_client_quantization_rms_bound'],
                           'global_tau_supported_by_this_analog_bound':rms<=codecs[bits]['global_all_client_quantization_rms_bound']} for bits in [6,8]}))
    result=dict(version='component_fidelity_v10_conditional_certificate',at_kst=datetime.now().isoformat(timespec='seconds'),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                updates_sha256=hashlib.sha256((source/'updates.npz').read_bytes()).hexdigest(),
                quantizer_source_sha256=hashlib.sha256((root/'wireless.py').read_bytes()).hexdigest(),
                reference_codecs=codecs,heads=rows,
                explanation='Separate-U accuracy is an explicit extra constraint absent from the original sum-MSE problem. Fixed tau values are the existing6/8-bit codec bounds, independent of head and not chosen from validation/test outcomes. Hybrid and digital-phase meet these quantization bounds only under successful ideal channel coding. Mixed recovers only the sum and has no separately available U output.',
                limits=['This is an optimistic MINIMUM OF THE CONSERVATIVE TRIANGLE BOUND, not a lower bound on actual MSE or a proof of physical infeasibility.',
                        'Maximum possible U time used; simultaneous learning quality, energy minimization and battery constraints are relaxed.',
                        'If floor>tau, this bound cannot certify the separate-U target under the stated peak/time limits. Other statistics/estimators can have smaller actual error.',
                        'No application-quality or deletion guarantee follows from a codec precision target. This diagnostic does not establish a need for individually decoded payloads.',
                        'This is a problem-revision diagnostic, not a replacement or selective filtering of the original additive results. Optimized digital-phase remains a required comparator.',
                        'No new training/inference, test-based tuning, GPU promotion or end-to-end energy winner.'])
    out=root/'results/overnight_energy_v2/component_fidelity_v10';out.mkdir(exist_ok=True)
    path=out/'results.json';assert not path.exists(),'Do not overwrite completed diagnostics'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'reference_codecs':codecs,'head_summary':[{k:v for k,v in x.items() if k not in ['per_block_floors','receivers','max_u_symbols']} for x in rows]}))


if __name__=='__main__':main()
