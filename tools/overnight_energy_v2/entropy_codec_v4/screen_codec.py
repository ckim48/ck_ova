"""Measure real lossless payloads; this alone is not an energy advantage."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

import codec


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    args=parser.parse_args();root=args.project.resolve()
    sys.path[:0]=[str(root),str(root/'tools/overnight_energy_v2')]
    from phase_split_grid import make_inputs
    source=root/'results/failed_round_diagnostic'
    config=json.loads((source/'metadata.json').read_text())['config']
    with np.load(source/'updates.npz') as data:
        learning,unlearning=[data[k].copy() for k in ['learning','unlearning']]
    output=root/'results/overnight_energy_v2/entropy_codec_v4';output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for bits in [8,6]:
        c=dict(config,quantization_bits=bits)
        _,u,decoded,delta,original,*_=make_inputs(learning,unlearning,c)
        payload=np.zeros_like(original)
        for client in range(c['users']):
            packet=codec.encode(u[client],bits)
            restored=codec.decode(packet,u.shape[1],u.shape[2])
            np.testing.assert_array_equal(restored,decoded[client])
            payload[client]=codec.fragment_bits(packet,u.shape[1])
            if not packet:continue
            samples=[]
            for _ in range(7):
                start=time.perf_counter();again=codec.encode(u[client],bits);middle=time.perf_counter()
                codec.decode(again,u.shape[1],u.shape[2]);end=time.perf_counter()
                assert again==packet
                samples.append([middle-start,end-middle])
            row={'bits':bits,'client':client,'original_bits':int(original[client].sum()),
                'packet_bits':len(packet)*8,'added_pre_design_length_control_bits':32,
                'packet_over_original_ratio':len(packet)*8/float(original[client].sum()),
                'codec_method':codec.HEADER.unpack(packet[:codec.HEADER.size])[1],
                'host_encode_median_s':float(np.median(np.array(samples)[:,0])),
                'host_decode_median_s':float(np.median(np.array(samples)[:,1])),
                'packet_sha256':hashlib.sha256(packet).hexdigest()}
            rows.append(row);(output/f'bits{bits}_client{client}.bin').write_bytes(packet)
        np.savez_compressed(output/f'bits{bits}_payloads.npz',payload=payload,original_payload=original,delta=delta)
    report={'rows':rows,'updates_sha256':hashlib.sha256((source/'updates.npz').read_bytes()).hexdigest(),
        'codec_sha256':hashlib.sha256(Path(codec.__file__).read_bytes()).hexdigest(),
        'screen_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'interpretation':'Quantized integers and float32 scales reconstruct exactly as the original quantizer. All packet/header bytes are counted. A client-wide stream is evenly fragmented across wireless blocks; digital fragments no longer correspond to U coordinate blocks, and reconstruction occurs after the round. Extra compressed-length control needs 32 bits per active client before head/power design. Host CPU timings are not UAV energy measurements. Compression/decoding energy, control energy and delay must be charged before claiming any deployable energy win; baselines receive the same codec.'}
    (output/'payload_screen.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
