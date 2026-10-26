"""Reject or identify low-rank prompt compression before any codec/radio run."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def q8(x):
    scale = np.float32(np.max(np.abs(x))/127) if np.any(x) else np.float32(1)
    return np.clip(np.rint(x/scale), -127, 127)*float(scale)


def blocks(x, size=512):
    return np.pad(x, (0, (-len(x)) % size)).reshape(-1, size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    root = parser.parse_args().project.resolve(); sys.path.insert(0, str(root))
    from wireless import quantize
    source = root/'results/failed_round_diagnostic'; p = source/'updates.npz'
    expected = json.loads((root/'results/comparisons_seed0_solver_v2/suite_metadata.json').read_text())['source_sha256']
    assert all(sha(root/n)==h for n,h in expected.items())
    with np.load(p) as f:
        learning, unlearning = [f[k].copy() for k in ['learning', 'unlearning']]
    c = json.loads((source/'metadata.json').read_text())['config']
    # Verified against DeepPromptCLIP.parts: V[8,2,768], T[8,2,512], scalar S.
    assert c['prompt_layers']==8 and c['prompt_tokens']==2 and unlearning.shape[1]==20481
    l = np.stack([blocks(x) for x in learning]); u = np.stack([blocks(x) for x in unlearning])
    epsilon = float(c['relative_tolerance']*np.linalg.norm(np.linalg.norm(l,axis=-1).sum(axis=0)))
    decoded6, delta6, payload6 = quantize(u, 6, c['scale_metadata_bits'])
    decoded8, delta8, payload8 = quantize(u, 8, c['scale_metadata_bits'])
    active = np.flatnonzero(np.any(unlearning != 0, axis=1))
    rows = []; spectra = []; pair_deltas = {}
    for user in active:
        original = unlearning[user]
        matrices = [original[:12288].reshape(16,768),original[12288:20480].reshape(16,512)]
        svds = [np.linalg.svd(x,full_matrices=False) for x in matrices]
        assert all(np.allclose((a*s)@b,x,atol=1e-15,rtol=1e-10) for x,(a,s,b) in zip(matrices,svds))
        shared = original[-1:].astype(np.float32).astype(np.float64)
        spectra.append({'user':int(user), 'unlearning_norm':float(np.linalg.norm(original)),
                        'v_singular_values':svds[0][1].tolist(),'t_singular_values':svds[1][1].tolist(),
                        'six_bit_payload_bits':int(payload6[user].sum()),
                        'six_bit_actual_error':float(np.linalg.norm(decoded6[user]-u[user])),
                        'six_bit_bound':float(np.linalg.norm(delta6[user])),
                        'eight_bit_payload_bits':int(payload8[user].sum())})
        for rank in range(1,17):
            reconstruction = []
            for left,s,right in svds:
                reconstruction.append((q8(left[:,:rank]*s[:rank])@q8(right[:rank])).ravel())
            reconstructed = np.concatenate(reconstruction+[shared])
            delta = np.linalg.norm(blocks(reconstructed-original),axis=-1)
            # Optimistic illustrative raw-int8 factors, four float32 scales,
            # float32-rounded S, and 32-byte metadata/header/checksum allowance.
            # No implemented packet, control signaling, SVD cost or RF claim.
            estimated_bits = 8*rank*(16+768+16+512)+4*32+32+256
            rows.append({'user':int(user),'representation':'modality_matrix_svd_q8', 'rank':rank,
                         'unimplemented_factor_payload_estimate_bits':estimated_bits,
                         'actual_error':float(np.linalg.norm(delta)),
                         'bound_ratio_to_total_epsilon':float(np.linalg.norm(delta)/epsilon),
                         'every_block_error_no_greater_than_original_six_bit_bound':bool(np.all(delta<=delta6[user]+1e-15)),
                         'actual_error_no_greater_than_original_six_bit_actual_error':bool(np.linalg.norm(delta)<=np.linalg.norm(decoded6[user]-u[user])),
                         'block_error':delta.tolist()})
        reconstruction = []
        for x in matrices:
            x = x.reshape(8,2,-1); parts=[]
            for layer in x:
                left,s,right=np.linalg.svd(layer,full_matrices=False)
                parts.append(q8(left[:,:1]*s[:1])@q8(right[:1]))
            reconstruction.append(np.stack(parts).ravel())
        delta=np.linalg.norm(blocks(np.concatenate(reconstruction+[shared])-original),axis=-1)
        pair_deltas[int(user)] = delta
        rows.append({'user':int(user),'representation':'per_layer_token_pair_rank1_q8',
                     'actual_error':float(np.linalg.norm(delta)),
                     'bound_ratio_to_total_epsilon':float(np.linalg.norm(delta)/epsilon),
                     'unimplemented_factor_payload_estimate_bits':8*(8*(2+768)+8*(2+512))+16*64+32+256})
    pair_gate = []
    for head in range(c['users']):
        remote = [v for user,v in pair_deltas.items() if user!=head]
        bound = float(np.linalg.norm(np.sum(remote,axis=0)))
        pair_gate.append({'head':head,'quantization_only_rms_bound':bound,
                          'necessary_error_condition_passes':bound<=epsilon})
    matched=[]
    for spectrum in spectra:
        eligible=[x for x in rows if x['user']==spectrum['user'] and x['representation']=='modality_matrix_svd_q8'
                  and x['every_block_error_no_greater_than_original_six_bit_bound']]
        best=min(eligible,key=lambda x:x['unimplemented_factor_payload_estimate_bits']) if eligible else None
        matched.append({'user':spectrum['user'],'six_bit_raw_payload_bits':spectrum['six_bit_payload_bits'],
                        'minimum_rank_matching_every_six_bit_block_bound':best['rank'] if best else None,
                        'optimistic_factor_payload_bits':best['unimplemented_factor_payload_estimate_bits'] if best else None,
                        'smaller_than_six_bit_raw':best['unimplemented_factor_payload_estimate_bits']<spectrum['six_bit_payload_bits'] if best else False})
    result={'at_kst':datetime.now().isoformat(timespec='seconds'),'epsilon':float(epsilon),
            'status':'cpu_rate_distortion_screen_only_no_codec_or_training_promotion',
            'spectra':spectra,'rows':rows,'pair_rank1_head_gates':pair_gate,'matched_six_bit_bound':matched,
            'source_sha256':expected,'updates_sha256':sha(p),'script_sha256':sha(__file__),
            'limits':['One saved legacy-round update, not GPU replay or training seeds.',
                      'Estimated factor payload is optimistic and does not define a packet or include control/SVD energy/time.',
                      'The pair gate rejects the current conservative Q<=epsilon formulation, not the original stochastic MSE problem: cross-component error cancellation is not modeled.',
                      'Matching every six-bit block bound is a sufficient conservative comparison gate; failing it does not prove every other aggregate-error allocation infeasible.',
                      'SVD truncation changes unlearning updates; no equal utility or forgetting claim.',
                      'No test data used, no core/queued source changed, no GPU job launched.',
                      'Failure of this specific factorization is not a theorem excluding all compression.']}
    out=root/'results/overnight_energy_v2/spectral_screen_v5';out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['저차원 프롬프트 압축 사전 점검', '',
           '기존 6비트 학습의 첫 두 요청 구간에서 절감이 작아 추가 구조 압축 가능성을 CPU에서 점검했다.',
           '대상은 기존 legacy 91라운드 고정 업데이트 하나이며 새로운 GPU 학습이나 무선 최적화 결과가 아니다.',
           f'원래 전체 RMS 예산: {epsilon:.12g}', '',
           '단순한 각 레이어 두 프롬프트의 rank-1/8비트 factor 표현:']
    for x in rows:
        if x['representation']=='per_layer_token_pair_rank1_q8':
            lines.append(f"client{x['user']}: 복원 오차 {x['actual_error']:.12g}, 전체 예산의 {x['bound_ratio_to_total_epsilon']:.4f}배")
    lines.append('이 표현은 '+('모든 head에서 양자화 오차 항만으로 기존 보수적 Q<=epsilon 예산을 초과해 제외했다. 다른 오차와의 상쇄를 허용하는 원래 MSE 문제의 불가능성을 증명한 것은 아니다.' if not any(x['necessary_error_condition_passes'] for x in pair_gate) else '일부 head의 보수적 오차 조건만 통과했다. 통신 가능성과 동등한 삭제 품질을 뜻하지 않는다.'))
    lines += ['', 'V/T 각각 16개 프롬프트 행 전체를 SVD하고 8비트 factor로 복원한 경우:']
    for x in matched:
        lines.append(f"client{x['user']}: 기존 6비트의 각 블록 오차 상한을 넘지 않는 최소 rank {x['minimum_rank_matching_every_six_bit_block_bound']}, 낙관적 payload {x['optimistic_factor_payload_bits']} bit, 기존 6비트 {x['six_bit_raw_payload_bits']} bit")
    lines += ['', '구현하지 않은 factor packet의 크기 추정이며 제어 전송·SVD 계산·수신 회로 비용은 미포함이다.',
              '각 6비트 블록 오차 상한을 모두 맞추는 조건은 보수적 비교 기준이다. 이 기준 미달이 다른 총 오차 배분의 불가능성을 뜻하지 않는다.',
              '기존 6비트와 같은 블록별 오차 상한을 유지할 때 더 작은 payload를 찾지 못하면 GPU 후보로 올리지 않는다.',
              '더 느슨한 압축 오차로 payload가 작아지는 점을 같은 품질의 에너지 절감으로 해석하지 않는다. 현재 큐와 burst_v3 프로토콜은 유지한다.']
    (out/'report_ko.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'epsilon':epsilon,'matched_six_bit_bound':matched,'pair_rank1_head_gates':pair_gate}))


if __name__=='__main__':
    main()
