"""Exploratory retrieval-space FU objectives; all settings and outcomes retained.

The existing retain-only proxy20 is the only forgetting teacher. Reference200
and validation predictions are never gradient inputs. This tests adaptations,
not faithful reproductions of CLIPErase, CLIP-KD or any unlearning guarantee.
"""
from pathlib import Path
import argparse
import inspect
import linecache
import hashlib
import os
import subprocess
import sys
import textwrap
import time
import math
import collections

import numpy as np
import fu_joint_optimization as parent
from fu_joint_optimization import ROOT, SOURCE, DATASETS, PROPOSED, read, write, sha, link

OUT = ROOT / "results/fu_retrieval_alignment_v1_20261027"
PREVIOUS = parent.OUT
VARIANTS = {
    "fu_retrieval_cm": dict(label="R-KL", proxy=False, relation=True, retention=1.),
    **{f"fu_proxy_{cm}_rm{tag}": dict(label=f"P-{'R' if cm=='rank' else 'E'}{tag}", proxy=True,
                                     relation=cm=="rank", retention=value)
       for cm in ["embed", "rank"] for tag, value in [("0", 0.), ("01", .1), ("1", 1.)]},
}
for config in VARIANTS.values():
    config.update(adaptive=False, request_lr=False, ideal=False, fusion=False)


def retrieval_kl(images, texts, old_images, old_texts, scale):
    import torch.nn.functional as F
    logits = scale.exp() * images @ texts.T
    target = scale.detach().exp() * old_images.detach() @ old_texts.detach().T
    return .5 * (F.kl_div(F.log_softmax(logits, 1), F.softmax(target, 1), reduction="batchmean") +
                 F.kl_div(F.log_softmax(logits.T, 1), F.softmax(target.T, 1), reduction="batchmean"))


def gradient(base, variant):
    config = VARIANTS[variant]
    def request_gradient(engine, theta, fids, rids, step, teacher):
        import torch
        fcids = [c for i in fids for c in engine.data.caption_ids[i]]
        owners = [j for j, i in enumerate(fids) for _ in engine.data.caption_ids[i]]
        variable = theta.detach().requires_grad_(True)
        im, tx, scale = engine.features(variable, fids, fcids)
        fm = base.clip.forget_loss(im, tx, owners)
        extra = 0
        if config["proxy"]:
            assert engine.retrieval_proxy is not None
            old = engine.retrieval_proxy.features(fids, fcids)
            forgetting = retrieval_kl(im, tx, *old, scale)
            # Student/teacher logits plus both student feature derivatives,
            # with conservative scalar overhead at one MAC-equivalent each.
            extra += len(fids) * len(fcids) * (4 * im.shape[1] + 32)
        else:
            forgetting = fm
        gf = torch.autograd.grad(forgetting, variable)[0].detach(); gf[-1] = 0
        rcids = engine.data.training_caption_ids(rids, 21 + step)
        old = teacher.features(rids, rcids)
        variable = theta.detach().requires_grad_(True)
        ri, rt, scale = engine.features(variable, rids, rcids)
        rm, embedding_cm = base.clip.retain_losses(ri, rt, *old, scale)
        if config["relation"]:
            cm = retrieval_kl(ri, rt, *old, scale)
            extra += len(rids) * len(rcids) * (4 * ri.shape[1] + 32)
        else:
            cm = embedding_cm
        preservation = config["retention"] * rm + 3 * cm
        gp = torch.autograd.grad(preservation, variable)[0].detach(); gp[-1] = 0
        g = 3 * gf + gp
        assert torch.isfinite(g).all()
        parent.charge(engine, fids, extra)
        return g, dict(FU_variant=variant, forgetting_kind="proxy_retrieval_KL" if config["proxy"] else "positive_pair_cosine",
                       consistency_kind="retrieval_KL" if config["relation"] else "embedding_KL",
                       RM_weight=config["retention"], forgetting_weight=3., consistency_weight=3.,
                       FM=float(fm.detach()), RM=float(rm.detach()), CM=float(cm.detach()),
                       embedding_CM_diagnostic=float(embedding_cm.detach()),
                       forget_loss=float(forgetting.detach()), preservation_loss=float(preservation.detach()),
                       objective=float((3 * forgetting + preservation).detach()),
                       forget_gradient_norm=float(gf.norm()), weighted_forget_gradient_norm=float((3 * gf).norm()),
                       preserve_gradient_norm=float(gp.norm()), FU_gradient_norm=float(g.norm()),
                       forget_retain_gradient_cosine=float(torch.dot(gf, gp)/(gf.norm()*gp.norm()+1e-30)),
                       extra_retrieval_MAC_equivalents=extra, images=len(fids), forget_captions=len(fcids),
                       retained_images=rids, retained_captions=rcids,
                       teacher_policy="existing clean proxy20 for forgetting; request-start checkpoint for retention; reference200 evaluation-only")
    return request_gradient


def vectors(base, objective, variant):
    original = base.control.synchronized_vectors(objective)
    if objective != "cliperase":
        return original
    source = textwrap.dedent(inspect.getsource(original))
    assert source.count("stats=[]") == 1
    source = source.replace("stats=[]", "stats=[];engine.retrieval_proxy=proxy")
    filename = "/tmp/fu_retrieval_vectors_" + hashlib.sha256(source.encode()).hexdigest() + ".py"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = dict(original.__globals__, request_gradient=gradient(base, variant))
    exec(compile(source, filename, "exec"), namespace)
    return namespace[original.__name__]


def unit_checks():
    import torch
    import torch.nn.functional as F
    torch.manual_seed(811)
    im = F.normalize(torch.randn(4, 8, dtype=torch.double), dim=-1)
    tx = F.normalize(torch.randn(12, 8, dtype=torch.double), dim=-1)
    teacher_im = im.clone().requires_grad_(); teacher_tx = tx.clone().requires_grad_()
    student = (im + .05 * torch.randn_like(im)).requires_grad_()
    scale = torch.tensor(1., dtype=torch.double)
    value = retrieval_kl(student, tx, teacher_im, teacher_tx, scale)
    grad, ti, tt = torch.autograd.grad(value, (student, teacher_im, teacher_tx), allow_unused=True)
    assert ti is None and tt is None
    assert retrieval_kl(student-.001*grad, tx, im, tx, scale) < value
    assert abs(float(retrieval_kl(im, tx, im, tx, scale))) < 1e-12
    order = torch.randperm(12)
    assert torch.allclose(value, retrieval_kl(student, tx[order], im, tx[order], scale))
    return dict(passed=True, teacher_stop_gradient=True, identity_KL_zero=True,
                retrieval_KL_descent=True, caption_permutation_invariance=True)


def component_checks(base, engine, plan, requests, warm, dd):
    import torch
    variant = parent.ACTIVE
    req = requests[0]
    pool = sorted(set(plan["partitions"][req["user"]])-set(plan["forgotten_images"]))
    rids = np.random.default_rng(910000+req["burst"]*1000+req["user"]*10).choice(pool, min(16, len(pool)), replace=False).tolist()
    assert set(rids + req["images"]) <= set(plan["partitions"][req["user"]])
    proxy_theta = base.core.load_theta(dd/"proxy20/final.pt", engine.device)
    engine.retrieval_proxy = base.core.Teacher(engine, proxy_theta)
    teacher = base.core.Teacher(engine, warm)
    original, _ = base.clip.request_gradient(engine, warm, req["images"], rids, 0, teacher)
    candidate, meta = gradient(base, variant)(engine, warm, req["images"], rids, 0, teacher)
    assert candidate[-1] == 0 and torch.isfinite(candidate).all()
    # At request start the retained distillation gradients are both zero.
    if not VARIANTS[variant]["proxy"]:
        assert torch.allclose(original, candidate, rtol=1e-4, atol=1e-7)
    queue = base.RoundQueue(requests)
    old = base.control.synchronized_vectors("cliperase")(engine, warm, plan, queue, 0., 0, "fl", [], None, {}, True)[0]
    new = vectors(base, "cliperase", variant)(engine, warm, plan, queue, 0., 0, "fl", [], None, {}, True)[0]
    assert np.array_equal(old, new)
    write(dd/variant/"component_checks.json", dict(passed=True, training_only=True,
        FL_vectors_bit_identical=True, variant=variant, no_reference_or_validation_input=True,
        clean_proxy_sha256=sha(dd/"proxy20/final.pt"), original_gradient_norm=float(original.norm()),
        candidate_gradient_norm=float(candidate.norm()), metadata=meta))


def register():
    path = OUT/"protocol.json"
    if path.exists():
        result = read(path)
        for group in ["source_sha256", "input_sha256"]:
            for name, digest in result[group].items():
                assert sha(ROOT/name) == digest, name
        return result
    old = read(PREVIOUS/"protocol.json")
    sources = dict(old["source_sha256"])
    sources[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    inputs = dict(old["input_sha256"])
    for path in [ROOT/"fu_joint_pilot/data/comparison.json", ROOT/"fu_joint_pilot/component_existing_budget/data/comparison.json"]:
        inputs[str(path.relative_to(ROOT))] = sha(path)
    result = dict(registered_unix=time.time(), datasets=DATASETS, variants=VARIANTS, new_arms=14, seed=0,
        hypothesis="Align FU preservation with bidirectional retrieval distributions rather than embedding-coordinate softmax. Compare positive-cosine forgetting with the existing retain-only proxy's retrieval target. Test RM weights0,0.1,1 because extra supervised FU training can move away from a fixed-budget retrain model.",
        objective="3*forgetting + lambda_R*retained_contrastive_CE + 3*consistency. Same forget captions and retained minibatches, two backward passes, frozen temperature. Lambda_R is the preregistered small grid0,0.1,1; all settings reported, no test-based selection.",
        fixed="Same data/splits/seed0, warm20, proxy20, reporting-only reference200, 200FL,45x5FU, original paired minibatch seeds, FL/FU cosine LR, original1MHz, geometry, optimized heads/radio and hard0.1J cap. All failed/infeasible outcomes retained. No baseline/data/metric change.",
        teacher_availability="Existing common proxy20 excludes the planned forget cohort before mission, as in existing S-AC/AC-OF/S-OF. This inherits an offline known-cohort proxy assumption, not a newly causal solution to unknown future deletion requests. It is not reference200. Request-start teacher is used only on retained samples.",
        accounting="Same executed adapter MAC accounting, including newly required clean-proxy forwards. Additionally charge4*N_image*N_text*embedding_dim+32*N_image*N_text MAC-equivalents per retrieval KL term for logits, backward products and scalar work. Deliberately conservative; no extra payload; modeled not measured energy. Common proxy preparation was already present for all arms and is excluded equally from active-service graphs.",
        evaluation="Exploratory hyperparameter development on the existing validation split. Compare all original baselines on retained mR, mean/worst KL, MIA, energy and delay. Report any validation Pareto candidate as tentative, requiring paired new-seed/independent-split confirmation; do not overwrite final_chart or label a selected outcome as an unbiased test result.",
        references=[dict(title="CLIPErase (ACL2025)", url="https://aclanthology.org/2025.acl-long.1469/"),
                    dict(title="CLIP-KD (CVPR2024)", url="https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_CLIP-KD_An_Empirical_Study_of_CLIP_Model_Distillation_CVPR_2024_paper.pdf"),
                    dict(title="Behavioral Guarantees for Proxy-Based Unlearning (arXiv2026, preprint)", url="https://arxiv.org/abs/2605.10680")],
        attribution="The combined loss is our exploratory adaptation, not an exact reproduction or a transferred theoretical guarantee.",
        source_sha256=sources, input_sha256=inputs)
    write(OUT/"protocol.json", result)
    link(SOURCE/"prepared/airscape_seed0", OUT/"prepared/airscape_seed0")
    write(OUT/"unit_checks.json", unit_checks())
    return result


ORIGINAL_VERIFY = parent.verify


def verify():
    result = ORIGINAL_VERIFY()
    for dataset in DATASETS:
        for method, config in VARIANTS.items():
            arm = OUT/"cases/baseline"/(dataset.lower()+"_seed0")/method
            for path in (arm/"work_attempts").glob("*.json"):
                record = read(path)
                for row in record["local"]:
                    assert row["RM_weight"] == config["retention"]
                    assert math.isclose(row["objective"], 3*row["forget_loss"]+config["retention"]*row["RM"]+3*row["CM"], rel_tol=1e-5, abs_tol=1e-6)
                for a, b, total in zip(record["compute"]["base_device_macs"], record["compute"]["FU_control_scalar_ops"], record["compute"]["device_macs"]):
                    assert total == a+b and b >= 0
    write(OUT/"verification.json", result)
    return result


def state(dest=OUT, **kwargs):
    write(Path(dest)/"status.json", dict(unix=time.time(), pid=os.getpid(), **kwargs))


parent.OUT = OUT; parent.VARIANTS = VARIANTS
parent.gradient = gradient; parent.vectors = vectors
parent.component_checks = component_checks; parent.register = register
parent.state = state; parent.verify = verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["unit", "run", "variant", "verify"])
    parser.add_argument("--variant", choices=list(VARIANTS))
    args = parser.parse_args()
    if args.action == "unit":
        print(unit_checks()); return
    if args.action == "variant":
        parent.run_variant(args.variant); return
    if args.action == "verify":
        print(verify()); return
    register()
    for variant in VARIANTS:
        subprocess.run([sys.executable, str(Path(__file__)), "variant", "--variant", variant], cwd=ROOT, check=True)
    result = verify()
    write(OUT/"finished.json", dict(unix=time.time(), **result)); state(stage="completed", outcomes=result["outcomes"])
    print(result, flush=True)


if __name__ == "__main__":
    main()
