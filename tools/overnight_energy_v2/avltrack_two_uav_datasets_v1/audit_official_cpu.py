"""Read-only CPU audit of the pinned official AVLTrack checkpoint and language path."""
import hashlib, json, pathlib, types
import torch
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings

torch.set_num_threads(2)
root = pathlib.Path("/home/wnlab/Documents/Codex/2026-10-20/files-pasted-by-the-user-standalone/outputs/ova_CK")
source = root / "third_party/AVLTrack_readonly_v1"
checkpoint = pathlib.Path("/mnt/hdd/wnlab/avltrack_two_uav_datasets_v1/OSTrack_ep0297.pth.tar")
out = root / "results/overnight_energy_v2/avltrack_two_uav_datasets_v1"
out.mkdir(parents=True, exist_ok=True)
allowed_names = {
    "lib.train.admin.stats.StatValue", "lib.train.admin.stats.AverageMeter",
    "lib.train.admin.settings.Settings", "lib.train.admin.local.EnvironmentSettings"
}
unsafe = set(torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint))
assert unsafe <= allowed_names, sorted(unsafe - allowed_names)
# Inert metadata containers: no repository constructors or pickle code execution.
containers = [type(name.rsplit(".", 1)[1], (), {"__module__": name.rsplit(".", 1)[0]}) for name in sorted(unsafe)]
with torch.serialization.safe_globals(containers):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
state = saved["net"]
cfg = BertConfig(vocab_size=30522, hidden_size=768, num_hidden_layers=12,
                 num_attention_heads=12, intermediate_size=48,
                 max_position_embeddings=40, hidden_dropout_prob=.1, attention_probs_dropout_prob=.1)
emb = BertEmbeddings(cfg).eval()
prefix = "language_embeddings."
language_state = {k[len(prefix):]: v for k,v in state.items() if k.startswith(prefix)}
position_ids = language_state.pop("position_ids", None)
if position_ids is not None:
    assert torch.equal(position_ids, emb.position_ids), "Position buffer mismatch"
load = emb.load_state_dict(language_state, strict=True)
# Two synthetic token sequences; common CLS=101, different content.
ids = torch.tensor([[101, 1037, 2417, 2482, 102] + [0]*35,
                    [101, 1037, 2630, 2482, 102] + [0]*35])
with torch.no_grad():
    all_tokens = emb(ids)
    source_selected = all_tokens[:,0,:]
code = (source / "lib/models/ostrack/ostrack.py").read_text()
assert "language_embeddings_[:, 0, :]" in code
report = {
    "kind":"read_only_cpu_component_audit",
    "official_commit":"43ef098e70af64a999d7c0ec1e5348760e02bd36",
    "checkpoint_path":str(checkpoint), "checkpoint_bytes":checkpoint.stat().st_size,
    "checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "source_sha256":hashlib.sha256(code.encode()).hexdigest(),
    "checkpoint_top_keys":list(saved),
    "checkpoint_state_tensor_entries":len(state),
    "checkpoint_state_tensor_elements":sum(v.numel() for v in state.values() if torch.is_tensor(v)),
    "language_embedding_elements":sum(v.numel() for v in language_state.values()),
    "state_tensor_elements_by_prefix":{p:sum(v.numel() for k,v in state.items() if k.startswith(p) and torch.is_tensor(v)) for p in ("backbone.","box_head.","language_embeddings.")},
    "safe_metadata_containers":sorted(unsafe),
    "language_weights_strict_load":True,
    "legacy_position_buffer_verified":position_ids is not None,
    "selected_cls_max_abs_diff":float((source_selected[0]-source_selected[1]).abs().max()),
    "all_token_max_abs_diff":float((all_tokens[0]-all_tokens[1]).abs().max()),
    "eval_mode":True,
    "finding":"The current official forward path selects position 0 directly from token embeddings, before a contextual text encoder. With the same CLS token, changing other tokens does not change this language input.",
    "limits":"Component/path audit, not a tracking accuracy evaluation. No source changes, FL/FU training, or GPU work.",
    "training_launched":False
}
(out / "cpu_language_path_audit.json").write_text(json.dumps(report, indent=2))
print("AUDIT_START"+json.dumps(report)+"AUDIT_END")
