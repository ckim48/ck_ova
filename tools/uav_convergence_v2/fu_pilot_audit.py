"""Audit all pilot outcomes, immutable inputs, raw archives and figure exports."""
import collections
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parents[2]
STUDIES = [
    ("fu_joint_optimization_v1_20261027", "", {"completed": 12}),
    ("fu_component_fusion_v2_20261027", "component_fusion", {"incomplete_no_feasible_radio_design": 4}),
    ("fu_component_existing_budget_v3_20261027", "component_existing_budget", {"completed": 4}),
]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_sources(mapping, cache):
    for name, digest in mapping.items():
        if name not in cache:
            cache[name] = sha(ROOT / name)
        assert cache[name] == digest, name


def main():
    cache, studies = {}, []
    for folder, gallery, expected in STUDIES:
        root = ROOT / "results" / folder
        dest = ROOT / "fu_joint_pilot" / gallery
        protocol = read(root / "protocol.json")
        for group in ["source_sha256", "input_sha256"]:
            check_sources(protocol[group], cache)
        outcomes, attempts, local_records = collections.Counter(), 0, 0
        for dataset in protocol["datasets"]:
            dd = root / "cases/baseline" / (dataset.lower() + "_seed0")
            baseline = ROOT / "results/learning_ablation_v1_20261027/cases/baseline" / dd.name
            for name in ["plan.json", "requests.json"]:
                assert read(dd / name) == read(baseline / name)
            for method in protocol["variants"]:
                arm = dd / method
                outcome = read(arm / "outcome.json")
                outcomes[outcome["status"]] += 1
                assert read(arm / "component_checks.json")["passed"]
                if outcome["status"] == "completed":
                    manifest = read(arm / "manifest.json")
                    verification = read(arm / "verification.json")
                    assert manifest["completed_FL_rounds"] == 200
                    assert verification["FL_rounds"] == 200 and verification["FU_steps"] == 225
                for path in sorted((arm / "work_attempts").glob("*.json")):
                    record = read(path)
                    cost = record["compute"]
                    for base, extra, total in zip(cost["base_device_macs"], cost["FU_control_scalar_ops"], cost["device_macs"]):
                        assert extra >= 0 and total == base + extra
                    for local in record["local"]:
                        if "retention_forget_dot_after" in local:
                            assert local["retention_forget_dot_after"] >= -1e-12
                        if folder.endswith("v3_20261027"):
                            assert "Existing guard protects composite U" in local["radio_component_definition"]
                    attempts += 1
                    local_records += len(record["local"])
        assert dict(outcomes) == expected
        assert read(root / "verification.json")["outcomes"] == expected
        assert read(root / "publication.json")["outcomes"] == expected
        archive = read(root / "raw_archive_manifest.json")
        path = root / archive["file"]
        assert sha(path) == archive["sha256"] and path.stat().st_size == archive["bytes"]
        with tarfile.open(path, "r:gz") as stream:
            members = stream.getmembers()
            assert len({m.name for m in members}) == len(members)
            assert {m.name for m in members} == set(archive["entries"])
            for member in members:
                name = PurePosixPath(member.name)
                assert member.isfile() and not name.is_absolute() and ".." not in name.parts
                metadata = archive["entries"][member.name]
                assert member.size == metadata["size_bytes"]
                with stream.extractfile(member) as data:
                    assert hashlib.file_digest(data, "sha256").hexdigest() == metadata["sha256"]
                assert sha(ROOT / member.name) == metadata["sha256"]
        manifest = read(dest / "data/manifest.json")
        check_sources(manifest["source_sha256"], cache)
        for figure in manifest["figures"]:
            precisions = set()
            for axis in figure["axes"]:
                for dim in ["x", "y"]:
                    settings = axis[dim]
                    ticks = settings["ticks"]
                    assert settings["limits"] == [ticks[0], ticks[-1]]
                    assert all(math.isclose(b-a, ticks[1]-ticks[0], abs_tol=1e-10) for a, b in zip(ticks, ticks[1:]))
                precisions.update(len(s.partition(".")[2]) for s in axis["y"]["labels"])
            assert len(precisions) == 1
            for export in figure["files"]:
                assert sha(dest / export["file"]) == export["sha256"]
            for marker in figure["status_markers"]:
                assert marker["metric_y"] is None and marker["axes_y_fraction"] > 1
        studies.append(dict(study=folder, outcomes=dict(outcomes), work_attempts=attempts,
                            local_FU_records=local_records, archive_files=len(members), figures=len(manifest["figures"])))
    result = dict(passed=True, studies=studies, verified_source_files=len(cache),
                  scope="Artifact integrity and accounting audit; not a statistical or deletion guarantee.",
                  audit_source_sha256=sha(Path(__file__)))
    (ROOT / "fu_joint_pilot/data/audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
