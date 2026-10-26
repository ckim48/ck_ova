import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent
REVISION = "57c216476eefef5ab752ec549e440a49ae4ae5f3"
MODEL_FILES = [
    "config.json", "pytorch_model.bin", "preprocessor_config.json",
    "merges.txt", "vocab.json", "tokenizer_config.json", "special_tokens_map.json",
]


def download(url, path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with requests.get(url, stream=True, timeout=(20, 120)) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                handle.write(chunk)
    temporary.replace(path)
    print(f"Downloaded {path.name}: {path.stat().st_size / 1e6:.1f} MB", flush=True)


def main(model_only=False):
    model_directory = ROOT / "checkpoints/clip-vit-base-patch16"
    for filename in MODEL_FILES:
        download(f"https://huggingface.co/openai/clip-vit-base-patch16/resolve/{REVISION}/{filename}",
                 model_directory / filename)
    model_manifest = {
        "model": "openai/clip-vit-base-patch16", "revision": REVISION,
        "files_sha256": {
            filename: hashlib.sha256((model_directory / filename).read_bytes()).hexdigest()
            for filename in MODEL_FILES
        },
    }
    (model_directory / "manifest.json").write_text(json.dumps(model_manifest, indent=2) + "\n")
    if model_only:
        print("Model files and provenance ready.", flush=True)
        return
    archive = ROOT / "data/cifar-100-python.tar.gz"
    download("https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz", archive)
    digest = hashlib.md5()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != "eb9058c3a382ffc7106e4002c42a8d85":
        raise RuntimeError("CIFAR-100 archive checksum mismatch.")
    with tarfile.open(archive, "r:gz") as archive_file:
        for filename in ("train", "test", "meta"):
            member = archive_file.getmember(f"cifar-100-python/{filename}")
            if not member.isfile():
                raise RuntimeError("Expected a regular dataset file.")
            contents = archive_file.extractfile(member)
            destination = ROOT / "data/cifar-100-python" / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(contents.read())
    manifest = {
        "model": "openai/clip-vit-base-patch16", "revision": REVISION,
        "cifar100_md5": digest.hexdigest(),
        "model_files_sha256": {
            filename: hashlib.sha256((model_directory / filename).read_bytes()).hexdigest()
            for filename in MODEL_FILES
        },
    }
    (ROOT / "data/assets_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Real model and CIFAR-100 assets verified.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-only", action="store_true")
    main(parser.parse_args().model_only)
