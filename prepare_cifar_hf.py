import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet
import requests
from PIL import Image

from prepare_real_assets import download


ROOT = Path(__file__).resolve().parent


def main():
    response = requests.get("https://huggingface.co/api/datasets/uoft-cs/cifar100", timeout=30)
    response.raise_for_status()
    revision = response.json()["sha"]
    destination = ROOT / "data/cifar-100-python"
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split in ("train", "test"):
        filename = f"{split}-00000-of-00001.parquet"
        path = ROOT / "data/hf-cifar100" / filename
        download(f"https://huggingface.co/datasets/uoft-cs/cifar100/resolve/{revision}/cifar100/{filename}", path)
        table = parquet.read_table(path)
        metadata = json.loads(table.schema.metadata[b"huggingface"])
        classes = metadata["info"]["features"]["fine_label"]["names"]
        expected = 50000 if split == "train" else 10000
        if table.num_rows != expected or len(classes) != 100:
            raise RuntimeError("Unexpected CIFAR-100 schema.")
        images = np.stack([
            np.asarray(Image.open(io.BytesIO(item["bytes"])).convert("RGB"))
            for item in table["img"].to_pylist()
        ])
        labels = np.array(table["fine_label"].to_pylist())
        if images.shape != (expected, 32, 32, 3):
            raise RuntimeError("Unexpected CIFAR-100 image dimensions.")
        np.savez_compressed(destination / f"{split}.npz", images=images, labels=labels)
        (destination / "classes.json").write_text(json.dumps(classes) + "\n")
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"Prepared {split}: {len(labels)} real CIFAR-100 images", flush=True)
    manifest = {
        "dataset": "uoft-cs/cifar100", "revision": revision,
        "source": "Hugging Face distribution of CIFAR-100",
        "sha256": hashes, "original_toronto_archive_md5_verified": False,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
