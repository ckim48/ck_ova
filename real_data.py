import json
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPTokenizer


class CIFAR100:
    def __init__(self, root, checkpoint):
        self.root = Path(root)
        if (self.root / "train.npz").exists():
            with np.load(self.root / "train.npz") as train:
                self.train_images, self.train_labels = train["images"], train["labels"]
            with np.load(self.root / "test.npz") as test:
                self.test_images, self.test_labels = test["images"], test["labels"]
            self.classes = [name.replace("_", " ") for name in json.loads(
                (self.root / "classes.json").read_text()
            )]
        else:
            with (self.root / "train").open("rb") as handle:
                train = pickle.load(handle, encoding="bytes")
            with (self.root / "test").open("rb") as handle:
                test = pickle.load(handle, encoding="bytes")
            with (self.root / "meta").open("rb") as handle:
                meta = pickle.load(handle, encoding="bytes")
            self.train_images = train[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            self.train_labels = np.array(train[b"fine_labels"])
            self.test_images = test[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            self.test_labels = np.array(test[b"fine_labels"])
            self.classes = [name.decode().replace("_", " ") for name in meta[b"fine_label_names"]]
        self.processor = CLIPImageProcessor.from_pretrained(checkpoint, local_files_only=True)
        self.tokenizer = CLIPTokenizer.from_pretrained(checkpoint, local_files_only=True)

    def text_inputs(self, device, prompt_tokens=2):
        encoded = self.tokenizer(
            [f"a photo of a {name}." for name in self.classes], return_tensors="pt",
            padding=True, truncation=True, max_length=77 - prompt_tokens,
        )
        return {key: value.to(device) for key, value in encoded.items()
                if key in ("input_ids", "attention_mask")}

    def batch(self, indices, device, split="train"):
        images = self.train_images if split == "train" else self.test_images
        labels = self.train_labels if split == "train" else self.test_labels
        selected = [Image.fromarray(images[index]) for index in indices]
        pixels = self.processor(images=selected, return_tensors="pt")["pixel_values"].to(device)
        targets = torch.tensor(labels[indices], device=device, dtype=torch.long)
        return pixels, targets

    def validation_split(self, seed, per_class=50):
        generator = np.random.default_rng(seed)
        train_indices, validation_indices = [], []
        for label in range(100):
            indices = generator.permutation(np.flatnonzero(self.train_labels == label))
            validation_indices.extend(indices[:per_class])
            train_indices.extend(indices[per_class:])
        return np.array(train_indices), np.array(validation_indices)
