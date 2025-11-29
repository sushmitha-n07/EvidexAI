# vision/crime_classifier_clip.py
import os
import torch
import json
import clip
from PIL import Image

class CrimeClipClassifier:
    def __init__(self, labels_path="data/crime_labels.json", device=None):
        """Load CLIP model and crime labels."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        labels_path = os.path.join(base_dir, labels_path)

        if not os.path.exists(labels_path):
            raise FileNotFoundError(
                f"Labels file not found at: {labels_path}. "
                f"Expected location: vision/data/crime_labels.json"
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        with open(labels_path, "r", encoding="utf-8") as f:
            self.labels = json.load(f)

        if not isinstance(self.labels, list) or not self.labels:
            raise ValueError("Labels file must contain a non-empty JSON list of labels.")

    def classify_image(self, img: Image.Image, labels=None, top_k=3, simple=False):
        """Classify an image into crime categories using CLIP similarity."""
        labels = labels or self.labels

        image_input = self.preprocess(img).unsqueeze(0).to(self.device)
        text_inputs = torch.cat([clip.tokenize(lbl) for lbl in labels]).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        values, indices = similarity[0].topk(top_k)
        preds = [(labels[idx], float(values[i])) for i, idx in enumerate(indices)]

        if simple:
            return preds
        return {
            "label": preds[0][0],
            "confidence": preds[0][1],
            "alternatives": [lbl for lbl, _ in preds[1:]],
            "rationale": f"CLIP similarity scores: {preds}"
        }