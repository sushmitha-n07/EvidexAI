import clip
import torch

model, preprocess = clip.load("ViT-B/32")
print("CLIP model loaded successfully")