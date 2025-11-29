# vision/detector.py

import os
import json
from pathlib import Path
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from torchvision.ops import nms
from PIL import Image

# Cache directory for detection results
CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Crime-relevant objects only (reduces false positives)
CRIME_OBJECTS = {
    "knife", "scissors", "bat", "bottle", "gun", "pistol", "rifle",
    "chair", "table", "bed", "couch", "phone", "laptop", "bag",
    "backpack", "suitcase", "box", "car", "truck", "bicycle"
}

# High-risk objects that need extra attention
HIGH_RISK_OBJECTS = {"knife", "gun", "pistol", "rifle", "bat", "scissors"}

class SceneDetector:
    """Enhanced YOLOS detector with caching and false-positive reduction"""

    def __init__(self, model_choice="yolos-small"):
        self.model_choice = model_choice

        # Try to load local model first
        model_path = f"models/{model_choice}"
        if os.path.exists(model_path):
            self.processor = AutoImageProcessor.from_pretrained(model_path)
            self.model = AutoModelForObjectDetection.from_pretrained(model_path)
        else:
            model_name = "hustvl/yolos-small" if model_choice == "yolos-small" else model_choice
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForObjectDetection.from_pretrained(model_name)

            # Save for future use
            os.makedirs(model_path, exist_ok=True)
            self.processor.save_pretrained(model_path)
            self.model.save_pretrained(model_path)

    def analyze_scene_image(self, image_path: str) -> dict:
        """Analyze image with caching and false-positive reduction"""
        cache_file = os.path.join(CACHE_DIR, f"{Path(image_path).stem}_detection.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    cached_result = json.load(f)
                    for obj in cached_result.get("objects", []):
                        if "source_image" not in obj:
                            obj["source_image"] = Path(image_path).name
                    return cached_result
            except Exception:
                pass

        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = self.model(**inputs)

            target_sizes = torch.tensor([image.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs,
                threshold=0.75,
                target_sizes=target_sizes
            )[0]

            if len(results["boxes"]) > 0:
                keep = nms(results["boxes"], results["scores"], iou_threshold=0.4)
                results = {k: v[keep] for k, v in results.items()}

            detected_objects = []
            confidence_warnings = []

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                label_name = self.model.config.id2label[label.item()].lower()
                confidence = float(score)

                if label_name in CRIME_OBJECTS:
                    obj_data = {
                        "object": label_name,
                        "confidence": confidence,
                        "source_image": Path(image_path).name,
                        "detection_method": "yolos_ai",
                        "bbox": [float(x) for x in box.tolist()]
                    }

                    if confidence < 0.85:
                        obj_data["warning"] = "Possible false detection - verify manually"
                        confidence_warnings.append(f"{label_name} ({confidence:.2%})")

                    if label_name in HIGH_RISK_OBJECTS:
                        obj_data["risk_level"] = "HIGH"
                        obj_data["priority"] = "URGENT"

                    detected_objects.append(obj_data)

            suspicious_patterns = self._identify_suspicious_patterns(detected_objects)
            if confidence_warnings:
                suspicious_patterns.append(f"LOW CONFIDENCE DETECTIONS: {', '.join(confidence_warnings)}")

            result = {
                "objects": detected_objects,
                "suspicious_patterns": suspicious_patterns,
                "total_detections": len(detected_objects),
                "image_dimensions": list(image.size),
                "detection_quality": self._assess_detection_quality(detected_objects),
                "cache_timestamp": str(Path(image_path).stat().st_mtime),
                "analysis_device": "cuda" if torch.cuda.is_available() else "cpu"
            }

            with open(cache_file, "w") as f:
                json.dump(result, f, indent=2)

            return result

        except Exception as e:
            return {
                "objects": [{
                    "object": "error_fallback",
                    "confidence": 0.50,
                    "source_image": Path(image_path).name,
                    "warning": "Detection failed - manual review required",
                    "error": str(e)
                }],
                "suspicious_patterns": ["DETECTION ERROR - Manual analysis required"],
                "total_detections": 0,
                "error": str(e)
            }

    def _identify_suspicious_patterns(self, objects: list) -> list:
        """Identify suspicious patterns in detected objects"""
        patterns = []
        object_names = [obj["object"].lower() for obj in objects]
        high_conf_objects = [obj for obj in objects if obj["confidence"] > 0.85]
        low_conf_objects = [obj for obj in objects if obj["confidence"] < 0.85]

        weapons = [obj for obj in objects if obj["object"] in HIGH_RISK_OBJECTS]
        if weapons:
            high_conf_weapons = [w for w in weapons if w["confidence"] > 0.85]
            low_conf_weapons = [w for w in weapons if w["confidence"] < 0.85]

            if high_conf_weapons:
                patterns.append(f"HIGH CONFIDENCE WEAPONS: {', '.join([w['object'] for w in high_conf_weapons])}")
            if low_conf_weapons:
                patterns.append(f"POSSIBLE WEAPONS (verify): {', '.join([w['object'] for w in low_conf_weapons])}")

        if len(weapons) > 1:
            patterns.append("MULTIPLE POTENTIAL WEAPONS: Escalated threat level")

        containers = ['bag', 'backpack', 'suitcase', 'box']
        if any(c in object_names for c in containers) and weapons:
            patterns.append("CONCEALMENT RISK: Weapon + container combination")

        vehicles = ['car', 'truck', 'bicycle']
        if any(v in object_names for v in vehicles) and weapons:
            patterns.append("ESCAPE PREPARATION: Vehicle + weapon detected")

        if len(high_conf_objects) > 3:
            patterns.append(f"{len(high_conf_objects)} high-confidence detections")

        if len(low_conf_objects) > len(high_conf_objects):
            patterns.append(f"{len(low_conf_objects)} low-confidence detections need verification")

        return patterns

    def _assess_detection_quality(self, objects: list) -> str:
        """Assess overall detection quality"""
        if not objects:
            return "no_detections"

        avg_confidence = sum(obj["confidence"] for obj in objects) / len(objects)
        high_conf_count = sum(1 for obj in objects if obj["confidence"] > 0.85)
        high_conf_ratio = high_conf_count / len(objects)

        if avg_confidence > 0.9 and high_conf_ratio > 0.8:
            return "excellent"
        elif avg_confidence > 0.8 and high_conf_ratio > 0.6:
            return "good"
        elif avg_confidence > 0.7:
            return "fair"
        else:
            return "poor_needs_verification"

    def clear_cache(self):
        """Clear detection cache"""
        import shutil
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
            os.makedirs(CACHE_DIR, exist_ok=True)

# Utility functions
def clear_detection_cache():
    """Clear all cached detections"""
    detector = SceneDetector()
    detector.clear_cache()

def get_cache_stats() -> dict:
    """Get cache statistics"""
    if not os.path.exists(CACHE_DIR):
        return {"cached_files": 0, "cache_size": "0 MB"}

    cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".json")]
    cache_size = sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in cache_files)

    return {
        "cached_files": len(cache_files),
        "cache_size": f"{cache_size / (1024 * 1024):.2f} MB"
    }
