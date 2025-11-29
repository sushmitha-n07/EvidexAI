# pipeline/crime_pipeline.py

from typing import Dict, List
from vision.text_classifier import format_output as text_fallback
# If you have a vision classifier, import it here:
# from vision.crime_classifier_clip import classify_scene

class CrimeScene:
    def __init__(self, fir_text: str, image_path: str = None,
                 detected_objects: List[str] = None,
                 location: str = None, timestamp=None):
        self.fir_text = fir_text
        self.image_path = image_path
        self.detected_objects = detected_objects or []
        self.location = location
        self.timestamp = timestamp

class CrimePipeline:
    def __init__(self):
        pass

    def analyze_crime_scene(self, scene: CrimeScene) -> Dict:
        """
        Analyze a crime scene using vision (if available) and text fallback.
        Always returns a possible crime type (never 'Unknown').
        """
        vision_result = {}
        # Example if vision classifier is integrated:
        # if scene.image_path:
        #     vision_result = classify_scene([scene.image_path])

        # Text fallback using FIR + detected objects
        fallback_result = text_fallback(scene.fir_text, image_tags=scene.detected_objects)

        # Decision: prefer vision if it has a label, else fallback
        if vision_result.get("label"):
            final_label = vision_result["label"]
            final_display = f"Possible: {final_label}"
            final_conf = max(vision_result.get("confidence", 0.0),
                             fallback_result.get("confidence", 0.0))
            rationale = "Vision: " + vision_result.get("rationale", "")
            alternatives = fallback_result.get("alternatives", [])
        else:
            final_label = fallback_result.get("label")
            final_display = fallback_result.get("displayLabel") or f"Possible: {final_label}"
            final_conf = fallback_result.get("confidence", 0.0)
            rationale = fallback_result.get("rationale", "")
            alternatives = fallback_result.get("alternatives", [])

        # Ensure we never return "Unknown"
        crime_block = {
            "crimeLabel": final_label or final_display.replace("Possible: ", ""),
            "displayLabel": final_display,
            "confidence": final_conf,
            "alternatives": alternatives,
            "rationale": rationale
        }

        result = {
            "crime": crime_block,
            "timelineEvents": [],   # placeholder for timeline integration
            "riskLevel": "Uncertain" if final_conf < 0.3 else "Assessed"
        }
        return result
    #comment 
    