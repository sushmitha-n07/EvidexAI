import re
from typing import Dict, List

class IntentClassifier:
    """Ultra-lightweight intent classifier using keyword patterns (no downloads)."""

    def __init__(self):
        self.crime_patterns = {
            "Homicide": ["murder", "killed", "dead body", "corpse", "fatal", "death",
                         "deceased", "shot dead", "stabbed", "strangled", "beaten to death",
                         "homicide", "slain", "victim found dead"],
            "Domestic Violence": ["domestic", "spouse", "husband", "wife", "family", "home violence",
                                  "partner", "boyfriend", "girlfriend", "marriage", "relationship",
                                  "family dispute", "marital", "domestic abuse", "restraining order"],
            "Robbery": ["robbed", "robbery", "stolen", "theft", "burglar", "break-in",
                        "loot", "cash", "money", "valuables", "store", "bank",
                        "heist", "armed robbery", "burglary", "shoplifting"],
            "Sexual Assault": ["rape", "sexual assault", "molestation", "harassment",
                               "inappropriate touching", "molest", "sexual abuse"],
            "Drug Offense": ["drugs", "narcotics", "cocaine", "heroin", "marijuana",
                             "substance", "dealer", "trafficking", "possession",
                             "overdose", "meth", "cannabis"],
            "Suicide": ["suicide", "self-harm", "hanging", "overdose", "jumped",
                        "pills", "suicide note", "depression", "self-inflicted"],
            "Accidental Death": ["accident", "accidental", "mishap", "fell", "slipped",
                                 "unintentional", "mistake", "car accident", "drowning"],
            "Missing Person / Kidnapping": ["missing", "disappeared", "vanished", "not found",
                                            "whereabouts unknown", "last seen", "abducted", "kidnapped"],
            "Fraud": ["fraud", "scam", "embezzlement", "forgery", "identity theft",
                      "credit card fraud", "ponzi scheme", "fake", "counterfeit"],
            "Assault": ["assault", "battery", "fight", "attacked", "beaten up",
                        "physical altercation", "punched", "kicked", "hit"],
            "Cybercrime": ["hacking", "phishing", "cyber", "online scam", "data breach",
                           "unauthorized access", "malware", "ransomware", "digital fraud"],
            "Terrorism": ["terror", "bomb", "explosive", "attack", "militant",
                          "terrorist group", "blast", "IED", "extremist"],
            "Human Trafficking": ["trafficking", "illegal confinement", "forced labor",
                                  "exploitation", "smuggling", "sold", "abuse network"]
        }

    def classify(self, text: str) -> Dict:
        """Classify FIR text using keyword patterns."""
        text_lower = text.lower()

        scores = {}
        best_score = 0
        best_crime = None
        best_keywords = []

        for crime_type, keywords in self.crime_patterns.items():
            matched_keywords = [kw for kw in keywords if kw in text_lower]
            score = len(matched_keywords)
            confidence = score / max(1, len(text_lower.split()))
            scores[crime_type] = confidence

            if confidence > best_score:
                best_score = confidence
                best_crime = crime_type
                best_keywords = matched_keywords

        # Always return a crime type, even if confidence is very low
        return {
            "predicted_intent": best_crime if best_crime else "No clear match",
            "confidence": best_score,
            "matched_keywords": best_keywords,
            "all_scores": scores,
            "method": "pattern_matching",
            "note": "Low confidence — possible crime type" if best_score < 0.05 else "Confident match"
        }

# Global instance
intent_classifier = IntentClassifier()

def classify_intent(text: str) -> str:
    """Quick classification function."""
    result = intent_classifier.classify(text)
    return f"{result['predicted_intent']} (confidence {result['confidence']:.2f})"