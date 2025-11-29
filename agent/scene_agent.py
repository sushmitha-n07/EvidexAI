import re
from typing import Dict, List, Tuple
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Intent classifier import with fallback
try:
    from nlp.intent_classifier import IntentClassifier
except ImportError:
    class IntentClassifier:
        """Fallback intent classifier with simple keyword rules."""
        def classify(self, text: str) -> Dict[str, float]:
            text_lower = text.lower()
            if any(word in text_lower for word in ['murder', 'killed', 'dead']):
                return {"predicted_intent": "Homicide", "confidence": 0.7}
            elif any(word in text_lower for word in ['domestic', 'spouse', 'husband', 'wife']):
                return {"predicted_intent": "Domestic Violence", "confidence": 0.7}
            elif any(word in text_lower for word in ['robbed', 'stolen', 'theft']):
                return {"predicted_intent": "Robbery", "confidence": 0.7}
            else:
                return {"predicted_intent": "Unknown", "confidence": 0.5}

# RAG retriever import with fallback
try:
    from nlp.rag_retriever import find_similar_firs, load_fir_corpus
except ImportError:
    def find_similar_firs(text: str, corpus: List[str], top_k: int = 3) -> List[str]:
        return ["No similar cases available"]

    def load_fir_corpus(path: str) -> List[str]:
        return []

class SceneAgent:
    """Crime scene analysis agent for FIR text and visual evidence."""

    def __init__(self):
        # Initialize intent classifier
        try:
            self.intent_classifier = IntentClassifier()
        except Exception:
            self.intent_classifier = None

        # Load FIR corpus
        try:
            self.corpus = load_fir_corpus("data/firs")
        except Exception:
            self.corpus = []

        # Crime patterns for extraction
        self.crime_patterns = {
            'weapons': ['knife', 'gun', 'weapon', 'bat', 'stick'],
            'violence': ['fight', 'hit', 'beat', 'attack', 'assault'],
            'locations': ['kitchen', 'bedroom', 'living room', 'bathroom'],
            'evidence': ['blood', 'fingerprint', 'dna', 'witness']
        }

    def analyze(self, fir_text: str, detected_objects: List[str] = None) -> Tuple[Dict, List[str], str]:
        """
        Analyze FIR text and detected visual objects.
        Returns structured info, similar cases, and reasoning.
        """
        if detected_objects is None:
            detected_objects = []

        crime_info = self._classify_crime_type(fir_text)
        extracted_info = self._extract_key_info(fir_text, detected_objects)
        similar_cases = self._find_similar_cases(fir_text)
        reasoning = self._generate_reasoning(crime_info, extracted_info, similar_cases, detected_objects)

        full_info = {
            **crime_info,
            **extracted_info,
            'detected_objects': detected_objects
        }
        return full_info, similar_cases, reasoning

    def _classify_crime_type(self, text: str) -> Dict:
        """Classify the crime type using intent classifier or fallback."""
        if self.intent_classifier:
            try:
                result = self.intent_classifier.classify(text)
                return {
                    'crime_type': result.get('predicted_intent', 'Unknown'),
                    'confidence': result.get('confidence', 0.5)
                }
            except Exception:
                pass
        return {'crime_type': 'Unknown', 'confidence': 0.5}

    def _extract_key_info(self, text: str, objects: List[str]) -> Dict:
        """Extract key evidence and risk factors from FIR text and objects."""
        info = {'key_phrases': [], 'potential_evidence': [], 'risk_factors': []}
        text_lower = text.lower()

        for weapon in self.crime_patterns['weapons']:
            if weapon in text_lower:
                info['potential_evidence'].append(weapon)
                info['risk_factors'].append(f'Weapon mentioned: {weapon}')

        for violence in self.crime_patterns['violence']:
            if violence in text_lower:
                info['risk_factors'].append(f'Violence indicator: {violence}')

        info['potential_evidence'].extend(objects)
        return info

    def _find_similar_cases(self, text: str) -> List[str]:
        """Find similar historical cases using RAG retriever or fallback."""
        if self.corpus:
            try:
                return find_similar_firs(text, self.corpus, top_k=3)
            except Exception:
                pass
        return ["No similar cases available"]

    def _generate_reasoning(self, crime_info: Dict, extracted_info: Dict,
                            similar_cases: List[str], objects: List[str]) -> str:
        """Generate reasoning narrative combining classification, evidence, risks, and cases."""
        parts = [
            f"Crime Classification: {crime_info['crime_type']}",
            f"Confidence: {crime_info['confidence']:.2%}"
        ]
        if extracted_info['potential_evidence']:
            parts.append(f"Evidence identified: {', '.join(extracted_info['potential_evidence'])}")
        if extracted_info['risk_factors']:
            parts.append(f"Risk factors: {len(extracted_info['risk_factors'])} identified")
        if objects:
            parts.append(f"Visual evidence supports analysis: {len(objects)} objects detected")
        if similar_cases and "No similar cases" not in similar_cases[0]:
            parts.append(f"Found {len(similar_cases)} similar historical cases")
        return "\n".join(parts)