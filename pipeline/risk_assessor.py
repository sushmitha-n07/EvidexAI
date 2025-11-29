# pipeline/risk_assessor.py

from typing import Dict, List, Optional, Set

def risk_score(
    evidence_tags: Set[str],
    entities: Optional[List] = None,
    detections: Optional[List[Dict]] = None,
    patterns: Optional[List] = None
) -> Dict:
    """Calculate risk score from evidence tags, entities, detections, and CKG patterns."""
    score = 0
    correlations = []

    violent = {"gun", "knife", "blood", "explosion"}
    public = {"crowd", "street", "school", "transport"}

    if any(t in evidence_tags for t in violent):
        score += 2
        correlations.append("Violent tags detected (e.g., gun, knife)")
    if any(t in evidence_tags for t in public):
        score += 1
        correlations.append("Public location tags detected (e.g., street, school)")

    if entities:
        for ent_text, ent_label in entities:
            if ent_label == "PERSON" and "suspect" in ent_text.lower():
                score += 1
                correlations.append(f"Entity '{ent_text}' indicates suspect involvement")
            elif ent_label == "GPE" and any(tag in public for tag in evidence_tags):
                score += 0.5
                correlations.append(f"Location '{ent_text}' matches public tags")

    if detections:
        for det in detections:
            label = det.get("label", "").lower()
            if label in violent:
                score += 1.5
                correlations.append(f"Detected '{label}' increases violence risk")

    if patterns:
        for pattern in patterns:
            if "suspect" in str(pattern).lower() and "weapon" in str(pattern).lower():
                score += 1
                correlations.append(f"Pattern '{pattern}' shows suspect-weapon link")

    level = "high" if score >= 3 else ("medium" if score >= 2 else "low")
    confidence = min(1.0, 0.6 + (len(correlations) * 0.1))

    return {
        "score": score,
        "level": level,
        "correlations": correlations,
        "confidence": confidence,
        "audit_trail": {
            "source_file": "pipeline/risk_assessor.py",
            "evidence_links": correlations,
            "confidence": confidence,
            "bias_note": "Risk assessment is AI-assisted; consult experts for real-world application."
        }
    }

def assess_overall_risk(risk_result: Dict, law_resolution: Optional[Dict] = None) -> Dict:
    """Holistic risk assessment integrating law resolution (e.g., homicide boost)."""
    overall_score = risk_result["score"]
    overall_correlations = risk_result["correlations"][:]

    if law_resolution and "Homicide" in str(law_resolution.get("laws", [])):
        overall_score += 1
        overall_correlations.append("Law resolution indicates homicide")

    overall_level = "high" if overall_score >= 4 else ("medium" if overall_score >= 2.5 else "low")

    return {
        "overall_score": overall_score,
        "overall_level": overall_level,
        "overall_correlations": overall_correlations,
        "integrated_audit": {
            "sources": ["risk_assessor.py", "law_resolver.py"],
            "confidence": min(1.0, risk_result["confidence"] * 0.9),
            "bias_note": "Integrated assessment combines risk and law data; verify with experts."
        }
    }

def get_bias_report() -> Dict:
    """Report on biases and limitations in risk assessment."""
    return {
        "model_name": "Enhanced Risk Assessor",
        "training_data": "Predefined tags + correlations (NER/CV/CKG)",
        "known_biases": [
            "Tag-based scoring may miss context (e.g., 'knife' in kitchen vs. crime).",
            "Over-reliance on violent/public tags.",
            "Assumes suspect involvement always raises risk.",
            "CKG patterns depend on graph completeness."
        ],
        "recommendations": [
            "Use as preliminary tool; combine with human expertise.",
            "Expand tag sets and validate with real case data.",
            "Audit scores regularly for fairness.",
            "Include oversight in high-stakes cases."
        ],
        "ethical_note": (
            "This tool aids investigations but is not predictive. "
            "Avoid misuse that could lead to unjust profiling."
        ),
        "xai_features": "Includes correlations, confidence scores, and audit trails."
    }