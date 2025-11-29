# pipeline/law_resolver.py

import os
import json
import re
import datetime
from pathlib import Path
from typing import List, Dict, Optional

# Path to law_mapping.json
law_path = os.path.join(os.path.dirname(__file__), 'data', 'law_mapping.json')

# ---------- LOADING LAW DEFINITIONS ----------

def load_laws() -> Dict[str, List[Dict[str, str]]]:
    """
    Load law definitions from law_mapping.json.
    Returns a dictionary mapping intent labels to IPC sections.
    """
    if not os.path.exists(law_path):
        raise FileNotFoundError(f"law_mapping.json NOT FOUND at: {law_path}")
    
    with open(law_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data.get("law_definitions", {})

def laws_for_label(label: str) -> List[str]:
    """
    Map predicted crime label to applicable IPC sections using keyword_to_intent.
    Returns a list of formatted IPC section strings.
    """
    if not os.path.exists(law_path):
        raise FileNotFoundError(f"law_mapping.json NOT FOUND at: {law_path}")
    
    with open(law_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Map label to intent
    intent = data.get("keyword_to_intent", {}).get(label.lower())
    if not intent:
        return ["No direct IPC section found"]
    
    # Retrieve IPC sections
    sections = data.get("law_definitions", {}).get(intent, [])
    if not sections:
        return ["No direct IPC section found"]
    
    return [f"{sec['code']} – {sec['section']}" for sec in sections]


# ---------- MATCHING PIPELINE ----------

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())

def keyword_hits(text: str, keywords: List[str]) -> int:
    return sum(1 for k in keywords if (k in text if " " in k else re.search(rf"\b{k}\b", text)))

def element_hits(text: str, elements: List[str]) -> int:
    return sum(1 for e in elements if e and e.split()[0] in text)

def score_section(text: str, law: Dict) -> float:
    return 0.4 * keyword_hits(text, law.get("keywords", [])) + 0.6 * element_hits(text, law.get("elements", []))

def analyze(text: str, laws: Dict) -> List[Dict]:
    """Analyze FIR text against law definitions and return best matches."""
    t = normalize(text)
    results = []
    for group, sections in laws.items():
        group_best = []
        group_score = 0.0
        for s in sections:
            sc = score_section(t, s)
            if sc > 0:
                group_best.append({**s, "score": sc})
                group_score += sc
        if group_best:
            group_best.sort(key=lambda x: x["score"], reverse=True)
            confidence = min(0.99, 1 - (1 / (1 + group_score)))
            results.append({
                "group": group,
                "primary_sections": [f'{x["code"]} §{x["section"]}' for x in group_best[:2]],
                "confidence": round(confidence, 3),
                "top_matches": group_best[:3],
                "evidence_hints": evidence_hints(group),
                "notes": group_notes(group)
            })
    return sorted(results, key=lambda x: x["confidence"], reverse=True)

def evidence_hints(group: str) -> List[str]:
    return {
        "burglary": ["CCTV footage", "Tool marks", "Inventory of stolen items"],
        "robbery": ["Victim statement", "Weapon description"],
        "arson": ["Fire report", "Accelerant traces"],
        "fraud": ["Transaction logs", "Fake documents"]
    }.get(group, [])

def group_notes(group: str) -> str:
    return {
        "burglary": "Use IPC §457 and §380 together if theft occurred during house-breaking.",
        "robbery": "Ensure fear or hurt was immediate to qualify as robbery."
    }.get(group, "")

def to_report_items(analysis: List[Dict]) -> List[Dict]:
    """Convert analysis results into structured report items."""
    items = []
    for r in analysis:
        items.append({
            "offense": r["group"].capitalize(),
            "primary_sections": r["primary_sections"],
            "confidence": r["confidence"],
            "matched_elements": list({e for m in r["top_matches"] for e in m.get("elements", [])}),
            "punishments": [{"section": f'{m["code"]} §{m["section"]}', "punishment": m.get("punishment","")} for m in r["top_matches"]],
            "flags": [{"section": f'{m["code"]} §{m["section"]}', "bailable": m.get("bailable"), "cognizable": m.get("cognizable")} for m in r["top_matches"]],
            "evidence_hints": r["evidence_hints"],
            "notes": r["notes"]
        })
    return items

# ---------- MULTIMODAL CORRELATION ----------

def resolve_laws(
    texts: List[str],
    entities: Optional[Dict] = None,
    detections: Optional[List[Dict]] = None,
    patterns: Optional[List] = None
) -> Dict:
    """Resolve laws using text, NER entities, CV detections, and patterns."""
    
    laws = []
    correlations = []

    for text in texts:
        t = text.lower()
        if "murder" in t or "homicide" in t:
            laws.extend(laws_for_label("homicide"))
            correlations.append("Keyword 'murder/homicide' detected in text")
        if "robbery" in t:
            laws.extend(laws_for_label("robbery"))
            correlations.append("Keyword 'robbery' detected in text")
        if "assault" in t or "violence" in t:
            laws.extend(laws_for_label("assault"))
            correlations.append("Keyword 'assault/violence' detected in text")

    if entities:
        for ent_text, ent_label in entities.get("entities", []):
            if ent_label == "PERSON" and "suspect" in ent_text.lower():
                laws.extend(laws_for_label("homicide"))
                correlations.append(f"Entity '{ent_text}' linked to homicide")
            if ent_label == "GPE" and "crime scene" in texts[0].lower():
                correlations.append(f"Entity location '{ent_text}' tied to crime scene")

    if detections:
        for det in detections:
            label = det.get("label", "").lower()
            if label in ["gun", "knife", "weapon"]:
                laws.extend(laws_for_label("homicide"))
                correlations.append(f"Detected '{label}' weapon – linked to homicide laws")

    if patterns:
        for pattern in patterns:
            if "suspect" in str(pattern).lower() and "weapon" in str(pattern).lower():
                laws.extend(laws_for_label("homicide"))
                correlations.append(f"Pattern {pattern} reinforces suspect-weapon link")

    laws = list(set(laws)) or ["No direct IPC section found"]
    confidence = min(1.0, 0.5 + len(correlations) * 0.1)

    audit = {
        "evidence_links": correlations,
        "confidence": confidence,
        "resolution_steps": [
            "Keyword detection in FIR text",
            "NER entity correlation",
            "Computer vision evidence integration",
            "CKG suspect-weapon pattern reasoning"
        ],
        "bias_note": (
            "Law resolution is AI-assisted. Based on predefined mappings and evidence correlations. "
            "Not a substitute for legal interpretation. Verify with legal authorities."
        )
    }

    return {
        "laws": laws,
        "correlations": correlations,
        "confidence": confidence,
        "audit_trail": audit
    }

# ---------- CKG & BIAS REPORTING ----------

def correlate_with_ckg(entities: List, detections: List, patterns: List) -> Dict:
    hidden_links = []
    if entities and detections:
        for ent in entities:
            for det in detections:
                if "suspect" in ent[0].lower() and det["label"].lower() in ["knife", "gun"]:
                    hidden_links.append(f"Suspect {ent[0]} linked to weapon '{det['label']}' via CKG")
    return {
        "hidden_links": hidden_links,
        "audit_trail": {
            "source": "CKG",
            "method": "Entity-weapon link reasoning",
            "confidence": 0.85
        }
    }
