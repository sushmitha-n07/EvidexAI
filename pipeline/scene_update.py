class SceneUpdate:
    def __init__(self, case_id, location, fir_text=None, timeline=None, evidence=None):
        self.case_id = case_id
        self.location = location
        self.fir_text = fir_text or ""
        self.timeline = timeline or []
        self.evidence = evidence or []

    def analysis_scene(self):
        findings = []
        text = self.fir_text.lower()

        if "forced entry" in text:
            findings.append("Possible forced entry detected.")
        if "blood" in text:
            findings.append("Blood evidence mentioned.")
        if "argument" in text or "struggle" in text:
            findings.append("Witnesses reported an argument before incident.")

        return {
            "case_id": self.case_id,
            "location": self.location,
            "fir_summary": self.fir_text[:300],
            "findings": findings or ["No findings"],
            "timeline": self.timeline,
            "evidence_count": len(self.evidence),
            "status": "ok"
        }