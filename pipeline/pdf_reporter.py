import datetime
from pathlib import Path
from weasyprint import HTML
from pipeline.law_resolver import load_laws, analyze, to_report_items

def generate_pdf(fir_text: str, out_path: str = "data/frs/report.pdf") -> str:
    """Generate PDF report directly from FIR text analysis."""
    laws = load_laws()
    analysis = analyze(fir_text, laws)
    offenses = to_report_items(analysis)

    context = {
        "case_id": f"FORENSAI-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "offenses": offenses,
        "version": {"app": "0.5.0", "laws": "2025.11.23"}
    }

    # Build simple inline HTML (no external template needed)
    html_str = f"""
    <html><body>
    <h1>Preliminary Law Analysis Report</h1>
    <p><strong>Case ID:</strong> {context['case_id']}</p>
    <p><strong>Generated:</strong> {context['timestamp']}</p>
    <h2>Offenses</h2>
    <ul>
    {''.join(f"<li>{o['offense']} – Sections: {', '.join(o['primary_sections'])} (Confidence {o['confidence']})</li>" for o in offenses)}
    </ul>
    </body></html>
    """

    Path("data/frs").mkdir(parents=True, exist_ok=True)
    HTML(string=html_str).write_pdf(out_path)
    return out_path