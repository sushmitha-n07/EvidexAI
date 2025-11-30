import streamlit as st
import sys, os, tempfile, datetime, io, json
from PIL import Image

# Add root folder to path so 'vision' can be found
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Core modules
from vision.crime_classifier_clip import CrimeClipClassifier
from vision.video_utils import sample_frames, majority_vote
from pipeline.law_resolver import laws_for_label
from scene_pipeline import ScenePipeline, CrimeSceneInput, AnalysisOutput

import re

# Expanded weapon terms (single tokens and phrases)
WEAPON_TERMS_SINGLE = {
    "gun", "handgun", "pistol", "revolver", "rifle", "shotgun", "firearm", "weapon",
    "ak47", "ak-47", "ak 47", "m16", "assault rifle", "machine gun", "submachine gun",
    "knife", "dagger", "blade", "sword", "machete",
    "bat", "club", "rod", "stick", "hammer", "crowbar", "wrench", "axe", "hatchet",
    "grenade", "bomb", "explosive", "molotov", "molotov cocktail",
    "taser", "stun gun", "pepper spray", "tear gas",
    "crossbow", "bow", "arrow"
}

WEAPON_PHRASES = {
    "assault rifle", "machine gun", "submachine gun",
    "stun gun", "pepper spray", "tear gas", "molotov cocktail"
}

# Optional local-language aliases (add/remove as needed)
LOCAL_ALIASES = {
    "bandook": "gun",
    "chaku": "knife",
    "talwar": "sword",
    "bombe": "bomb",
    "hand grenade": "grenade",
}

def normalize_text(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[\-_/]", " ", t)      # normalize hyphens/slashes
    t = re.sub(r"[^\w\s]", " ", t)     # strip punctuation
    t = re.sub(r"\s+", " ", t).strip() # collapse spaces
    return t

def detect_weapons_in_text(text: str):
    if not text:
        return []

    tnorm = normalize_text(text)
    found = set()

    # 1) Detect multi-word phrases first
    for phrase in WEAPON_PHRASES:
        pnorm = normalize_text(phrase)
        if re.search(rf"\b{re.escape(pnorm)}\b", tnorm):
            found.add(phrase)

    # 2) Map local aliases to canonical terms
    for alias, canonical in LOCAL_ALIASES.items():
        anorm = normalize_text(alias)
        if re.search(rf"\b{re.escape(anorm)}\b", tnorm):
            found.add(canonical)

    # 3) Detect single tokens (with simple plural handling)
    for term in WEAPON_TERMS_SINGLE:
        term_norm = normalize_text(term)
        # Build plural-aware pattern: gun|guns, knife|knives
        if term_norm.endswith("fe"):
            plural = term_norm[:-2] + "ves"
        elif term_norm.endswith("f"):
            plural = term_norm[:-1] + "ves"
        elif term_norm.endswith("y"):
            plural = term_norm[:-1] + "ies"
        else:
            plural = term_norm + "s"
        pattern = rf"\b({re.escape(term_norm)}|{re.escape(plural)})\b"
        if re.search(pattern, tnorm):
            found.add(term_norm)

    return sorted(found)

def highlight_matches(text: str, terms: list):
    # Lightweight highlighter to show matched words for debugging and trust
    highlighted = text
    for term in terms:
        term_norm = normalize_text(term)
        highlighted = re.sub(rf"(?i)\b{re.escape(term_norm)}\b", f"[{term}]", highlighted)
    return highlighted

@st.cache_resource
def get_classifier():
    return CrimeClipClassifier()

clf = get_classifier()

def init_pipeline(model_choice):
    try:
        return ScenePipeline(model_choice=model_choice)
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {e}")
        return None

def display_results(analysis: AnalysisOutput, pipeline):
    st.success("✅ Analysis Complete!")

    crime_display = analysis.crime_type
    if crime_display.lower() == "unknown":
        crime_display = "Possible crime type (low confidence)"

    col1, col2, col3 = st.columns(3)
    col1.metric("Crime Type", crime_display)
    col2.metric("Confidence", f"{analysis.confidence_score:.1%}")
    col3.metric("Risk Level", analysis.risk_assessment.get('overall_risk', 'Uncertain'))

    st.subheader("🔫 Weapons detected")
    weapons_summary = analysis.risk_assessment.get("weapons_detected", [])
    if weapons_summary:
        st.write(", ".join(sorted(set(weapons_summary))))
    else:
        st.write("No weapons mentioned")

    st.subheader("💾 Export Results")
    report_text = pipeline.generate_report(analysis)
    st.download_button("📄 Download Report (TXT)", report_text, f"{analysis.case_id}_report.txt", "text/plain")

    analysis_dict = {
        "case_id": analysis.case_id,
        "crime_type": analysis.crime_type,
        "confidence_score": analysis.confidence_score,
        "timeline": [
            {"description": e.description, "event_type": e.event_type, "confidence": e.confidence, "evidence": e.evidence}
            for e in analysis.timeline
        ],
        "risk_assessment": analysis.risk_assessment,
        "recommendations": analysis.recommendations
    }
    st.download_button("📊 Download Data (JSON)", json.dumps(analysis_dict, indent=2), f"{analysis.case_id}_data.json", "application/json")

# --- UI Layout ---
st.set_page_config(page_title="EvidexAI", layout="centered")

st.markdown("""
    <div style='background-color:#6A0DAD;padding:10px;border-radius:5px'>
        <h2 style='color:white;text-align:center;'>🔎 EvidexAI Crime Scene Analyzer</h2>
        <p style='color:white;text-align:center;'>AI-powered forensic dashboard for crime scene analysis</p>
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.01)
    model_choice = st.radio("Detection Mode", ["Lite", "Ultra-lite"])
    st.markdown("---")
    st.info("Upload FIR text and evidence in tabs below.")

if 'pipeline' not in st.session_state:
    with st.spinner("Initializing AI models..."):
        st.session_state.pipeline = init_pipeline(model_choice)

if st.session_state.pipeline is None:
    st.error("❌ Pipeline initialization failed.")
    st.stop()

pipeline = st.session_state.pipeline

tab_images, tab_videos, tab_fir = st.tabs(["🖼️ Images", "🎥 Videos", "📄 FIR"])

# --- Images Tab ---
with tab_images:
    st.subheader("🖼️ Upload Crime Scene Images")
    st.caption("📂 Drag and drop your images here or click to browse.")
    st.markdown("**Max file size:** 200MB per file  \n**Supported formats:** JPG, JPEG, PNG")

    uploads = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploads:
        for f in uploads:
            img = Image.open(io.BytesIO(f.read())).convert("RGB")
            preds = clf.classify_image(img, top_k=3, simple=True)
            st.image(img, caption=f.name, use_container_width=True)
            top_label, top_conf = preds[0]

            if top_conf >= confidence_threshold:
                st.success(f"Predicted crime type: {top_label} ({top_conf:.2f})")
            else:
                st.warning(f"Possible crime type: {top_label} (low confidence {top_conf:.2f})")

            st.write({"Applicable Laws": laws_for_label(top_label)})

# --- Videos Tab ---
with tab_videos:
    st.subheader("🎥 Upload Crime Scene Video")
    st.caption("📂 Drag and drop your video file here or click to browse.")
    st.markdown("**Max file size:** 200MB per file  \n**Supported formats:** MP4, MOV, AVI, MKV")

    vid = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

    if vid:
        with tempfile.NamedTemporaryFile(delete=False, suffix=vid.name) as tmp:
            tmp.write(vid.read())
            path = tmp.name

        frames, times = sample_frames(path, step_seconds=2)
        if frames:
            labels = []
            for i, (img, t) in enumerate(zip(frames, times)):
                preds = clf.classify_image(img, top_k=1, simple=True)
                labels.append(preds[0][0])
                if i < 6:
                    st.image(img, caption=f"t={t:.1f}s: {preds}", use_container_width=True)

            maj_label, votes = majority_vote(labels)
            if maj_label:
                st.success(f"Video predicted crime type: {maj_label} (votes {votes})")
                st.write({"Applicable Laws": laws_for_label(maj_label)})
            else:
                st.warning("No clear crime type detected — showing best guess.")
        else:
            st.error("Could not read frames.")

# --- FIR Tab ---
with tab_fir:
    st.subheader("📄 FIR Text")
    fir_text = st.text_area("Enter FIR details", height=150)

    with st.expander("⏰ Timeline Events"):
        if 'timeline_events' not in st.session_state:
            st.session_state.timeline_events = []
        for i, event in enumerate(st.session_state.timeline_events):
            st.write(f"- {event['time']}: {event['description']}")
        event_time = st.text_input("Time", key="new_time")
        event_desc = st.text_input("Event Description", key="new_desc")
        if st.button("➕ Add Event"):
            if event_time and event_desc:
                st.session_state.timeline_events.append({'time': event_time, 'description': event_desc})
                st.rerun()

    with st.expander("📸 Visual Evidence"):
        uploaded_images = st.file_uploader(
            "Upload Crime Scene Images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            label_visibility="visible"
        )

    timeline_text = "\n".join([f"- {e['time']}: {e['description']}" for e in st.session_state.timeline_events])
    full_fir_text = fir_text + ("\nTimeline:\n" + timeline_text if timeline_text else "")

    case_id = st.text_input("🆔 Case ID", value=f"CASE_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    location = st.text_input("📍 Location")

if st.button("🔍 Analyze Crime Scene", type="primary"):
    # Detect weapons from FIR text only after button is clicked
    weapons_in_fir = detect_weapons_in_text(full_fir_text)
    if not full_fir_text:
        st.warning("⚠️ Please enter FIR text.")
    else:
        image_paths = []
        if uploaded_images:
            temp_dir = tempfile.mkdtemp()
            for i, uploaded_image in enumerate(uploaded_images):
                temp_path = os.path.join(temp_dir, f"scene_{i}.jpg")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_image.read())
                image_paths.append(temp_path)

        scene_input = CrimeSceneInput(
            fir_text=full_fir_text,
            image_paths=image_paths,
            case_id=case_id,
            location=location,
            timestamp=datetime.datetime.now()
        )

        # Use only FIR-based weapon detection
        all_weapons = weapons_in_fir

        with st.spinner("🔍 Analyzing crime scene..."):
            try:
                analysis = pipeline.analyze_scene(scene_input)
                analysis.risk_assessment = analysis.risk_assessment or {}
                analysis.risk_assessment["weapons_detected"] = all_weapons
                if all_weapons and analysis.risk_assessment.get("overall_risk", "").lower() != "high":
                    analysis.risk_assessment["overall_risk"] = "High"
                st.session_state.last_analysis = analysis
                display_results(analysis, pipeline)
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")

    st.markdown("""
    <hr>
    <p style='text-align:center;'>© 2025 EvidexAI | VTU Project Submission</p>
""", unsafe_allow_html=True)