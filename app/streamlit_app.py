import streamlit as st
import sys
import os
import tempfile
import datetime
from PIL import Image
import io
import json

# Add root folder to path so 'vision' can be found
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Core modules
from vision.crime_classifier_clip import CrimeClipClassifier
from vision.video_utils import sample_frames, majority_vote
from pipeline.law_resolver import laws_for_label
from scene_pipeline import ScenePipeline, CrimeSceneInput, AnalysisOutput

# Cache classifier
@st.cache_resource
def get_classifier():
    return CrimeClipClassifier()

clf = get_classifier()

def init_pipeline(model_choice):
    """Initialize the pipeline with error handling"""
    try:
        return ScenePipeline(model_choice=model_choice)
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {e}")
        return None

def main():
    # Page setup
    st.set_page_config(page_title="EvidexAI", layout="centered")
    st.title("🔎 EvidexAI")
    st.caption("AI-powered forensic crime scene analyzer")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.01)
        model_choice = st.radio("Detection Mode", ["Lite", "Ultra-lite"])
        st.markdown("---")
        st.info("Upload FIR text and evidence in tabs below.")

    # Initialize pipeline
    if 'pipeline' not in st.session_state:
        with st.spinner("Initializing AI models..."):
            st.session_state.pipeline = init_pipeline(model_choice)

    if st.session_state.pipeline is None:
        st.error("❌ Pipeline initialization failed.")
        return

    pipeline = st.session_state.pipeline

    # Tabs
    tab_images, tab_videos, tab_fir = st.tabs(["🖼️ Images", "🎥 Videos", "📄 FIR"])

    # Image tab
    with tab_images:
        uploads = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if uploads:
            for f in uploads:
                img = Image.open(io.BytesIO(f.read())).convert("RGB")
                preds = clf.classify_image(img, top_k=3, simple=True)
                st.image(img, caption=f.name, use_column_width=True)
                top_label, top_conf = preds[0]

                # ✅ Always show possible crime type
                if top_conf >= confidence_threshold:
                    st.success(f"Predicted crime type: {top_label} ({top_conf:.2f})")
                else:
                    st.warning(f"Possible crime type: {top_label} (low confidence {top_conf:.2f})")

                st.write({"Applicable Laws": laws_for_label(top_label)})

    # Video tab
    with tab_videos:
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
                        st.image(img, caption=f"t={t:.1f}s: {preds}", use_column_width=True)
                maj_label, votes = majority_vote(labels)
                if maj_label:
                    st.success(f"Video predicted crime type: {maj_label} (votes {votes})")
                    st.write({"Applicable Laws": laws_for_label(maj_label)})
                else:
                    st.warning("No clear crime type detected — showing best guess.")
            else:
                st.error("Could not read frames.")

    # FIR tab
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
            uploaded_images = st.file_uploader("Upload Crime Scene Images", type=["jpg","jpeg","png"], accept_multiple_files=True)

        timeline_text = "\n".join([f"- {e['time']}: {e['description']}" for e in st.session_state.timeline_events])
        full_fir_text = fir_text + ("\nTimeline:\n" + timeline_text if timeline_text else "")

        case_id = st.text_input("🆔 Case ID", value=f"CASE_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        location = st.text_input("📍 Location")

        if st.button("🔍 Analyze Crime Scene", type="primary"):
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

                with st.spinner("🔍 Analyzing crime scene..."):
                    try:
                        analysis = pipeline.analyze_scene(scene_input)
                        st.session_state.last_analysis = analysis
                        display_results(analysis, pipeline)
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")

def display_results(analysis: AnalysisOutput, pipeline):
    st.success("✅ Analysis Complete!")

    # ✅ Always show possible crime type with confidence
    crime_display = analysis.crime_type
    if crime_display.lower() == "unknown":
        crime_display = f"Possible crime type (low confidence)"

    col1, col2, col3 = st.columns(3)
    col1.metric("Crime Type", crime_display)
    col2.metric("Confidence", f"{analysis.confidence_score:.1%}")
    col3.metric("Risk Level", analysis.risk_assessment.get('overall_risk', 'Uncertain'))

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

if __name__ == "__main__":
    main()