# EvidexAI

**EvidexAI** is a modular, multi-modal AI framework that combines natural language processing, computer vision, and pipeline orchestration to analyze crime scene evidence. It supports image/video classification, FIR text analysis, and structured risk assessment — all within a streamlined dashboard built for forensic workflows and VTU project submission.

---

## 🧩 Project Structure
│── agent/            # AI reasoning agents and orchestration logic
│── app/              # Streamlit dashboard and user-facing modules
│── data/             # Raw inputs, processed data, and cache storage
│── models/           # Pretrained and custom ML/DL models
│── nlp/              # FIR text analysis and NLP modules
│── pipeline/         # Core processing flow and integration logic
│── vision/           # Image/video classification and CV modules
│── requirements.txt  # Python dependencies
│── *.py              # Entry-point or utility scripts
└── …


---

## 🔑 Key Features

- **Multi-modal Evidence Analysis:** Combines FIR text, images, and videos for crime scene understanding  
- **Modular Architecture:** Clean separation of NLP, vision, pipeline, and agent logic  
- **Streamlit Dashboard:** Intuitive interface for uploading evidence, analyzing scenes, and exporting reports  
- **FIR-based Weapon Detection:** Detects weapons mentioned in FIR text using robust keyword and phrase matching  
- **Pipeline-driven Workflow:** Central pipeline handles preprocessing, model inference, and result aggregation  
- **Extensible Model Support:** Easily plug in custom-trained models (e.g., YOLOv8, CLIP)  
- **Evaluator-Ready Output:** Generates TXT and JSON reports with crime type, confidence, risk level, and evidence summary  

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+  
- (Optional) GPU / CUDA support for faster inference

### Installation

```bash
git clone https://github.com/sushmitha-n07/EvidexAI.git
cd EvidexAI
python -m venv venv
source venv/bin/activate   
# Windows: venv\Scripts\activate
pip install -r requirements.txt

### Running
streamlit run app/streamlit_app.py

## 🧪 Testing
pytest

📦 Dependencies
See requirements.txt for all dependencies including:
- Streamlit, Altair, Plotly for dashboard and visualization
- Torch, Transformers, SpaCy for ML/NLP
- Ultralytics (YOLOv8) for optional weapon detection
- OpenCV, Pillow for image handling
- WeasyPrint, ReportLab for PDF generation
- CLIP (via GitHub) for image/video classification
- Regex-based FIR weapon detection logic

⚙️ High-Level Flow
- User uploads FIR text, images, or videos
- Preprocessing and sampling (frames, text cleaning)
- NLP and Vision models classify crime type
- FIR text scanned for weapon mentions (after analysis trigger)
- Pipeline aggregates results and generates report


🔄 Contribution
- Add new models under models/
- Extend NLP logic in nlp/ or CV logic in vision/
- Modify orchestration in pipeline/
- Add unit tests for new components
- Submit issues or pull requests via GitHub

## 📞 Contact

For issues or contributions, open a GitHub issue.