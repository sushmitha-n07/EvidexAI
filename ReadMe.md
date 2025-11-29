# EvidexAI

EvidexAI is a modular, multi‑modal AI framework combining NLP, computer
vision and pipeline orchestration to analyze evidence --- enabling
functionality such as image/video processing, text analysis, and
data-driven workflows.

## 🧩 Project Structure

    │── agent/            # AI reasoning agents / orchestration logic  
    │── app/              # Application / user‑facing modules  
    │── data/             # Data storage (raw inputs / processed data / caches)  
    │── models/           # Pretrained / custom ML & deep learning models  
    │── nlp/              # Natural language processing modules  
    │── pipeline/         # Pipeline orchestration (core processing flow)  
    │── vision/           # Computer vision modules (image/video analysis)  
    │── requirements.txt  # Python dependencies  
    │── *.py              # Entry‑point or utility scripts  
    └── …  

## 🔑 Key Features

-   **Multi‑modal Processing:** Supports both textual and visual inputs
    (NLP + computer vision).\
-   **Modular Architecture:** Well‑organized into submodules (agents,
    nlp, vision, pipeline, data) for easy extension and maintenance.\
-   **Pipeline-driven Workflow:** Central pipeline ensures orderly data
    flow, pre-/post-processing, and integration of various modules.\
-   **Extensible Models:** Easily integrate new or custom-trained models
    in the `models/` directory.\
-   **Test Suite & Utilities:** Includes unit tests to ensure code
    quality and reliability.

## 🚀 Getting Started

### Prerequisites

-   Python 3.x\
-   (Optional) GPU / CUDA support

### Installation

``` bash
git clone https://github.com/sushmitha-n07/EvidexAI.git
cd EvidexAI
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running

``` bash
python pipeline/some_entry_point.py
```

## 🧪 Testing

``` bash
pytest
```

## 📦 Dependencies

See `requirements.txt` for all dependencies.

## ⚙️ High-Level Flow

1.  Input handling\
2.  Preprocessing\
3.  NLP / Vision processing\
4.  Pipeline orchestration\
5.  Output generation

## 🔄 Contribution

-   Add new models under `models/`\
-   Add modules in `nlp/` or `vision/`\
-   Extend pipelines in `pipeline/`\
-   Add tests for new components

## 📞 Contact

For issues or contributions, open a GitHub issue.