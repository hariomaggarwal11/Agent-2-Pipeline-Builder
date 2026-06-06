# 🔬 Agent-2-Pipeline-Builder

**NeuroPipeline** — An automated ML pipeline code generation agent for EEG and ECG biomedical signal analysis. Consumes Agent 1 (NeuroInspect) JSON reports and generates complete, runnable Python ML pipelines tailored to the dataset, from raw signal to evaluated model.

---

## 🧭 Where Agent 2 Sits

```
┌─────────────────────┐          ┌─────────────────────┐          ┌──────────────────┐
│  Agent 1            │  JSON    │  Agent 2            │  .zip    │  Researcher      │
│  NeuroInspect       │─────────▶│  NeuroPipeline      │─────────▶│  Runs pipeline   │
│  (Dataset Inspector)│  report  │  (Pipeline Builder) │  project │  on their data   │
└─────────────────────┘          └─────────────────────┘          └──────────────────┘
```

Agent 1 ([neuroinspect](https://github.com/hariomaggarwal11/neuroinspect)) inspects EEG/ECG datasets and exports a structured JSON report. Agent 2 consumes that JSON and builds a **complete, runnable ML pipeline** - the researcher never writes boilerplate preprocessing or training loop code again.

---

## 🎯 Overview

NeuroPipeline is a Streamlit-based code-generation agent that:

1. **Accepts** Agent 1's JSON report OR a new file upload (re-runs NeuroInspect internally)
2. **Presents** a visual, configurable pipeline builder with connected stage nodes
3. **Generates** complete, runnable Python scripts for each pipeline stage using Jinja2 templates
4. **Previews** live code - the researcher sees exact code as they adjust configuration
5. **Executes** each stage with real-time stdout streaming
6. **Exports** the finished pipeline as a downloadable `.zip` project

Everything generated is tailored to the detected dataset (DREAMER, MIT-BIH, PTB-XL), modality (EEG/ECG), task (emotion recognition, arrhythmia detection, motor imagery), and chosen model family (GAT-TCN, EEGNet, ResNet-1D, etc.).

---

## ✨ Key Features

### 🧠 EEG Pipeline (8 Stages)
| Stage | Purpose |
|-------|---------|
| 01. Quality Check | Validates Agent 1 findings, gates preprocessing |
| 02. Preprocessing | Bandpass filter, notch, ICA, re-reference, bad channel interpolation |
| 03. Epoching | Event-based, fixed-length, or trial-based (DREAMER/DEAP) with label binarization |
| 04. Feature Extraction | Band power, PLV connectivity, FAA, combined features for GAT-TCN |
| 05. Feature Selection | mRMR, Mutual Information, ANOVA, RFE, or skip for DL |
| 06. Model Development | GAT-TCN, EEGNet, LSTM, Transformer, or Custom (Claude API) |
| 07. Evaluation | LOSO-CV, K-Fold, confusion matrix, ROC, permutation test |
| 08. Interpretation | SHAP, GNN-Explainer, attention weights, FAA per class |

### 🫀 ECG Pipeline (9 Stages)
| Stage | Purpose |
|-------|---------|
| 01. Quality Check | ECG-specific quality gates from Agent 1 |
| 02. Preprocessing | Bandpass, baseline correction, wavelet denoising, lead selection |
| 03. R-Peak Detection | Pan-Tompkins, Hamilton, NeuroKit2, ectopic beat correction |
| 04. Beat Segmentation | R-peak centered windows, per-beat normalization |
| 05. Feature Extraction | Time-domain HRV, frequency-domain HRV, nonlinear HRV, morphological |
| 06. Feature Selection | mRMR (default for clinical interpretability) |
| 07. Model Development | ResNet-1D, CNN-1D+LSTM, EfficientNet-1D, Transformer-ECG |
| 08. Evaluation | Sensitivity, specificity, PPV, NPV, AHA compliance |
| 09. Clinical Interpretation | Rhythm classification, SNOMED-CT mapping, ICD-10 codes |

### 🤖 Auto-Recommendation Engine
The Pipeline Planner automatically recommends optimal configurations based on Agent 1's report:

| Dataset | Recommended Model | Recommended Strategy |
|---------|------------------|---------------------|
| DREAMER | GAT-TCN | Trial epoching + peak60 crop + LOSO-CV |
| DEAP | GAT-TCN | Trial epoching + LOSO-CV |
| SEED | LSTM | Trial epoching + 3-class |
| PhysioNet MI | EEGNet | Event-based + K-Fold |
| MIT-BIH | CNN-1D+LSTM | Beat segmentation + patient split |
| PTB-XL | ResNet-1D | 12-lead + stratified split |

### 🧬 Custom Model Generation (Claude API)
Researchers describe their model in plain English, and NeuroPipeline uses Claude to generate a complete PyTorch architecture:

> "I want a dual-branch model where one branch processes frontal channels for alpha asymmetry and another processes all channels for connectivity. Both branches should be GRU-based."

---

## 🛠 Technology Stack

| Component | Technology |
|-----------|-----------|
| Frontend / UI | Streamlit >=1.35.0 |
| Code Generation | Anthropic Claude API + Jinja2 >=3.1.0 |
| EEG Processing | MNE-Python >=1.7.0 |
| ECG Processing | WFDB >=4.1.0, NeuroKit2 >=0.21.0 |
| ML / Deep Learning | PyTorch >=2.3.0, torch-geometric >=2.5.0, scikit-learn >=1.5.0 |
| Explainability | SHAP >=0.45.0, captum >=0.7.0 |
| Visualization | Plotly >=5.22.0, Matplotlib >=3.9.0, Seaborn >=0.13.0 |
| Export | zipfile (stdlib), ReportLab >=4.2.0 |
| Code Formatting | Black >=24.4.0 |
| Testing | pytest >=8.2.0 |

---

## 📁 Project Structure

```
Agent-2-Pipeline-Builder/
├── README.md
└── neuropipeline/
    ├── app.py                          # Streamlit entry point
    ├── requirements.txt
    ├── config.py                       # MODEL_REGISTRY, stages, dataset tasks
    ├── core/
    │   ├── agent1_bridge.py            # Reads Agent 1 JSON / re-runs NeuroInspect
    │   ├── pipeline_planner.py         # Auto-recommendation engine
    │   ├── code_generator.py           # Anthropic Claude API integration
    │   ├── executor.py                 # Subprocess-based stage execution
    │   ├── eeg/stages/                 # 8 EEG stage modules
    │   ├── ecg/stages/                 # 9 ECG stage modules
    │   └── shared/
    │       ├── model_registry.py       # 8 supported architectures
    │       ├── exporter.py             # ZIP project export
    │       └── report_writer.py        # PDF pipeline report
    ├── templates/
    │   ├── eeg/                        # 11 Jinja2 templates (.py.j2)
    │   │   ├── quality_check.py.j2
    │   │   ├── preprocessing.py.j2
    │   │   ├── epoching.py.j2
    │   │   ├── feature_extraction.py.j2
    │   │   ├── feature_selection.py.j2
    │   │   ├── model_gat_tcn.py.j2
    │   │   ├── model_eegnet.py.j2
    │   │   ├── model_lstm.py.j2
    │   │   ├── model_transformer.py.j2
    │   │   ├── evaluation.py.j2
    │   │   └── interpretation.py.j2
    │   └── ecg/                        # 11 Jinja2 templates (.py.j2)
    │       ├── quality_check.py.j2
    │       ├── preprocessing.py.j2
    │       ├── rpeak_detection.py.j2
    │       ├── beat_segmentation.py.j2
    │       ├── feature_extraction.py.j2
    │       ├── feature_selection.py.j2
    │       ├── model_cnn1d.py.j2
    │       ├── model_resnet.py.j2
    │       ├── model_lstm.py.j2
    │       ├── evaluation.py.j2
    │       └── clinical_interpretation.py.j2
    ├── ui/
    │   ├── landing.py                  # JSON upload / file upload screen
    │   ├── pipeline_canvas.py          # Visual pipeline flow builder
    │   ├── stage_configurator.py       # Per-stage config panels
    │   ├── code_viewer.py              # Live code preview with syntax highlighting
    │   ├── executor_panel.py           # Live execution + stdout streaming
    │   ├── results_dashboard.py        # Metrics, plots, model summary
    │   ├── components.py               # Cards, badges, stage nodes
    │   └── styles.py                   # Dark scientific theme CSS
    └── tests/
        ├── conftest.py
        ├── test_agent1_bridge.py
        ├── test_pipeline_planner.py
        ├── test_eeg_stages.py
        ├── test_ecg_stages.py
        └── test_code_generator.py
```

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/hariomaggarwal11/Agent-2-Pipeline-Builder.git
cd Agent-2-Pipeline-Builder/neuropipeline

# Install dependencies
pip install -r requirements.txt

# Set Anthropic API key (for custom model generation)
export ANTHROPIC_API_KEY=your_api_key_here

# Run the application
streamlit run app.py

# Open in browser: http://localhost:8501
```

---

## 🧪 Running Tests

```bash
cd Agent-2-Pipeline-Builder
python -m pytest neuropipeline/tests/ -v
```

**234 tests** covering pipeline planner, agent1 bridge, EEG stages, ECG stages, and code generator - all passing.

---

## 🎨 UI Theme

NeuroPipeline extends Agent 1's precision-scientific dark theme with pipeline-specific additions:

- **Space Grotesk** for headers (distinguishes from Agent 1's Inter)
- **JetBrains Mono** for code display
- Pipeline nodes as connected blocks (visual dataflow editor feel)
- Node states: 🔧 Pending / ⚡ Running / ✅ Done / ❌ Failed / ⏭ Skipped
- Deep navy background with electric cyan accents

---

## 📊 Supported Model Architectures

| Architecture | Best For | Key Feature |
|-------------|----------|-------------|
| **GAT-TCN** | EEG Emotion Recognition | Spatial (graph) + temporal (dilated conv) |
| **EEGNet** | Motor Imagery / Low-channel EEG | Compact, depthwise separable convolutions |
| **LSTM / BiLSTM** | Sequential EEG/ECG | Captures long-range temporal dependencies |
| **Transformer** | EEG with attention | Multi-head self-attention over channels |
| **ResNet-1D** | 12-lead ECG Arrhythmia | Residual connections, deep architecture |
| **CNN-1D + LSTM** | Holter/Wearable ECG | Hybrid feature extraction + sequence |
| **EfficientNet-1D** | Mobile ECG Deployment | Lightweight, compound scaling |
| **Custom (Claude API)** | Any novel architecture | AI-generated from plain English description |

---

## 📦 Pipeline Export

The exported `.zip` project contains:

```
neuropipeline_project/
├── README.md               # Auto-generated description of each stage
├── requirements.txt        # Exact versions for reproducibility
├── run_pipeline.sh         # Bash script to run all stages in order
├── pipeline_config.json    # Full configuration for reproducibility
├── 01_quality_check.py
├── 02_preprocessing.py
├── 03_epoching.py          # (or rpeak_detection for ECG)
├── 04_feature_extraction.py
├── 05_feature_selection.py
├── 06_model_<architecture>.py
├── 07_evaluation.py
└── 08_interpretation.py    # (or 09_clinical_interpretation for ECG)
```

Each script is self-contained, well-documented, and directly runnable.

---

## 🔬 Research Features

- **LOSO Cross-Validation**: Leave-One-Subject-Out for cross-subject generalization
- **Frontal Alpha Asymmetry (FAA)**: Biomarker for emotional valence in EEG
- **Phase Locking Value (PLV)**: Brain connectivity for GAT-based models
- **HRV Analysis**: Complete time/frequency/nonlinear domain features
- **Clinical Metrics**: Sensitivity, specificity, PPV, NPV with AHA compliance
- **Permutation Tests**: Statistical significance validation
- **SHAP / GNN-Explainer**: Model interpretability for publication

---

## 🏗 Part of the BioMed Research Agent Suite

| Agent | Role | Repository |
|-------|------|-----------|
| **Agent 1** | Dataset Inspector | [neuroinspect](https://github.com/hariomaggarwal11/neuroinspect) |
| **Agent 2** | Pipeline Builder | [Agent-2-Pipeline-Builder](https://github.com/hariomaggarwal11/Agent-2-Pipeline-Builder) |

Agent 1 inspects -> Agent 2 builds -> Researcher executes and publishes.

---

## 📝 License

This project is part of the BioMed Research Agent Suite.

---

*NeuroPipeline v1.0 - Powered by MNE-Python + WFDB + PyTorch + Claude API*
*Consumes: Agent 1 (NeuroInspect) JSON export*
*Produces: Complete, runnable Python ML pipeline project (.zip)*
