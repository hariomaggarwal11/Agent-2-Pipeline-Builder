"""NeuroPipeline configuration constants."""

# Anthropic API configuration
ANTHROPIC_API_KEY = "fe_oa_04d4162e8c7e6e8a5584ee4491040b73f9895884e2ed85b3"
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
ANTHROPIC_MAX_TOKENS = 4096

# Dark theme colors (extends Agent 1 palette with pipeline-specific colors)
THEME_COLORS = {
    # Base theme from Agent 1
    "--bg-primary": "#0a0e1a",
    "--bg-surface": "#111827",
    "--bg-elevated": "#1a2235",
    "--accent-eeg": "#00d4ff",
    "--accent-ecg": "#ff4d6d",
    "--accent-ok": "#22c55e",
    "--accent-warn": "#f59e0b",
    "--accent-fail": "#ef4444",
    "--text-primary": "#f1f5f9",
    "--text-muted": "#64748b",
    "--border": "#1e2d45",
    # Pipeline-specific colors
    "--pipeline-node-bg": "#1a2235",
    "--pipeline-node-border": "#2d3f5e",
    "--pipeline-node-active": "#00d4ff",
    "--pipeline-edge-color": "#374151",
    "--pipeline-done": "#22c55e",
    "--pipeline-running": "#f59e0b",
    "--pipeline-pending": "#64748b",
    "--code-bg": "#0d1117",
    "--code-border": "#21262d",
}

# Model Registry - all supported architectures
MODEL_REGISTRY = {
    "gat_tcn": {
        "name": "GAT-TCN",
        "description": "Graph Attention Network with Temporal Convolutional Network",
        "input_type": "graph_temporal",
        "recommended_for": ["emotion_recognition", "motor_imagery"],
        "min_channels": 8,
        "hyperparameters": {
            "num_heads": 4,
            "hidden_dim": 64,
            "tcn_channels": [64, 128, 256],
            "kernel_size": 3,
            "dropout": 0.3,
            "num_layers": 3,
        },
    },
    "eegnet": {
        "name": "EEGNet",
        "description": "Compact CNN designed for EEG-based BCI",
        "input_type": "temporal",
        "recommended_for": ["motor_imagery", "p300", "ssvep"],
        "min_channels": 2,
        "hyperparameters": {
            "F1": 8,
            "D": 2,
            "F2": 16,
            "kernel_length": 64,
            "dropout": 0.5,
        },
    },
    "cnn1d": {
        "name": "1D-CNN",
        "description": "One-dimensional Convolutional Neural Network",
        "input_type": "temporal",
        "recommended_for": ["arrhythmia_detection", "sleep_staging"],
        "min_channels": 1,
        "hyperparameters": {
            "num_filters": [32, 64, 128],
            "kernel_sizes": [7, 5, 3],
            "pool_sizes": [2, 2, 2],
            "dropout": 0.3,
            "fc_dim": 128,
        },
    },
    "lstm": {
        "name": "LSTM",
        "description": "Long Short-Term Memory recurrent network",
        "input_type": "temporal",
        "recommended_for": ["hrv_analysis", "sequence_classification"],
        "min_channels": 1,
        "hyperparameters": {
            "hidden_size": 128,
            "num_layers": 2,
            "bidirectional": True,
            "dropout": 0.3,
        },
    },
    "transformer": {
        "name": "Transformer",
        "description": "Multi-head self-attention transformer encoder",
        "input_type": "temporal",
        "recommended_for": ["emotion_recognition", "motor_imagery", "arrhythmia_detection"],
        "min_channels": 1,
        "hyperparameters": {
            "d_model": 128,
            "nhead": 8,
            "num_layers": 4,
            "dim_feedforward": 256,
            "dropout": 0.1,
        },
    },
    "resnet1d": {
        "name": "ResNet-1D",
        "description": "1D Residual Network for multi-lead ECG",
        "input_type": "temporal",
        "recommended_for": ["arrhythmia_detection", "ecg_classification"],
        "min_channels": 1,
        "hyperparameters": {
            "num_blocks": [2, 2, 2, 2],
            "base_filters": 64,
            "kernel_size": 15,
            "dropout": 0.2,
        },
    },
    "cnn1d_lstm": {
        "name": "CNN-LSTM",
        "description": "Hybrid CNN feature extractor with LSTM temporal modeling",
        "input_type": "temporal",
        "recommended_for": ["arrhythmia_detection", "hrv_analysis"],
        "min_channels": 1,
        "hyperparameters": {
            "cnn_filters": [32, 64],
            "cnn_kernel": 5,
            "lstm_hidden": 64,
            "lstm_layers": 2,
            "dropout": 0.3,
        },
    },
    "custom": {
        "name": "Custom Model",
        "description": "User-defined architecture generated via Claude API",
        "input_type": "any",
        "recommended_for": [],
        "min_channels": 1,
        "hyperparameters": {},
    },
}

# Template registry - maps stage names to template files
TEMPLATE_REGISTRY = {
    # EEG templates
    "eeg_quality_check": "templates/eeg/s1_quality_check.py.j2",
    "eeg_preprocessing": "templates/eeg/s2_preprocessing.py.j2",
    "eeg_feature_extraction": "templates/eeg/s3_feature_extraction.py.j2",
    "eeg_dataset_split": "templates/eeg/s4_dataset_split.py.j2",
    "eeg_model_definition": "templates/eeg/s5_model_definition.py.j2",
    "eeg_training": "templates/eeg/s6_training.py.j2",
    "eeg_evaluation": "templates/eeg/s7_evaluation.py.j2",
    "eeg_interpretation": "templates/eeg/s8_interpretation.py.j2",
    # ECG templates
    "ecg_quality_check": "templates/ecg/s1_quality_check.py.j2",
    "ecg_preprocessing": "templates/ecg/s2_preprocessing.py.j2",
    "ecg_beat_segmentation": "templates/ecg/s3_beat_segmentation.py.j2",
    "ecg_feature_extraction": "templates/ecg/s4_feature_extraction.py.j2",
    "ecg_dataset_split": "templates/ecg/s5_dataset_split.py.j2",
    "ecg_model_definition": "templates/ecg/s6_model_definition.py.j2",
    "ecg_training": "templates/ecg/s7_training.py.j2",
    "ecg_evaluation": "templates/ecg/s8_evaluation.py.j2",
    "ecg_clinical_interpretation": "templates/ecg/s9_clinical_interpretation.py.j2",
}

# EEG pipeline stages (ordered)
EEG_STAGES = [
    "s1_quality_check",
    "s2_preprocessing",
    "s3_feature_extraction",
    "s4_dataset_split",
    "s5_model_definition",
    "s6_training",
    "s7_evaluation",
    "s8_interpretation",
]

# ECG pipeline stages (ordered)
ECG_STAGES = [
    "s1_quality_check",
    "s2_preprocessing",
    "s3_beat_segmentation",
    "s4_feature_extraction",
    "s5_dataset_split",
    "s6_model_definition",
    "s7_training",
    "s8_evaluation",
    "s9_clinical_interpretation",
]

# Dataset to task mapping
DATASET_TASKS = {
    # EEG emotion datasets
    "DREAMER": {
        "task": "emotion_recognition",
        "epoch_type": "trial-based",
        "labels": ["arousal", "valence", "dominance"],
    },
    "DEAP": {
        "task": "emotion_recognition",
        "epoch_type": "trial-based",
        "labels": ["arousal", "valence", "dominance", "liking"],
    },
    "SEED": {
        "task": "emotion_recognition",
        "epoch_type": "trial-based",
        "labels": ["emotion"],
    },
    "MAHNOB-HCI": {
        "task": "emotion_recognition",
        "epoch_type": "trial-based",
        "labels": ["arousal", "valence"],
    },
    # EEG motor imagery datasets
    "PhysioNet Motor Imagery": {
        "task": "motor_imagery",
        "epoch_type": "event-based",
        "labels": ["left_hand", "right_hand", "both_fists", "both_feet"],
    },
    "BCICIV": {
        "task": "motor_imagery",
        "epoch_type": "event-based",
        "labels": ["left_hand", "right_hand", "both_feet", "tongue"],
    },
    # ECG datasets
    "MIT-BIH Arrhythmia": {
        "task": "arrhythmia_detection",
        "epoch_type": "beat-based",
        "labels": ["N", "L", "R", "V", "A", "F", "/", "Q"],
    },
    "PTB Diagnostic": {
        "task": "ecg_classification",
        "epoch_type": "record-based",
        "labels": ["MI", "Normal"],
    },
    "PTB-XL": {
        "task": "ecg_classification",
        "epoch_type": "record-based",
        "labels": ["NORM", "MI", "STTC", "CD", "HYP"],
    },
    "CPSC2018": {
        "task": "ecg_classification",
        "epoch_type": "record-based",
        "labels": ["AF", "I-AVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE", "Normal"],
    },
    "PhysioNet Challenge": {
        "task": "ecg_classification",
        "epoch_type": "record-based",
        "labels": [],
    },
}

# System prompt for Claude API code generation
CODE_GEN_SYSTEM_PROMPT = """You are an expert ML engineer specializing in EEG and ECG signal processing pipelines.
Generate clean, well-documented Python code for the requested pipeline stage.
Follow these conventions:
- Use type hints throughout
- Include docstrings for all functions and classes
- Use standard scientific Python libraries (numpy, scipy, scikit-learn, torch, mne)
- Handle edge cases gracefully with try/except blocks
- Include logging with the standard logging module
- Make code modular and testable
- Follow PEP 8 style guidelines
- Include inline comments for complex signal processing steps
"""
