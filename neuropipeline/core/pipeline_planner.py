"""Pipeline Planner - maps Agent 1 reports to recommended pipeline configurations."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import DATASET_TASKS, MODEL_REGISTRY, EEG_STAGES, ECG_STAGES


def plan_pipeline(agent1_report: dict) -> dict:
    """Map an Agent 1 report to a recommended pipeline configuration.

    Implements the following logic:
    - DREAMER/DEAP/SEED -> emotion_recognition, trial-based epochs, gat_tcn if >=8 channels
    - PhysioNet Motor Imagery/BCICIV -> motor_imagery, event-based epochs
    - ECG with >=12 leads -> resnet1d, else cnn1d_lstm
    - Quality >= 70 and n_ch >= 8 -> enable ICA
    - Include FAA for emotion_recognition tasks

    Parameters
    ----------
    agent1_report : dict
        Parsed Agent 1 report with keys: modality, metadata, quality_score,
        artifacts, known_dataset, events/annotations.

    Returns
    -------
    dict
        Pipeline configuration with: task, model, stages, epoch_type,
        preprocessing options, feature extraction settings.
    """
    modality = agent1_report.get("modality", "EEG")
    metadata = agent1_report.get("metadata", {})
    quality_score = agent1_report.get("quality_score", {})
    known_dataset = agent1_report.get("known_dataset", None)

    # Extract numeric values
    score = quality_score.get("score", 0) if isinstance(quality_score, dict) else 0
    n_ch = metadata.get("Number of Channels", metadata.get("Number of Leads", 0))
    if isinstance(n_ch, str):
        try:
            n_ch = int(n_ch)
        except ValueError:
            n_ch = 0

    # Determine dataset name
    dataset_name = None
    if known_dataset and isinstance(known_dataset, dict):
        dataset_name = known_dataset.get("name", None)

    # Get task info from dataset mapping
    task_info = DATASET_TASKS.get(dataset_name, {}) if dataset_name else {}
    task = task_info.get("task", _infer_task(modality, n_ch))
    epoch_type = task_info.get("epoch_type", _infer_epoch_type(modality))
    labels = task_info.get("labels", [])

    # Determine model recommendation
    model = _recommend_model(modality, task, n_ch)

    # Determine stages
    stages = EEG_STAGES if modality == "EEG" else ECG_STAGES

    # Build preprocessing config
    preprocessing = _build_preprocessing_config(modality, score, n_ch, task)

    # Build feature extraction config
    features = _build_feature_config(modality, task, n_ch)

    # Build full pipeline config
    config = {
        "modality": modality,
        "task": task,
        "dataset": dataset_name,
        "model": model,
        "model_config": MODEL_REGISTRY.get(model, {}).get("hyperparameters", {}),
        "stages": stages,
        "epoch_type": epoch_type,
        "labels": labels,
        "preprocessing": preprocessing,
        "features": features,
        "training": {
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "scheduler": "cosine",
            "early_stopping_patience": 10,
            "validation_split": 0.2,
        },
        "evaluation": {
            "metrics": ["accuracy", "f1_weighted", "confusion_matrix"],
            "cross_validation": "5-fold",
            "interpretation": True,
        },
    }

    return config


def _infer_task(modality: str, n_ch: int) -> str:
    """Infer task type when no known dataset is detected."""
    if modality == "ECG":
        return "arrhythmia_detection"
    if n_ch >= 8:
        return "emotion_recognition"
    return "classification"


def _infer_epoch_type(modality: str) -> str:
    """Infer epoch type based on modality."""
    if modality == "ECG":
        return "beat-based"
    return "trial-based"


def _recommend_model(modality: str, task: str, n_ch: int) -> str:
    """Recommend the best model architecture based on modality, task, and channels.

    Logic:
    - EEG emotion_recognition with >= 8 channels -> gat_tcn
    - EEG motor_imagery -> eegnet
    - ECG with >= 12 leads -> resnet1d
    - ECG with < 12 leads -> cnn1d_lstm
    """
    if modality == "EEG":
        if task == "emotion_recognition" and n_ch >= 8:
            return "gat_tcn"
        elif task == "motor_imagery":
            return "eegnet"
        elif n_ch >= 8:
            return "gat_tcn"
        else:
            return "cnn1d"
    elif modality == "ECG":
        if n_ch >= 12:
            return "resnet1d"
        else:
            return "cnn1d_lstm"
    return "cnn1d"


def _build_preprocessing_config(modality: str, quality_score: int, n_ch: int, task: str) -> dict:
    """Build preprocessing configuration based on data quality and modality.

    Logic:
    - Quality >= 70 and n_ch >= 8 -> enable ICA
    """
    config = {}

    if modality == "EEG":
        config["bandpass_filter"] = {"low": 0.5, "high": 45.0}
        config["notch_filter"] = 50.0
        config["resample"] = None  # Keep original unless needed
        config["reference"] = "average"

        # Enable ICA if quality is good and enough channels
        if quality_score >= 70 and n_ch >= 8:
            config["ica"] = {
                "enabled": True,
                "n_components": min(n_ch - 1, 20),
                "method": "fastica",
                "max_iter": 500,
            }
        else:
            config["ica"] = {"enabled": False}

        # Bad channel detection
        config["bad_channel_detection"] = True
        config["interpolate_bads"] = True

    elif modality == "ECG":
        config["bandpass_filter"] = {"low": 0.5, "high": 40.0}
        config["notch_filter"] = 50.0
        config["baseline_correction"] = True
        config["r_peak_detection"] = "neurokit2"

    return config


def _build_feature_config(modality: str, task: str, n_ch: int) -> dict:
    """Build feature extraction configuration.

    Includes FAA for emotion_recognition tasks.
    """
    config = {}

    if modality == "EEG":
        config["psd_bands"] = {
            "delta": [0.5, 4.0],
            "theta": [4.0, 8.0],
            "alpha": [8.0, 13.0],
            "beta": [13.0, 30.0],
            "gamma": [30.0, 45.0],
        }
        config["connectivity"] = n_ch >= 8
        config["temporal_features"] = True

        # Include FAA for emotion_recognition tasks
        if task == "emotion_recognition":
            config["faa"] = True
            config["asymmetry_pairs"] = [
                ("F3", "F4"),
                ("F7", "F8"),
                ("FC5", "FC6"),
            ]
        else:
            config["faa"] = False

    elif modality == "ECG":
        config["morphological"] = True
        config["hrv_time_domain"] = True
        config["hrv_frequency_domain"] = True
        config["hrv_nonlinear"] = True
        config["wavelet_features"] = True

    return config
