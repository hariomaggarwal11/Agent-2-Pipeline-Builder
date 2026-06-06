"""Shared fixtures for NeuroPipeline test suite."""

import sys
import os
import pytest

# Ensure neuropipeline package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.fixture
def sample_agent1_eeg_report():
    """DREAMER-like Agent 1 EEG report."""
    return {
        "modality": "EEG",
        "known_dataset": {"name": "DREAMER"},
        "quality_score": {
            "score": 85,
            "breakdown": [
                {"check": "flat_channels", "score": 100},
                {"check": "line_noise", "score": 80},
                {"check": "eog_artifacts", "score": 75},
                {"check": "emg_artifacts", "score": 85},
            ],
        },
        "metadata": {
            "Number of Channels": 14,
            "Sampling Frequency": "128 Hz",
            "Duration": "60 min 0 s",
        },
        "artifacts": {
            "flat_channels": {"status": "pass", "channels": []},
            "line_noise": {
                "status": "pass",
                "detected": False,
                "frequency_hz": None,
            },
            "eog_artifacts": {"status": "pass"},
            "emg_artifacts": {"status": "pass"},
        },
        "events": {"has_events": True},
    }


@pytest.fixture
def sample_agent1_ecg_report():
    """MIT-BIH Arrhythmia-like Agent 1 ECG report."""
    return {
        "modality": "ECG",
        "known_dataset": {"name": "MIT-BIH Arrhythmia"},
        "quality_score": {"score": 90},
        "metadata": {
            "Number of Leads": 2,
            "Sampling Frequency": "360 Hz",
            "Duration": "30 min 0 s",
        },
        "artifacts": {
            "signal_clipping": {"status": "pass"},
            "powerline_noise": {"status": "pass"},
        },
    }


@pytest.fixture
def sample_pipeline_config(sample_agent1_eeg_report):
    """Pipeline config output from plan_pipeline for DREAMER."""
    from neuropipeline.core.pipeline_planner import plan_pipeline

    return plan_pipeline(sample_agent1_eeg_report)
