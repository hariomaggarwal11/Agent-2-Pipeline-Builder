"""EEG Stage 1: Quality Check - assess data quality and detect artifacts."""


def get_config_schema() -> dict:
    """Return the configuration schema for this stage.

    Returns
    -------
    dict
        Schema with parameter names, types, defaults, and descriptions.
    """
    return {
        "check_flat_channels": {
            "type": "bool",
            "default": True,
            "description": "Detect flat/dead channels",
        },
        "check_line_noise": {
            "type": "bool",
            "default": True,
            "description": "Detect powerline interference (50/60 Hz)",
        },
        "check_eog": {
            "type": "bool",
            "default": True,
            "description": "Detect eye movement artifacts",
        },
        "check_emg": {
            "type": "bool",
            "default": True,
            "description": "Detect muscle artifacts",
        },
        "flat_threshold": {
            "type": "float",
            "default": 1e-6,
            "description": "Threshold for flat channel detection (uV)",
        },
        "line_noise_freq": {
            "type": "float",
            "default": 50.0,
            "description": "Powerline frequency (50 or 60 Hz)",
        },
        "min_quality_score": {
            "type": "int",
            "default": 50,
            "description": "Minimum quality score to proceed (0-100)",
        },
    }


def get_default_config(dataset_info: dict) -> dict:
    """Return default configuration based on dataset info.

    Parameters
    ----------
    dataset_info : dict
        Information from Agent 1 report.

    Returns
    -------
    dict
        Default configuration values.
    """
    config = {
        "check_flat_channels": True,
        "check_line_noise": True,
        "check_eog": True,
        "check_emg": True,
        "flat_threshold": 1e-6,
        "line_noise_freq": 50.0,
        "min_quality_score": 50,
    }

    # Adjust for known datasets
    quality = dataset_info.get("quality_score", {})
    if isinstance(quality, dict) and quality.get("score", 0) >= 80:
        config["min_quality_score"] = 70

    return config


def generate_code(config: dict) -> str:
    """Generate Python code for the quality check stage.

    Parameters
    ----------
    config : dict
        Stage configuration.

    Returns
    -------
    str
        Generated Python code.
    """
    return f'''"""Stage 1: EEG Quality Check"""

import numpy as np
import mne
import logging

logger = logging.getLogger(__name__)

# Configuration
CHECK_FLAT = {config.get("check_flat_channels", True)}
CHECK_LINE_NOISE = {config.get("check_line_noise", True)}
CHECK_EOG = {config.get("check_eog", True)}
CHECK_EMG = {config.get("check_emg", True)}
FLAT_THRESHOLD = {config.get("flat_threshold", 1e-6)}
LINE_NOISE_FREQ = {config.get("line_noise_freq", 50.0)}
MIN_QUALITY_SCORE = {config.get("min_quality_score", 50)}


def check_flat_channels(raw: mne.io.Raw) -> dict:
    """Detect flat/dead channels."""
    flat_channels = []
    data = raw.get_data()
    for idx, ch_name in enumerate(raw.ch_names):
        if np.std(data[idx]) < FLAT_THRESHOLD:
            flat_channels.append(ch_name)
    return {{"flat_channels": flat_channels, "status": "pass" if not flat_channels else "fail"}}


def check_line_noise(raw: mne.io.Raw) -> dict:
    """Check for powerline noise."""
    from scipy.signal import welch
    data = raw.get_data()
    fs = raw.info["sfreq"]
    affected = []
    for idx, ch_name in enumerate(raw.ch_names):
        freqs, psd = welch(data[idx], fs=fs, nperseg=min(len(data[idx]), int(fs * 4)))
        noise_idx = np.argmin(np.abs(freqs - LINE_NOISE_FREQ))
        neighbors = psd[max(0, noise_idx - 5):noise_idx + 5]
        if psd[noise_idx] > np.median(neighbors) * 3:
            affected.append(ch_name)
    return {{"affected_channels": affected, "status": "pass" if not affected else "warn"}}


def run_quality_check(raw: mne.io.Raw) -> dict:
    """Run all quality checks."""
    results = {{"score": 100}}

    if CHECK_FLAT:
        flat_result = check_flat_channels(raw)
        results["flat_channels"] = flat_result
        if flat_result["status"] == "fail":
            results["score"] -= 20

    if CHECK_LINE_NOISE:
        noise_result = check_line_noise(raw)
        results["line_noise"] = noise_result
        if noise_result["status"] == "warn":
            results["score"] -= 10

    results["passed"] = results["score"] >= MIN_QUALITY_SCORE
    logger.info(f"Quality score: {{results['score']}}/100 - {{'PASS' if results['passed'] else 'FAIL'}}")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Quality check stage - requires raw EEG data input")
'''
