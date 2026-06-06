"""ECG Stage 1: Quality Check - assess signal quality and detect artifacts."""


def get_config_schema() -> dict:
    """Return the configuration schema for this stage."""
    return {
        "check_clipping": {
            "type": "bool",
            "default": True,
            "description": "Detect signal clipping",
        },
        "check_powerline": {
            "type": "bool",
            "default": True,
            "description": "Detect powerline interference",
        },
        "check_baseline_wander": {
            "type": "bool",
            "default": True,
            "description": "Detect baseline wander",
        },
        "check_lead_quality": {
            "type": "bool",
            "default": True,
            "description": "Assess per-lead signal quality index",
        },
        "min_quality_index": {
            "type": "float",
            "default": 0.7,
            "description": "Minimum acceptable quality index (0-1)",
        },
        "clipping_threshold": {
            "type": "float",
            "default": 0.95,
            "description": "Fraction of max ADC value to flag as clipping",
        },
        "powerline_freq": {
            "type": "float",
            "default": 50.0,
            "description": "Powerline frequency (50 or 60 Hz)",
        },
    }


def get_default_config(dataset_info: dict) -> dict:
    """Return default configuration based on dataset info."""
    config = {
        "check_clipping": True,
        "check_powerline": True,
        "check_baseline_wander": True,
        "check_lead_quality": True,
        "min_quality_index": 0.7,
        "clipping_threshold": 0.95,
        "powerline_freq": 50.0,
    }
    return config


def generate_code(config: dict) -> str:
    """Generate Python code for the ECG quality check stage."""
    return f'''"""Stage 1: ECG Quality Check"""

import numpy as np
from scipy.signal import welch
import logging

logger = logging.getLogger(__name__)

# Configuration
CHECK_CLIPPING = {config.get("check_clipping", True)}
CHECK_POWERLINE = {config.get("check_powerline", True)}
CHECK_BASELINE_WANDER = {config.get("check_baseline_wander", True)}
CHECK_LEAD_QUALITY = {config.get("check_lead_quality", True)}
MIN_QUALITY_INDEX = {config.get("min_quality_index", 0.7)}
CLIPPING_THRESHOLD = {config.get("clipping_threshold", 0.95)}
POWERLINE_FREQ = {config.get("powerline_freq", 50.0)}


def check_clipping(signal: np.ndarray) -> dict:
    """Detect signal clipping in ECG leads."""
    max_val = np.max(np.abs(signal))
    threshold = max_val * CLIPPING_THRESHOLD
    clipped_samples = np.sum(np.abs(signal) >= threshold)
    fraction = clipped_samples / len(signal)
    return {{
        "clipped_fraction": fraction,
        "status": "pass" if fraction < 0.01 else "fail",
    }}


def check_powerline_noise(signal: np.ndarray, fs: float) -> dict:
    """Detect powerline interference."""
    freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), int(fs * 4)))
    noise_idx = np.argmin(np.abs(freqs - POWERLINE_FREQ))
    noise_power = psd[noise_idx]
    median_power = np.median(psd)
    ratio = noise_power / (median_power + 1e-10)
    return {{
        "noise_ratio": ratio,
        "status": "pass" if ratio < 5.0 else "warn",
    }}


def compute_signal_quality_index(signal: np.ndarray, fs: float) -> float:
    """Compute signal quality index (SQI) for a single lead."""
    # Template-based SQI: correlation with running average
    window = int(fs * 0.6)  # 600ms window (typical RR interval)
    if len(signal) < window * 3:
        return 0.5
    segments = [signal[i:i+window] for i in range(0, len(signal) - window, window)]
    if len(segments) < 2:
        return 0.5
    template = np.mean(segments[:10], axis=0)
    correlations = []
    for seg in segments:
        if len(seg) == len(template):
            corr = np.corrcoef(template, seg)[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))
    return np.mean(correlations) if correlations else 0.5


def run_quality_check(signals: np.ndarray, fs: float, lead_names: list = None) -> dict:
    """Run all quality checks on multi-lead ECG.

    Parameters
    ----------
    signals : np.ndarray
        Shape (n_leads, n_samples)
    fs : float
        Sampling frequency
    lead_names : list
        Lead names

    Returns
    -------
    dict
        Quality assessment results
    """
    n_leads = signals.shape[0]
    results = {{"overall_quality": 0.0, "per_lead": {{}}}}

    lead_names = lead_names or [f"Lead_{{i}}" for i in range(n_leads)]
    quality_scores = []

    for i, name in enumerate(lead_names):
        lead_result = {{}}

        if CHECK_CLIPPING:
            lead_result["clipping"] = check_clipping(signals[i])

        if CHECK_POWERLINE:
            lead_result["powerline"] = check_powerline_noise(signals[i], fs)

        if CHECK_LEAD_QUALITY:
            sqi = compute_signal_quality_index(signals[i], fs)
            lead_result["quality_index"] = sqi
            quality_scores.append(sqi)

        results["per_lead"][name] = lead_result

    results["overall_quality"] = np.mean(quality_scores) if quality_scores else 0.5
    results["passed"] = results["overall_quality"] >= MIN_QUALITY_INDEX
    logger.info(f"Overall quality: {{results['overall_quality']:.3f}} - {{'PASS' if results['passed'] else 'FAIL'}}")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("ECG Quality check stage - requires multi-lead ECG data")
'''
