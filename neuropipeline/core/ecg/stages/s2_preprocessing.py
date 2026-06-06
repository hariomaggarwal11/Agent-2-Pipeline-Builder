"""ECG Stage 2: Preprocessing - filtering, baseline correction, noise removal."""


def get_config_schema() -> dict:
    """Return the configuration schema for this stage."""
    return {
        "bandpass_low": {
            "type": "float",
            "default": 0.5,
            "description": "High-pass filter cutoff (Hz)",
        },
        "bandpass_high": {
            "type": "float",
            "default": 40.0,
            "description": "Low-pass filter cutoff (Hz)",
        },
        "notch_filter": {
            "type": "float",
            "default": 50.0,
            "description": "Notch filter frequency (Hz)",
        },
        "baseline_correction": {
            "type": "bool",
            "default": True,
            "description": "Apply baseline wander removal",
        },
        "baseline_method": {
            "type": "str",
            "default": "wavelet",
            "description": "Baseline correction method",
            "options": ["wavelet", "polynomial", "median"],
        },
        "resample_freq": {
            "type": "float",
            "default": 0,
            "description": "Resample frequency (Hz), 0 to keep original",
        },
        "normalize": {
            "type": "bool",
            "default": True,
            "description": "Z-score normalize each lead",
        },
    }


def get_default_config(dataset_info: dict) -> dict:
    """Return default configuration based on dataset info."""
    config = {
        "bandpass_low": 0.5,
        "bandpass_high": 40.0,
        "notch_filter": 50.0,
        "baseline_correction": True,
        "baseline_method": "wavelet",
        "resample_freq": 0,
        "normalize": True,
    }
    return config


def generate_code(config: dict) -> str:
    """Generate Python code for the ECG preprocessing stage."""
    return f'''"""Stage 2: ECG Preprocessing"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
import logging

logger = logging.getLogger(__name__)

# Configuration
BANDPASS_LOW = {config.get("bandpass_low", 0.5)}
BANDPASS_HIGH = {config.get("bandpass_high", 40.0)}
NOTCH_FREQ = {config.get("notch_filter", 50.0)}
BASELINE_CORRECTION = {config.get("baseline_correction", True)}
BASELINE_METHOD = "{config.get("baseline_method", "wavelet")}"
RESAMPLE_FREQ = {config.get("resample_freq", 0)}
NORMALIZE = {config.get("normalize", True)}


def bandpass_filter(signal: np.ndarray, fs: float) -> np.ndarray:
    """Apply bandpass filter."""
    nyq = fs / 2.0
    low = BANDPASS_LOW / nyq
    high = BANDPASS_HIGH / nyq
    b, a = butter(4, [low, high], btype="band")
    return filtfilt(b, a, signal)


def notch_filter(signal: np.ndarray, fs: float) -> np.ndarray:
    """Apply notch filter to remove powerline noise."""
    quality_factor = 30.0
    b, a = iirnotch(NOTCH_FREQ, quality_factor, fs)
    return filtfilt(b, a, signal)


def remove_baseline(signal: np.ndarray, fs: float) -> np.ndarray:
    """Remove baseline wander."""
    if BASELINE_METHOD == "median":
        window = int(0.6 * fs)
        if window % 2 == 0:
            window += 1
        from scipy.signal import medfilt
        baseline = medfilt(signal, kernel_size=window)
        return signal - baseline
    elif BASELINE_METHOD == "polynomial":
        x = np.arange(len(signal))
        coeffs = np.polyfit(x, signal, deg=6)
        baseline = np.polyval(coeffs, x)
        return signal - baseline
    else:  # wavelet
        # Simple high-pass approximation
        nyq = fs / 2.0
        b, a = butter(2, 0.5 / nyq, btype="high")
        return filtfilt(b, a, signal)


def preprocess_ecg(signals: np.ndarray, fs: float) -> np.ndarray:
    """Preprocess multi-lead ECG signals.

    Parameters
    ----------
    signals : np.ndarray
        Shape (n_leads, n_samples)
    fs : float
        Sampling frequency

    Returns
    -------
    np.ndarray
        Preprocessed signals
    """
    n_leads, n_samples = signals.shape
    processed = np.zeros_like(signals)

    for i in range(n_leads):
        sig = signals[i].copy()

        # Bandpass filter
        sig = bandpass_filter(sig, fs)

        # Notch filter
        if NOTCH_FREQ > 0:
            sig = notch_filter(sig, fs)

        # Baseline correction
        if BASELINE_CORRECTION:
            sig = remove_baseline(sig, fs)

        # Normalize
        if NORMALIZE:
            sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)

        processed[i] = sig

    logger.info(f"Preprocessed {{n_leads}} leads, {{n_samples}} samples")
    return processed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("ECG Preprocessing stage - requires multi-lead ECG data")
'''
