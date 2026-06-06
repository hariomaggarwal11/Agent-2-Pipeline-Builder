"""ECG Stage 4: Feature Extraction - morphological, HRV, and wavelet features."""


def get_config_schema() -> dict:
    """Return the configuration schema for this stage."""
    return {
        "morphological": {
            "type": "bool",
            "default": True,
            "description": "Extract beat morphology features (P-QRS-T amplitudes, durations)",
        },
        "hrv_time_domain": {
            "type": "bool",
            "default": True,
            "description": "HRV time-domain features (SDNN, RMSSD, pNN50)",
        },
        "hrv_frequency_domain": {
            "type": "bool",
            "default": True,
            "description": "HRV frequency-domain features (LF/HF ratio)",
        },
        "hrv_nonlinear": {
            "type": "bool",
            "default": True,
            "description": "HRV nonlinear features (SD1, SD2, ApEn, SampEn)",
        },
        "wavelet_features": {
            "type": "bool",
            "default": True,
            "description": "Wavelet decomposition features",
        },
        "wavelet_family": {
            "type": "str",
            "default": "db4",
            "description": "Wavelet family for decomposition",
            "options": ["db4", "sym4", "coif3", "bior3.3"],
        },
        "wavelet_levels": {
            "type": "int",
            "default": 5,
            "description": "Number of wavelet decomposition levels",
        },
        "statistical_features": {
            "type": "bool",
            "default": True,
            "description": "Statistical features (mean, std, skew, kurtosis per segment)",
        },
    }


def get_default_config(dataset_info: dict) -> dict:
    """Return default configuration based on dataset info."""
    config = {
        "morphological": True,
        "hrv_time_domain": True,
        "hrv_frequency_domain": True,
        "hrv_nonlinear": True,
        "wavelet_features": True,
        "wavelet_family": "db4",
        "wavelet_levels": 5,
        "statistical_features": True,
    }
    return config


def generate_code(config: dict) -> str:
    """Generate Python code for the ECG feature extraction stage."""
    return f'''"""Stage 4: ECG Feature Extraction"""

import numpy as np
from scipy.stats import kurtosis, skew
from scipy.signal import welch
import logging

logger = logging.getLogger(__name__)

# Configuration
MORPHOLOGICAL = {config.get("morphological", True)}
HRV_TIME = {config.get("hrv_time_domain", True)}
HRV_FREQ = {config.get("hrv_frequency_domain", True)}
HRV_NONLINEAR = {config.get("hrv_nonlinear", True)}
WAVELET = {config.get("wavelet_features", True)}
WAVELET_FAMILY = "{config.get("wavelet_family", "db4")}"
WAVELET_LEVELS = {config.get("wavelet_levels", 5)}
STATISTICAL = {config.get("statistical_features", True)}


def extract_hrv_time_domain(rr_intervals: np.ndarray) -> dict:
    """Extract HRV time-domain features.

    Parameters
    ----------
    rr_intervals : np.ndarray
        RR intervals in milliseconds

    Returns
    -------
    dict
        HRV features: SDNN, RMSSD, pNN50, mean_HR
    """
    if len(rr_intervals) < 5:
        return {{"SDNN": 0, "RMSSD": 0, "pNN50": 0, "mean_HR": 0}}

    sdnn = np.std(rr_intervals)
    diff_rr = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(diff_rr ** 2))
    pnn50 = np.sum(np.abs(diff_rr) > 50) / len(diff_rr) * 100
    mean_hr = 60000.0 / np.mean(rr_intervals)

    return {{"SDNN": sdnn, "RMSSD": rmssd, "pNN50": pnn50, "mean_HR": mean_hr}}


def extract_hrv_frequency_domain(rr_intervals: np.ndarray, fs_rr: float = 4.0) -> dict:
    """Extract HRV frequency-domain features."""
    if len(rr_intervals) < 20:
        return {{"LF_power": 0, "HF_power": 0, "LF_HF_ratio": 0}}

    from scipy.interpolate import interp1d
    rr_times = np.cumsum(rr_intervals) / 1000.0
    rr_interp = interp1d(rr_times, rr_intervals, kind="cubic", fill_value="extrapolate")
    t_uniform = np.arange(rr_times[0], rr_times[-1], 1.0 / fs_rr)
    rr_uniform = rr_interp(t_uniform)

    freqs, psd = welch(rr_uniform, fs=fs_rr, nperseg=min(len(rr_uniform), 256))

    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.4)
    lf_power = np.trapz(psd[lf_mask], freqs[lf_mask])
    hf_power = np.trapz(psd[hf_mask], freqs[hf_mask])
    ratio = lf_power / (hf_power + 1e-10)

    return {{"LF_power": lf_power, "HF_power": hf_power, "LF_HF_ratio": ratio}}


def extract_morphological_features(beats: np.ndarray) -> np.ndarray:
    """Extract morphological features from beat waveforms.

    Parameters
    ----------
    beats : np.ndarray
        Shape (n_beats, n_leads, beat_length)

    Returns
    -------
    np.ndarray
        Shape (n_beats, n_features)
    """
    n_beats, n_leads, beat_len = beats.shape
    features_per_lead = 6  # max, min, range, energy, zero_crossings, peak_position
    features = np.zeros((n_beats, n_leads * features_per_lead))

    for i in range(n_beats):
        for lead in range(n_leads):
            beat = beats[i, lead]
            base = lead * features_per_lead
            features[i, base] = np.max(beat)
            features[i, base + 1] = np.min(beat)
            features[i, base + 2] = np.max(beat) - np.min(beat)
            features[i, base + 3] = np.sum(beat ** 2)
            features[i, base + 4] = np.sum(np.diff(np.sign(beat)) != 0)
            features[i, base + 5] = np.argmax(beat) / beat_len

    return features


def extract_statistical_features(beats: np.ndarray) -> np.ndarray:
    """Extract statistical features from beat waveforms."""
    n_beats, n_leads, _ = beats.shape
    features = np.zeros((n_beats, n_leads * 4))

    for i in range(n_beats):
        for lead in range(n_leads):
            base = lead * 4
            features[i, base] = np.mean(beats[i, lead])
            features[i, base + 1] = np.std(beats[i, lead])
            features[i, base + 2] = kurtosis(beats[i, lead])
            features[i, base + 3] = skew(beats[i, lead])

    return features


def extract_all_features(beats: np.ndarray, rr_intervals: np.ndarray = None) -> dict:
    """Extract all configured features."""
    logger.info("Extracting ECG features...")
    features = {{}}

    if MORPHOLOGICAL:
        features["morphological"] = extract_morphological_features(beats)
        logger.info(f"Morphological features: {{features['morphological'].shape}}")

    if STATISTICAL:
        features["statistical"] = extract_statistical_features(beats)
        logger.info(f"Statistical features: {{features['statistical'].shape}}")

    if rr_intervals is not None and HRV_TIME:
        hrv_time = extract_hrv_time_domain(rr_intervals)
        features["hrv_time"] = hrv_time
        logger.info(f"HRV time-domain: {{hrv_time}}")

    if rr_intervals is not None and HRV_FREQ:
        hrv_freq = extract_hrv_frequency_domain(rr_intervals)
        features["hrv_frequency"] = hrv_freq
        logger.info(f"HRV frequency-domain: {{hrv_freq}}")

    # Combine beat-level features
    beat_features = [v for k, v in features.items() if isinstance(v, np.ndarray)]
    if beat_features:
        features["combined"] = np.concatenate(beat_features, axis=1)
        logger.info(f"Combined features shape: {{features['combined'].shape}}")

    return features


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Feature extraction stage - requires segmented beats")
'''
