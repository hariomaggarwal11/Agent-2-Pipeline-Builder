"""EEG Stage 3: Feature Extraction - PSD, connectivity, FAA, temporal features."""


def get_config_schema() -> dict:
    """Return the configuration schema for this stage."""
    return {
        "psd_method": {
            "type": "str",
            "default": "welch",
            "description": "PSD estimation method",
            "options": ["welch", "multitaper"],
        },
        "psd_bands": {
            "type": "dict",
            "default": {
                "delta": [0.5, 4.0],
                "theta": [4.0, 8.0],
                "alpha": [8.0, 13.0],
                "beta": [13.0, 30.0],
                "gamma": [30.0, 45.0],
            },
            "description": "Frequency bands for PSD computation",
        },
        "compute_connectivity": {
            "type": "bool",
            "default": True,
            "description": "Compute inter-channel connectivity features",
        },
        "connectivity_method": {
            "type": "str",
            "default": "plv",
            "description": "Connectivity metric",
            "options": ["plv", "coherence", "pli", "wpli"],
        },
        "compute_faa": {
            "type": "bool",
            "default": False,
            "description": "Compute Frontal Alpha Asymmetry",
        },
        "faa_pairs": {
            "type": "list",
            "default": [["F3", "F4"], ["F7", "F8"]],
            "description": "Channel pairs for FAA computation",
        },
        "temporal_features": {
            "type": "bool",
            "default": True,
            "description": "Extract temporal domain features (variance, kurtosis, etc.)",
        },
        "epoch_length": {
            "type": "float",
            "default": 4.0,
            "description": "Epoch length in seconds for feature computation",
        },
    }


def get_default_config(dataset_info: dict) -> dict:
    """Return default configuration based on dataset info."""
    metadata = dataset_info.get("metadata", {})
    n_ch = metadata.get("Number of Channels", 14)
    if isinstance(n_ch, str):
        n_ch = int(n_ch) if n_ch.isdigit() else 14

    known = dataset_info.get("known_dataset", None)
    task = ""
    if known and isinstance(known, dict):
        task = known.get("task", "")

    config = {
        "psd_method": "welch",
        "psd_bands": {
            "delta": [0.5, 4.0],
            "theta": [4.0, 8.0],
            "alpha": [8.0, 13.0],
            "beta": [13.0, 30.0],
            "gamma": [30.0, 45.0],
        },
        "compute_connectivity": n_ch >= 8,
        "connectivity_method": "plv",
        "compute_faa": task == "emotion_recognition",
        "faa_pairs": [["F3", "F4"], ["F7", "F8"], ["FC5", "FC6"]],
        "temporal_features": True,
        "epoch_length": 4.0,
    }

    return config


def generate_code(config: dict) -> str:
    """Generate Python code for the feature extraction stage."""
    return f'''"""Stage 3: EEG Feature Extraction"""

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, skew
import logging

logger = logging.getLogger(__name__)

# Configuration
PSD_METHOD = "{config.get("psd_method", "welch")}"
PSD_BANDS = {config.get("psd_bands", {"delta": [0.5, 4.0], "theta": [4.0, 8.0], "alpha": [8.0, 13.0], "beta": [13.0, 30.0], "gamma": [30.0, 45.0]})}
COMPUTE_CONNECTIVITY = {config.get("compute_connectivity", True)}
CONNECTIVITY_METHOD = "{config.get("connectivity_method", "plv")}"
COMPUTE_FAA = {config.get("compute_faa", False)}
FAA_PAIRS = {config.get("faa_pairs", [["F3", "F4"], ["F7", "F8"]])}
TEMPORAL_FEATURES = {config.get("temporal_features", True)}
EPOCH_LENGTH = {config.get("epoch_length", 4.0)}


def compute_band_power(data: np.ndarray, fs: float, band: tuple) -> float:
    """Compute power in a frequency band."""
    freqs, psd = welch(data, fs=fs, nperseg=min(len(data), int(fs * 2)))
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.trapz(psd[idx], freqs[idx])


def extract_psd_features(epochs: np.ndarray, fs: float) -> np.ndarray:
    """Extract PSD band power features for all epochs.

    Parameters
    ----------
    epochs : np.ndarray
        Shape (n_epochs, n_channels, n_samples)
    fs : float
        Sampling frequency

    Returns
    -------
    np.ndarray
        Shape (n_epochs, n_channels * n_bands)
    """
    n_epochs, n_channels, n_samples = epochs.shape
    n_bands = len(PSD_BANDS)
    features = np.zeros((n_epochs, n_channels * n_bands))

    for e in range(n_epochs):
        for ch in range(n_channels):
            for b_idx, (band_name, band_range) in enumerate(PSD_BANDS.items()):
                power = compute_band_power(epochs[e, ch], fs, tuple(band_range))
                features[e, ch * n_bands + b_idx] = power

    return features


def compute_faa(epochs: np.ndarray, fs: float, ch_names: list) -> np.ndarray:
    """Compute Frontal Alpha Asymmetry."""
    alpha_band = PSD_BANDS.get("alpha", [8.0, 13.0])
    n_epochs = epochs.shape[0]
    faa_features = []

    for pair in FAA_PAIRS:
        left_name, right_name = pair
        if left_name in ch_names and right_name in ch_names:
            left_idx = ch_names.index(left_name)
            right_idx = ch_names.index(right_name)
            for e in range(n_epochs):
                left_power = compute_band_power(epochs[e, left_idx], fs, tuple(alpha_band))
                right_power = compute_band_power(epochs[e, right_idx], fs, tuple(alpha_band))
                asymmetry = np.log(right_power + 1e-10) - np.log(left_power + 1e-10)
                faa_features.append(asymmetry)

    return np.array(faa_features).reshape(n_epochs, -1) if faa_features else np.zeros((n_epochs, 0))


def extract_temporal_features(epochs: np.ndarray) -> np.ndarray:
    """Extract temporal domain features (mean, std, kurtosis, skewness)."""
    n_epochs, n_channels, _ = epochs.shape
    features = np.zeros((n_epochs, n_channels * 4))

    for e in range(n_epochs):
        for ch in range(n_channels):
            base = ch * 4
            features[e, base] = np.mean(epochs[e, ch])
            features[e, base + 1] = np.std(epochs[e, ch])
            features[e, base + 2] = kurtosis(epochs[e, ch])
            features[e, base + 3] = skew(epochs[e, ch])

    return features


def extract_all_features(epochs: np.ndarray, fs: float, ch_names: list = None) -> dict:
    """Extract all configured features."""
    logger.info("Extracting features...")
    features = {{}}

    # PSD features
    features["psd"] = extract_psd_features(epochs, fs)
    logger.info(f"PSD features shape: {{features['psd'].shape}}")

    # Temporal features
    if TEMPORAL_FEATURES:
        features["temporal"] = extract_temporal_features(epochs)
        logger.info(f"Temporal features shape: {{features['temporal'].shape}}")

    # FAA features
    if COMPUTE_FAA and ch_names:
        features["faa"] = compute_faa(epochs, fs, ch_names)
        logger.info(f"FAA features shape: {{features['faa'].shape}}")

    # Concatenate all features
    all_feats = [v for v in features.values() if v.shape[1] > 0]
    features["combined"] = np.concatenate(all_feats, axis=1) if all_feats else features["psd"]
    logger.info(f"Combined features shape: {{features['combined'].shape}}")

    return features


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Feature extraction stage - requires epoched EEG data input")
'''
