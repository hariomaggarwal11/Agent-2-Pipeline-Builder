"""ECG Stage 3: Beat Segmentation - R-peak detection and heartbeat extraction."""


def get_config_schema() -> dict:
    """Return the configuration schema for this stage."""
    return {
        "rpeak_method": {
            "type": "str",
            "default": "neurokit2",
            "description": "R-peak detection algorithm",
            "options": ["neurokit2", "hamilton", "pan_tompkins", "engzee"],
        },
        "beat_window_before": {
            "type": "float",
            "default": 0.25,
            "description": "Seconds before R-peak to include in beat",
        },
        "beat_window_after": {
            "type": "float",
            "default": 0.45,
            "description": "Seconds after R-peak to include in beat",
        },
        "fixed_length": {
            "type": "int",
            "default": 256,
            "description": "Fixed beat length (resample if needed)",
        },
        "lead_for_detection": {
            "type": "str",
            "default": "II",
            "description": "Primary lead for R-peak detection",
        },
        "reject_ectopic": {
            "type": "bool",
            "default": True,
            "description": "Reject ectopic/abnormal RR intervals",
        },
        "ectopic_threshold": {
            "type": "float",
            "default": 0.2,
            "description": "RR deviation threshold for ectopic beat rejection",
        },
    }


def get_default_config(dataset_info: dict) -> dict:
    """Return default configuration based on dataset info."""
    config = {
        "rpeak_method": "neurokit2",
        "beat_window_before": 0.25,
        "beat_window_after": 0.45,
        "fixed_length": 256,
        "lead_for_detection": "II",
        "reject_ectopic": True,
        "ectopic_threshold": 0.2,
    }
    return config


def generate_code(config: dict) -> str:
    """Generate Python code for the beat segmentation stage."""
    return f'''"""Stage 3: ECG Beat Segmentation"""

import numpy as np
from scipy.signal import resample
import logging

logger = logging.getLogger(__name__)

# Configuration
RPEAK_METHOD = "{config.get("rpeak_method", "neurokit2")}"
BEAT_WINDOW_BEFORE = {config.get("beat_window_before", 0.25)}
BEAT_WINDOW_AFTER = {config.get("beat_window_after", 0.45)}
FIXED_LENGTH = {config.get("fixed_length", 256)}
LEAD_FOR_DETECTION = "{config.get("lead_for_detection", "II")}"
REJECT_ECTOPIC = {config.get("reject_ectopic", True)}
ECTOPIC_THRESHOLD = {config.get("ectopic_threshold", 0.2)}


def detect_rpeaks(signal: np.ndarray, fs: float) -> np.ndarray:
    """Detect R-peaks in ECG signal.

    Parameters
    ----------
    signal : np.ndarray
        Single-lead ECG signal
    fs : float
        Sampling frequency

    Returns
    -------
    np.ndarray
        R-peak indices
    """
    try:
        import neurokit2 as nk
        _, info = nk.ecg_peaks(signal, sampling_rate=int(fs), method=RPEAK_METHOD)
        return info["ECG_R_Peaks"]
    except ImportError:
        # Fallback: simple threshold-based detection
        from scipy.signal import find_peaks
        min_distance = int(0.4 * fs)  # minimum 400ms between peaks
        threshold = np.std(signal) * 0.6
        peaks, _ = find_peaks(signal, height=threshold, distance=min_distance)
        return peaks


def segment_beats(signals: np.ndarray, rpeaks: np.ndarray, fs: float) -> np.ndarray:
    """Segment heartbeats around R-peaks.

    Parameters
    ----------
    signals : np.ndarray
        Shape (n_leads, n_samples)
    rpeaks : np.ndarray
        R-peak indices
    fs : float
        Sampling frequency

    Returns
    -------
    np.ndarray
        Shape (n_beats, n_leads, fixed_length)
    """
    before_samples = int(BEAT_WINDOW_BEFORE * fs)
    after_samples = int(BEAT_WINDOW_AFTER * fs)
    beat_length = before_samples + after_samples
    n_leads = signals.shape[0]

    beats = []
    for rpeak in rpeaks:
        start = rpeak - before_samples
        end = rpeak + after_samples
        if start < 0 or end > signals.shape[1]:
            continue

        beat = signals[:, start:end]

        # Resample to fixed length
        if beat.shape[1] != FIXED_LENGTH:
            beat_resampled = np.zeros((n_leads, FIXED_LENGTH))
            for lead in range(n_leads):
                beat_resampled[lead] = resample(beat[lead], FIXED_LENGTH)
            beat = beat_resampled

        beats.append(beat)

    if not beats:
        logger.warning("No valid beats extracted!")
        return np.zeros((0, n_leads, FIXED_LENGTH))

    beats_array = np.array(beats)
    logger.info(f"Extracted {{len(beats)}} beats from {{len(rpeaks)}} R-peaks")
    return beats_array


def reject_ectopic_beats(beats: np.ndarray, rpeaks: np.ndarray, fs: float) -> tuple:
    """Reject ectopic beats based on RR interval deviation."""
    if len(rpeaks) < 3:
        return beats, rpeaks

    rr_intervals = np.diff(rpeaks) / fs
    median_rr = np.median(rr_intervals)

    valid_mask = np.ones(len(beats), dtype=bool)
    for i in range(1, len(rr_intervals)):
        deviation = abs(rr_intervals[i] - median_rr) / median_rr
        if deviation > ECTOPIC_THRESHOLD:
            valid_mask[i + 1] = False  # reject the beat after abnormal RR

    filtered_beats = beats[valid_mask[:len(beats)]]
    logger.info(f"Rejected {{np.sum(~valid_mask[:len(beats)])}}/{{len(beats)}} ectopic beats")
    return filtered_beats, rpeaks[valid_mask[:len(rpeaks)]]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Beat segmentation stage - requires preprocessed ECG data")
'''
