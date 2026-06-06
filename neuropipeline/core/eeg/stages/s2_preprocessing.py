"""EEG Stage 2: Preprocessing - filtering, artifact removal, rereferencing."""


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
            "default": 45.0,
            "description": "Low-pass filter cutoff (Hz)",
        },
        "notch_filter": {
            "type": "float",
            "default": 50.0,
            "description": "Notch filter frequency (Hz), 0 to disable",
        },
        "reference": {
            "type": "str",
            "default": "average",
            "description": "Reference type: average, linked_mastoids, cz",
            "options": ["average", "linked_mastoids", "cz"],
        },
        "ica_enabled": {
            "type": "bool",
            "default": True,
            "description": "Enable ICA artifact removal",
        },
        "ica_n_components": {
            "type": "int",
            "default": 15,
            "description": "Number of ICA components",
        },
        "ica_method": {
            "type": "str",
            "default": "fastica",
            "description": "ICA algorithm",
            "options": ["fastica", "infomax", "picard"],
        },
        "interpolate_bads": {
            "type": "bool",
            "default": True,
            "description": "Interpolate bad channels",
        },
        "resample_freq": {
            "type": "float",
            "default": 0,
            "description": "Resample frequency (Hz), 0 to keep original",
        },
    }


def get_default_config(dataset_info: dict) -> dict:
    """Return default configuration based on dataset info."""
    metadata = dataset_info.get("metadata", {})
    quality = dataset_info.get("quality_score", {})
    score = quality.get("score", 0) if isinstance(quality, dict) else 0
    n_ch = metadata.get("Number of Channels", 14)
    if isinstance(n_ch, str):
        n_ch = int(n_ch) if n_ch.isdigit() else 14

    config = {
        "bandpass_low": 0.5,
        "bandpass_high": 45.0,
        "notch_filter": 50.0,
        "reference": "average",
        "ica_enabled": score >= 70 and n_ch >= 8,
        "ica_n_components": min(n_ch - 1, 20),
        "ica_method": "fastica",
        "interpolate_bads": True,
        "resample_freq": 0,
    }

    return config


def generate_code(config: dict) -> str:
    """Generate Python code for the preprocessing stage."""
    return f'''"""Stage 2: EEG Preprocessing"""

import numpy as np
import mne
import logging

logger = logging.getLogger(__name__)

# Configuration
BANDPASS_LOW = {config.get("bandpass_low", 0.5)}
BANDPASS_HIGH = {config.get("bandpass_high", 45.0)}
NOTCH_FREQ = {config.get("notch_filter", 50.0)}
REFERENCE = "{config.get("reference", "average")}"
ICA_ENABLED = {config.get("ica_enabled", True)}
ICA_N_COMPONENTS = {config.get("ica_n_components", 15)}
ICA_METHOD = "{config.get("ica_method", "fastica")}"
INTERPOLATE_BADS = {config.get("interpolate_bads", True)}
RESAMPLE_FREQ = {config.get("resample_freq", 0)}


def preprocess(raw: mne.io.Raw) -> mne.io.Raw:
    """Apply preprocessing pipeline to raw EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data.

    Returns
    -------
    mne.io.Raw
        Preprocessed EEG data.
    """
    raw = raw.copy()

    # 1. Resample if requested
    if RESAMPLE_FREQ > 0:
        logger.info(f"Resampling to {{RESAMPLE_FREQ}} Hz")
        raw.resample(RESAMPLE_FREQ)

    # 2. Bandpass filter
    logger.info(f"Applying bandpass filter: {{BANDPASS_LOW}}-{{BANDPASS_HIGH}} Hz")
    raw.filter(l_freq=BANDPASS_LOW, h_freq=BANDPASS_HIGH, fir_design="firwin")

    # 3. Notch filter
    if NOTCH_FREQ > 0:
        logger.info(f"Applying notch filter at {{NOTCH_FREQ}} Hz")
        raw.notch_filter(freqs=NOTCH_FREQ)

    # 4. Interpolate bad channels
    if INTERPOLATE_BADS and raw.info["bads"]:
        logger.info(f"Interpolating bad channels: {{raw.info['bads']}}")
        raw.interpolate_bads(reset_bads=True)

    # 5. Re-reference
    logger.info(f"Setting reference: {{REFERENCE}}")
    if REFERENCE == "average":
        raw.set_eeg_reference("average", projection=True)
        raw.apply_proj()
    elif REFERENCE == "cz":
        raw.set_eeg_reference(["Cz"])

    # 6. ICA artifact removal
    if ICA_ENABLED:
        logger.info(f"Running ICA ({{ICA_METHOD}}, n_components={{ICA_N_COMPONENTS}})")
        ica = mne.preprocessing.ICA(
            n_components=ICA_N_COMPONENTS,
            method=ICA_METHOD,
            max_iter=500,
            random_state=42,
        )
        ica.fit(raw)
        # Auto-detect EOG components
        try:
            eog_indices, eog_scores = ica.find_bads_eog(raw)
            ica.exclude = eog_indices[:2]  # Remove at most 2 EOG components
            logger.info(f"Removing {{len(ica.exclude)}} EOG components")
        except Exception:
            logger.warning("EOG detection failed, skipping ICA exclusion")
        raw = ica.apply(raw)

    logger.info("Preprocessing complete")
    return raw


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Preprocessing stage - requires raw EEG data input")
'''
