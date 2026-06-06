"""Model Registry - catalogue of all supported architectures with metadata."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config import MODEL_REGISTRY


def get_all_models() -> dict:
    """Get all available model architectures.

    Returns
    -------
    dict
        Complete model registry.
    """
    return MODEL_REGISTRY


def get_model(model_key: str) -> dict:
    """Get a specific model's metadata.

    Parameters
    ----------
    model_key : str
        Key for the model (e.g., 'gat_tcn', 'eegnet').

    Returns
    -------
    dict
        Model metadata including name, description, hyperparameters.

    Raises
    ------
    KeyError
        If model_key is not in the registry.
    """
    if model_key not in MODEL_REGISTRY:
        raise KeyError(
            f"Model '{model_key}' not in registry. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_key]


def get_models_for_task(task: str) -> list[dict]:
    """Get models recommended for a specific task.

    Parameters
    ----------
    task : str
        Task name (e.g., 'emotion_recognition', 'motor_imagery').

    Returns
    -------
    list of dict
        Models recommended for the task, each with a 'key' field added.
    """
    results = []
    for key, model in MODEL_REGISTRY.items():
        if task in model.get("recommended_for", []):
            results.append({"key": key, **model})
    return results


def get_default_hyperparameters(model_key: str) -> dict:
    """Get default hyperparameters for a model.

    Parameters
    ----------
    model_key : str
        Key for the model.

    Returns
    -------
    dict
        Default hyperparameter values.
    """
    model = get_model(model_key)
    return model.get("hyperparameters", {})


def validate_model_for_data(model_key: str, n_channels: int) -> tuple[bool, str]:
    """Check if a model is compatible with the given data dimensions.

    Parameters
    ----------
    model_key : str
        Key for the model.
    n_channels : int
        Number of input channels.

    Returns
    -------
    tuple of (bool, str)
        (is_valid, message)
    """
    model = get_model(model_key)
    min_ch = model.get("min_channels", 1)

    if n_channels < min_ch:
        return False, (
            f"Model '{model_key}' requires at least {min_ch} channels, "
            f"but data has {n_channels}."
        )
    return True, "Compatible"


def list_model_keys() -> list[str]:
    """Get list of all model keys.

    Returns
    -------
    list of str
        Available model keys.
    """
    return list(MODEL_REGISTRY.keys())
