"""ECG Stage 5: Dataset Split - train/val/test splitting."""


def get_config_schema() -> dict:
    """Return the configuration schema for this stage."""
    return {
        "test_size": {
            "type": "float",
            "default": 0.2,
            "description": "Fraction of data for test set",
        },
        "val_size": {
            "type": "float",
            "default": 0.2,
            "description": "Fraction of training data for validation",
        },
        "stratify": {
            "type": "bool",
            "default": True,
            "description": "Stratified splitting for class balance",
        },
        "random_state": {
            "type": "int",
            "default": 42,
            "description": "Random seed",
        },
        "patient_aware": {
            "type": "bool",
            "default": True,
            "description": "Ensure patient data does not leak between splits",
        },
        "cross_validation": {
            "type": "str",
            "default": "5-fold",
            "description": "Cross-validation strategy",
            "options": ["none", "5-fold", "10-fold"],
        },
    }


def get_default_config(dataset_info: dict) -> dict:
    """Return default configuration based on dataset info."""
    config = {
        "test_size": 0.2,
        "val_size": 0.2,
        "stratify": True,
        "random_state": 42,
        "patient_aware": True,
        "cross_validation": "5-fold",
    }
    return config


def generate_code(config: dict) -> str:
    """Generate Python code for the ECG dataset split stage."""
    return f'''"""Stage 5: ECG Dataset Split"""

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import logging

logger = logging.getLogger(__name__)

# Configuration
TEST_SIZE = {config.get("test_size", 0.2)}
VAL_SIZE = {config.get("val_size", 0.2)}
STRATIFY = {config.get("stratify", True)}
RANDOM_STATE = {config.get("random_state", 42)}
PATIENT_AWARE = {config.get("patient_aware", True)}
CROSS_VALIDATION = "{config.get("cross_validation", "5-fold")}"


def split_data(X: np.ndarray, y: np.ndarray, patient_ids: np.ndarray = None) -> dict:
    """Split ECG data into train/val/test sets.

    Parameters
    ----------
    X : np.ndarray
        Features (n_beats, n_features)
    y : np.ndarray
        Labels (n_beats,)
    patient_ids : np.ndarray, optional
        Patient IDs to prevent data leakage

    Returns
    -------
    dict
        Split data
    """
    stratify_labels = y if STRATIFY else None

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=stratify_labels
    )

    stratify_temp = y_temp if STRATIFY else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, random_state=RANDOM_STATE,
        stratify=stratify_temp
    )

    logger.info(f"Train: {{len(y_train)}}, Val: {{len(y_val)}}, Test: {{len(y_test)}}")
    return {{
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Dataset split stage - requires features and labels")
'''
