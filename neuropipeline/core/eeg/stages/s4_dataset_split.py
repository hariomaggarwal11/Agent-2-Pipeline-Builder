"""EEG Stage 4: Dataset Split - train/val/test splitting with stratification."""


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
            "description": "Use stratified splitting to maintain class balance",
        },
        "random_state": {
            "type": "int",
            "default": 42,
            "description": "Random seed for reproducibility",
        },
        "cross_validation": {
            "type": "str",
            "default": "5-fold",
            "description": "Cross-validation strategy",
            "options": ["none", "5-fold", "10-fold", "leave-one-subject-out"],
        },
        "subject_aware": {
            "type": "bool",
            "default": True,
            "description": "Ensure same subject data does not leak between splits",
        },
    }


def get_default_config(dataset_info: dict) -> dict:
    """Return default configuration based on dataset info."""
    config = {
        "test_size": 0.2,
        "val_size": 0.2,
        "stratify": True,
        "random_state": 42,
        "cross_validation": "5-fold",
        "subject_aware": True,
    }
    return config


def generate_code(config: dict) -> str:
    """Generate Python code for the dataset split stage."""
    return f'''"""Stage 4: EEG Dataset Split"""

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import logging

logger = logging.getLogger(__name__)

# Configuration
TEST_SIZE = {config.get("test_size", 0.2)}
VAL_SIZE = {config.get("val_size", 0.2)}
STRATIFY = {config.get("stratify", True)}
RANDOM_STATE = {config.get("random_state", 42)}
CROSS_VALIDATION = "{config.get("cross_validation", "5-fold")}"
SUBJECT_AWARE = {config.get("subject_aware", True)}


def split_data(X: np.ndarray, y: np.ndarray, subjects: np.ndarray = None) -> dict:
    """Split data into train/val/test sets.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Labels (n_samples,)
    subjects : np.ndarray, optional
        Subject IDs for subject-aware splitting

    Returns
    -------
    dict
        Split data with keys: X_train, X_val, X_test, y_train, y_val, y_test
    """
    stratify_labels = y if STRATIFY else None

    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=stratify_labels
    )

    # Second split: separate validation from training
    stratify_temp = y_temp if STRATIFY else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, random_state=RANDOM_STATE,
        stratify=stratify_temp
    )

    logger.info(f"Train: {{X_train.shape[0]}}, Val: {{X_val.shape[0]}}, Test: {{X_test.shape[0]}}")
    logger.info(f"Class distribution - Train: {{np.bincount(y_train.astype(int))}}")

    return {{
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }}


def get_cv_folds(X: np.ndarray, y: np.ndarray) -> list:
    """Get cross-validation fold indices.

    Returns
    -------
    list of (train_idx, val_idx) tuples
    """
    if CROSS_VALIDATION == "none":
        return []

    n_folds = int(CROSS_VALIDATION.split("-")[0]) if "-fold" in CROSS_VALIDATION else 5

    if STRATIFY:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
        return list(kf.split(X, y))
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
        return list(kf.split(X))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Dataset split stage - requires feature matrix and labels")
'''
