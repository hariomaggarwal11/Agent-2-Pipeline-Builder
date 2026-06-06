"""EEG Stage 7: Evaluation - model evaluation with comprehensive metrics."""


def get_config_schema() -> dict:
    """Return the configuration schema for this stage."""
    return {
        "metrics": {
            "type": "list",
            "default": ["accuracy", "f1_weighted", "precision", "recall", "confusion_matrix"],
            "description": "Evaluation metrics to compute",
        },
        "generate_plots": {
            "type": "bool",
            "default": True,
            "description": "Generate evaluation plots (confusion matrix, ROC, etc.)",
        },
        "per_class_metrics": {
            "type": "bool",
            "default": True,
            "description": "Report per-class metrics",
        },
        "confidence_intervals": {
            "type": "bool",
            "default": True,
            "description": "Compute 95% confidence intervals via bootstrap",
        },
        "n_bootstrap": {
            "type": "int",
            "default": 1000,
            "description": "Number of bootstrap iterations for CI",
        },
    }


def get_default_config(dataset_info: dict) -> dict:
    """Return default configuration based on dataset info."""
    config = {
        "metrics": ["accuracy", "f1_weighted", "precision", "recall", "confusion_matrix"],
        "generate_plots": True,
        "per_class_metrics": True,
        "confidence_intervals": True,
        "n_bootstrap": 1000,
    }
    return config


def generate_code(config: dict) -> str:
    """Generate Python code for the evaluation stage."""
    return f'''"""Stage 7: EEG Model Evaluation"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
import logging

logger = logging.getLogger(__name__)

# Configuration
METRICS = {config.get("metrics", ["accuracy", "f1_weighted", "precision", "recall", "confusion_matrix"])}
GENERATE_PLOTS = {config.get("generate_plots", True)}
PER_CLASS = {config.get("per_class_metrics", True)}
CONFIDENCE_INTERVALS = {config.get("confidence_intervals", True)}
N_BOOTSTRAP = {config.get("n_bootstrap", 1000)}


def evaluate_model(model, X_test, y_test, device="cpu"):
    """Evaluate model on test set.

    Parameters
    ----------
    model : nn.Module
        Trained model
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    device : str
        Device

    Returns
    -------
    dict
        Evaluation results with all metrics
    """
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        outputs = model(X_tensor)
        _, predictions = torch.max(outputs, 1)
        predictions = predictions.cpu().numpy()
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

    results = {{}}

    # Core metrics
    results["accuracy"] = accuracy_score(y_test, predictions)
    results["f1_weighted"] = f1_score(y_test, predictions, average="weighted")
    results["precision_weighted"] = precision_score(y_test, predictions, average="weighted")
    results["recall_weighted"] = recall_score(y_test, predictions, average="weighted")
    results["confusion_matrix"] = confusion_matrix(y_test, predictions).tolist()

    # Per-class metrics
    if PER_CLASS:
        results["classification_report"] = classification_report(
            y_test, predictions, output_dict=True
        )

    # Confidence intervals via bootstrap
    if CONFIDENCE_INTERVALS:
        results["ci_95"] = compute_bootstrap_ci(y_test, predictions)

    # Log results
    logger.info(f"Accuracy: {{results['accuracy']:.4f}}")
    logger.info(f"F1 (weighted): {{results['f1_weighted']:.4f}}")
    logger.info(f"Precision: {{results['precision_weighted']:.4f}}")
    logger.info(f"Recall: {{results['recall_weighted']:.4f}}")

    return results


def compute_bootstrap_ci(y_true, y_pred, n_bootstrap=N_BOOTSTRAP, alpha=0.05):
    """Compute 95% confidence intervals via bootstrap."""
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        score = accuracy_score(y_true[idx], y_pred[idx])
        scores.append(score)
    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    return {{"lower": lower, "upper": upper, "mean": np.mean(scores)}}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Evaluation stage - requires trained model and test data")
'''
