"""ECG Stage 8: Evaluation - model evaluation with ECG-specific metrics."""


def get_config_schema() -> dict:
    """Return the configuration schema for this stage."""
    return {
        "metrics": {
            "type": "list",
            "default": ["accuracy", "f1_weighted", "sensitivity", "specificity", "auc"],
            "description": "Evaluation metrics",
        },
        "per_class_metrics": {
            "type": "bool",
            "default": True,
            "description": "Report per-class metrics",
        },
        "generate_roc": {
            "type": "bool",
            "default": True,
            "description": "Generate ROC curves",
        },
        "confidence_intervals": {
            "type": "bool",
            "default": True,
            "description": "Compute 95% CI via bootstrap",
        },
        "clinical_metrics": {
            "type": "bool",
            "default": True,
            "description": "Include clinical-relevant metrics (PPV, NPV)",
        },
    }


def get_default_config(dataset_info: dict) -> dict:
    """Return default configuration based on dataset info."""
    config = {
        "metrics": ["accuracy", "f1_weighted", "sensitivity", "specificity", "auc"],
        "per_class_metrics": True,
        "generate_roc": True,
        "confidence_intervals": True,
        "clinical_metrics": True,
    }
    return config


def generate_code(config: dict) -> str:
    """Generate Python code for the ECG evaluation stage."""
    return f'''"""Stage 8: ECG Model Evaluation"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
import logging

logger = logging.getLogger(__name__)

# Configuration
METRICS = {config.get("metrics", ["accuracy", "f1_weighted", "sensitivity", "specificity", "auc"])}
PER_CLASS = {config.get("per_class_metrics", True)}
GENERATE_ROC = {config.get("generate_roc", True)}
CONFIDENCE_INTERVALS = {config.get("confidence_intervals", True)}
CLINICAL_METRICS = {config.get("clinical_metrics", True)}


def evaluate_model(model, X_test, y_test, device="cpu"):
    """Evaluate ECG model.

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
        Comprehensive evaluation results
    """
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        outputs = model(X_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        predictions = np.argmax(probabilities, axis=1)

    results = {{}}
    results["accuracy"] = accuracy_score(y_test, predictions)
    results["f1_weighted"] = f1_score(y_test, predictions, average="weighted")
    results["sensitivity"] = recall_score(y_test, predictions, average="macro")
    results["specificity"] = compute_specificity(y_test, predictions)
    results["confusion_matrix"] = confusion_matrix(y_test, predictions).tolist()

    # AUC
    try:
        if len(np.unique(y_test)) == 2:
            results["auc"] = roc_auc_score(y_test, probabilities[:, 1])
        else:
            results["auc"] = roc_auc_score(y_test, probabilities, multi_class="ovr", average="macro")
    except ValueError:
        results["auc"] = None

    # Clinical metrics
    if CLINICAL_METRICS:
        results["ppv"] = precision_score(y_test, predictions, average="macro")
        results["npv"] = compute_npv(y_test, predictions)

    if PER_CLASS:
        results["classification_report"] = classification_report(y_test, predictions, output_dict=True)

    logger.info(f"Accuracy: {{results['accuracy']:.4f}}, F1: {{results['f1_weighted']:.4f}}")
    return results


def compute_specificity(y_true, y_pred):
    """Compute macro-averaged specificity."""
    cm = confusion_matrix(y_true, y_pred)
    specificities = []
    for i in range(cm.shape[0]):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        spec = tn / (tn + fp + 1e-10)
        specificities.append(spec)
    return np.mean(specificities)


def compute_npv(y_true, y_pred):
    """Compute macro-averaged Negative Predictive Value."""
    cm = confusion_matrix(y_true, y_pred)
    npvs = []
    for i in range(cm.shape[0]):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fn = cm[i, :].sum() - cm[i, i]
        npv = tn / (tn + fn + 1e-10)
        npvs.append(npv)
    return np.mean(npvs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Evaluation stage - requires trained model and test data")
'''
