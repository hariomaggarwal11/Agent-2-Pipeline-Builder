"""EEG Stage 8: Interpretation - model explainability with SHAP and attention maps."""


def get_config_schema() -> dict:
    """Return the configuration schema for this stage."""
    return {
        "method": {
            "type": "str",
            "default": "shap",
            "description": "Interpretation method",
            "options": ["shap", "gradcam", "attention", "integrated_gradients"],
        },
        "n_samples": {
            "type": "int",
            "default": 100,
            "description": "Number of samples for SHAP explanation",
        },
        "plot_topk": {
            "type": "int",
            "default": 20,
            "description": "Number of top features to plot",
        },
        "generate_report": {
            "type": "bool",
            "default": True,
            "description": "Generate interpretation report with visualizations",
        },
        "channel_importance": {
            "type": "bool",
            "default": True,
            "description": "Compute per-channel importance scores",
        },
        "temporal_importance": {
            "type": "bool",
            "default": True,
            "description": "Compute temporal importance (which time points matter)",
        },
    }


def get_default_config(dataset_info: dict) -> dict:
    """Return default configuration based on dataset info."""
    config = {
        "method": "shap",
        "n_samples": 100,
        "plot_topk": 20,
        "generate_report": True,
        "channel_importance": True,
        "temporal_importance": True,
    }
    return config


def generate_code(config: dict) -> str:
    """Generate Python code for the interpretation stage."""
    return f'''"""Stage 8: EEG Model Interpretation"""

import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

# Configuration
METHOD = "{config.get("method", "shap")}"
N_SAMPLES = {config.get("n_samples", 100)}
PLOT_TOPK = {config.get("plot_topk", 20)}
GENERATE_REPORT = {config.get("generate_report", True)}
CHANNEL_IMPORTANCE = {config.get("channel_importance", True)}
TEMPORAL_IMPORTANCE = {config.get("temporal_importance", True)}


def compute_shap_values(model, X_background, X_explain, device="cpu"):
    """Compute SHAP values for model predictions.

    Parameters
    ----------
    model : nn.Module
        Trained model
    X_background : np.ndarray
        Background data for SHAP (subset of training data)
    X_explain : np.ndarray
        Data to explain
    device : str
        Device

    Returns
    -------
    np.ndarray
        SHAP values for each feature
    """
    try:
        import shap

        model.eval()
        model = model.to(device)

        def model_predict(x):
            with torch.no_grad():
                tensor = torch.FloatTensor(x).to(device)
                outputs = model(tensor)
                return torch.softmax(outputs, dim=1).cpu().numpy()

        explainer = shap.KernelExplainer(model_predict, X_background[:50])
        shap_values = explainer.shap_values(X_explain[:N_SAMPLES])
        return shap_values

    except ImportError:
        logger.warning("SHAP not available, using gradient-based attribution")
        return compute_gradient_attribution(model, X_explain, device)


def compute_gradient_attribution(model, X_data, device="cpu"):
    """Compute gradient-based feature attribution."""
    model.eval()
    model = model.to(device)

    X_tensor = torch.FloatTensor(X_data[:N_SAMPLES]).to(device)
    X_tensor.requires_grad_(True)

    outputs = model(X_tensor)
    target_class = outputs.argmax(dim=1)

    attributions = []
    for i in range(len(X_tensor)):
        model.zero_grad()
        outputs[i, target_class[i]].backward(retain_graph=True)
        attr = X_tensor.grad[i].cpu().numpy()
        attributions.append(attr)

    return np.array(attributions)


def get_channel_importance(attributions: np.ndarray, ch_names: list = None) -> dict:
    """Compute per-channel importance from attributions."""
    # attributions shape: (n_samples, n_channels, seq_length) or (n_samples, n_features)
    if attributions.ndim == 3:
        importance = np.mean(np.abs(attributions), axis=(0, 2))
    else:
        n_channels = len(ch_names) if ch_names else attributions.shape[1]
        importance = np.mean(np.abs(attributions), axis=0)[:n_channels]

    importance = importance / (importance.sum() + 1e-10)

    result = {{}}
    names = ch_names or [f"Ch_{{i}}" for i in range(len(importance))]
    for name, imp in zip(names, importance):
        result[name] = float(imp)

    return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))


def interpret_model(model, X_train, X_test, ch_names=None, device="cpu"):
    """Run full model interpretation pipeline."""
    logger.info(f"Running interpretation with method: {{METHOD}}")

    results = {{}}

    # Compute attributions
    if METHOD == "shap":
        attributions = compute_shap_values(model, X_train, X_test, device)
    else:
        attributions = compute_gradient_attribution(model, X_test, device)

    results["attributions"] = attributions

    # Channel importance
    if CHANNEL_IMPORTANCE:
        results["channel_importance"] = get_channel_importance(attributions, ch_names)
        logger.info(f"Top channels: {{list(results['channel_importance'].items())[:5]}}")

    logger.info("Interpretation complete")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Interpretation stage - requires trained model and data")
'''
