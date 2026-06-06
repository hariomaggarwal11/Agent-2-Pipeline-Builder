"""ECG Stage 9: Clinical Interpretation - clinical context and explainability."""


def get_config_schema() -> dict:
    """Return the configuration schema for this stage."""
    return {
        "interpretation_method": {
            "type": "str",
            "default": "gradcam",
            "description": "Attribution method for model interpretation",
            "options": ["gradcam", "shap", "integrated_gradients", "attention"],
        },
        "clinical_guidelines": {
            "type": "bool",
            "default": True,
            "description": "Map predictions to clinical guidelines",
        },
        "generate_ecg_overlay": {
            "type": "bool",
            "default": True,
            "description": "Overlay attributions on ECG waveform",
        },
        "risk_stratification": {
            "type": "bool",
            "default": True,
            "description": "Include risk stratification based on predictions",
        },
        "confidence_threshold": {
            "type": "float",
            "default": 0.7,
            "description": "Minimum confidence for clinical reporting",
        },
        "n_samples": {
            "type": "int",
            "default": 50,
            "description": "Number of samples for interpretation",
        },
    }


def get_default_config(dataset_info: dict) -> dict:
    """Return default configuration based on dataset info."""
    config = {
        "interpretation_method": "gradcam",
        "clinical_guidelines": True,
        "generate_ecg_overlay": True,
        "risk_stratification": True,
        "confidence_threshold": 0.7,
        "n_samples": 50,
    }
    return config


def generate_code(config: dict) -> str:
    """Generate Python code for the clinical interpretation stage."""
    return f'''"""Stage 9: ECG Clinical Interpretation"""

import numpy as np
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

# Configuration
METHOD = "{config.get("interpretation_method", "gradcam")}"
CLINICAL_GUIDELINES = {config.get("clinical_guidelines", True)}
GENERATE_OVERLAY = {config.get("generate_ecg_overlay", True)}
RISK_STRATIFICATION = {config.get("risk_stratification", True)}
CONFIDENCE_THRESHOLD = {config.get("confidence_threshold", 0.7)}
N_SAMPLES = {config.get("n_samples", 50)}

# Clinical label descriptions
CLINICAL_DESCRIPTIONS = {{
    "AF": "Atrial Fibrillation - irregular rhythm, absent P-waves",
    "LBBB": "Left Bundle Branch Block - wide QRS, delayed left ventricular activation",
    "RBBB": "Right Bundle Branch Block - wide QRS, RSR\' pattern in V1-V2",
    "MI": "Myocardial Infarction - ST elevation/depression, Q-waves",
    "PVC": "Premature Ventricular Contraction - wide QRS, compensatory pause",
    "PAC": "Premature Atrial Contraction - early P-wave, narrow QRS",
    "STD": "ST Depression - ischemia indicator",
    "STE": "ST Elevation - acute injury indicator",
    "Normal": "Normal Sinus Rhythm",
    "N": "Normal beat",
}}


def compute_gradcam(model, x, target_layer=None):
    """Compute Grad-CAM attribution for ECG.

    Parameters
    ----------
    model : nn.Module
        Trained model
    x : torch.Tensor
        Input tensor (1, n_leads, seq_length)
    target_layer : nn.Module, optional
        Layer to compute gradients for

    Returns
    -------
    np.ndarray
        Attribution map (n_leads, seq_length)
    """
    model.eval()
    x.requires_grad_(True)

    output = model(x)
    pred_class = output.argmax(dim=1)
    score = output[0, pred_class]
    score.backward()

    # Use input gradients as attribution
    attribution = (x.grad[0] * x[0]).detach().cpu().numpy()
    # Normalize
    attribution = np.abs(attribution)
    attribution = attribution / (attribution.max() + 1e-10)
    return attribution


def interpret_prediction(model, x, class_names=None, device="cpu"):
    """Generate clinical interpretation for a single prediction.

    Parameters
    ----------
    model : nn.Module
        Trained model
    x : np.ndarray
        Single ECG recording (n_leads, seq_length)
    class_names : list
        Class label names
    device : str
        Device

    Returns
    -------
    dict
        Interpretation results
    """
    model.eval()
    x_tensor = torch.FloatTensor(x).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(x_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    pred_class = np.argmax(probs)
    confidence = probs[pred_class]
    pred_label = class_names[pred_class] if class_names else str(pred_class)

    result = {{
        "predicted_class": pred_label,
        "confidence": float(confidence),
        "all_probabilities": {{class_names[i] if class_names else str(i): float(p) for i, p in enumerate(probs)}},
        "high_confidence": confidence >= CONFIDENCE_THRESHOLD,
    }}

    # Attribution
    x_tensor = torch.FloatTensor(x).unsqueeze(0).to(device)
    x_tensor.requires_grad_(True)
    attribution = compute_gradcam(model, x_tensor)
    result["attribution"] = attribution

    # Clinical description
    if CLINICAL_GUIDELINES and pred_label in CLINICAL_DESCRIPTIONS:
        result["clinical_description"] = CLINICAL_DESCRIPTIONS[pred_label]

    # Risk stratification
    if RISK_STRATIFICATION:
        result["risk_level"] = assess_risk(pred_label, confidence)

    return result


def assess_risk(prediction: str, confidence: float) -> str:
    """Assess clinical risk level."""
    high_risk = {{"MI", "STE", "AF"}}
    moderate_risk = {{"LBBB", "RBBB", "STD", "PVC"}}

    if prediction in high_risk and confidence >= CONFIDENCE_THRESHOLD:
        return "HIGH"
    elif prediction in moderate_risk and confidence >= CONFIDENCE_THRESHOLD:
        return "MODERATE"
    elif prediction in {{"Normal", "N", "NORM"}}:
        return "LOW"
    else:
        return "UNCERTAIN"


def batch_interpret(model, X_data, class_names=None, device="cpu"):
    """Run interpretation on multiple samples."""
    logger.info(f"Interpreting {{min(len(X_data), N_SAMPLES)}} samples...")
    results = []
    for i in range(min(len(X_data), N_SAMPLES)):
        result = interpret_prediction(model, X_data[i], class_names, device)
        results.append(result)

    # Summary
    risk_counts = {{"HIGH": 0, "MODERATE": 0, "LOW": 0, "UNCERTAIN": 0}}
    for r in results:
        risk_counts[r.get("risk_level", "UNCERTAIN")] += 1

    logger.info(f"Risk distribution: {{risk_counts}}")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Clinical interpretation stage - requires trained model and ECG data")
'''
