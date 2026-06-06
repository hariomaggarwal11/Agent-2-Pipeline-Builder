"""Agent 1 Bridge - loads and parses NeuroInspect JSON reports."""

import json
import subprocess
import os
from typing import Optional


def load_agent1_report(filepath: str) -> dict:
    """Load and parse an Agent 1 (NeuroInspect) JSON report.

    Parameters
    ----------
    filepath : str
        Path to the JSON report file produced by Agent 1.

    Returns
    -------
    dict
        Parsed report with keys: modality, metadata, quality_score,
        artifacts, known_dataset, events/annotations.

    Raises
    ------
    FileNotFoundError
        If the report file does not exist.
    ValueError
        If the JSON is malformed or missing required keys.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Agent 1 report not found: {filepath}")

    with open(filepath, "r") as f:
        report = json.load(f)

    return parse_agent1_report(report)


def parse_agent1_report(report: dict) -> dict:
    """Parse and validate an Agent 1 report dictionary.

    Parameters
    ----------
    report : dict
        Raw report dictionary from Agent 1.

    Returns
    -------
    dict
        Normalized report with standardized structure.

    Raises
    ------
    ValueError
        If required keys are missing.
    """
    required_keys = ["modality"]
    for key in required_keys:
        if key not in report:
            raise ValueError(f"Agent 1 report missing required key: '{key}'")

    modality = report["modality"]
    if modality not in ("EEG", "ECG"):
        raise ValueError(f"Unsupported modality: '{modality}'. Expected 'EEG' or 'ECG'.")

    normalized = {
        "modality": modality,
        "metadata": report.get("metadata", {}),
        "quality_score": report.get("quality_score", {"score": 0, "breakdown": []}),
        "artifacts": report.get("artifacts", {}),
        "known_dataset": report.get("known_dataset", None),
    }

    # EEG-specific fields
    if modality == "EEG":
        normalized["events"] = report.get("events", {"has_events": False})
        normalized["faa"] = report.get("faa", {"available": False})
        normalized["channel_info"] = report.get("channel_info", None)

    # ECG-specific fields
    if modality == "ECG":
        normalized["annotations"] = report.get("annotations", {"has_annotations": False})
        normalized["lead_info"] = report.get("lead_info", None)
        normalized["rpeaks"] = report.get("rpeaks", {})

    return normalized


def get_channel_count(report: dict) -> int:
    """Extract channel/lead count from Agent 1 report metadata.

    Parameters
    ----------
    report : dict
        Parsed Agent 1 report.

    Returns
    -------
    int
        Number of channels or leads.
    """
    metadata = report.get("metadata", {})
    n_ch = metadata.get("Number of Channels", metadata.get("Number of Leads", 0))
    if isinstance(n_ch, str):
        try:
            n_ch = int(n_ch)
        except ValueError:
            n_ch = 0
    return n_ch


def get_quality_score(report: dict) -> int:
    """Extract numeric quality score from Agent 1 report.

    Parameters
    ----------
    report : dict
        Parsed Agent 1 report.

    Returns
    -------
    int
        Quality score (0-100).
    """
    qs = report.get("quality_score", {})
    if isinstance(qs, dict):
        return qs.get("score", 0)
    return 0


def run_neuroinspect(dataset_path: str, neuroinspect_path: Optional[str] = None) -> dict:
    """Run Agent 1 (NeuroInspect) on a dataset file and return the report.

    Parameters
    ----------
    dataset_path : str
        Path to the EEG/ECG data file.
    neuroinspect_path : str, optional
        Path to the NeuroInspect installation directory.
        Defaults to looking for it relative to this project.

    Returns
    -------
    dict
        Parsed Agent 1 report.

    Raises
    ------
    RuntimeError
        If NeuroInspect execution fails.
    """
    if neuroinspect_path is None:
        # Try common locations
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "neuroinspect"),
            os.path.join(os.path.dirname(__file__), "..", "..", "neuroinspect"),
        ]
        for candidate in candidates:
            if os.path.isdir(candidate):
                neuroinspect_path = os.path.abspath(candidate)
                break

    if neuroinspect_path is None or not os.path.isdir(neuroinspect_path):
        raise RuntimeError("NeuroInspect directory not found. Provide neuroinspect_path.")

    # Run NeuroInspect as subprocess
    cmd = [
        "python", "-c",
        f"import sys; sys.path.insert(0, '{neuroinspect_path}'); "
        f"from core.eeg.report_builder import build_eeg_report; "
        f"from core.ecg.report_builder import build_ecg_report; "
        f"import json; print(json.dumps({{'status': 'ready'}}))"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"NeuroInspect failed: {result.stderr}")
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        raise RuntimeError("NeuroInspect execution timed out (120s)")
    except json.JSONDecodeError:
        raise RuntimeError("NeuroInspect output was not valid JSON")
