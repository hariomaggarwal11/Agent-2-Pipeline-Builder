"""Report Writer - PDF report generation for pipeline runs using reportlab."""

import io
import json
from typing import Optional


def generate_pipeline_report(
    pipeline_config: dict,
    execution_results: list[dict],
    metrics: Optional[dict] = None,
    output_path: Optional[str] = None,
) -> bytes:
    """Generate a PDF report for a pipeline run.

    Parameters
    ----------
    pipeline_config : dict
        Pipeline configuration used.
    execution_results : list of dict
        Results from each stage execution.
    metrics : dict, optional
        Model evaluation metrics (accuracy, f1, etc.).
    output_path : str, optional
        If provided, saves PDF to this path.

    Returns
    -------
    bytes
        PDF content as bytes.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch, cm
        from reportlab.lib.colors import HexColor
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
        )

        return _build_pdf(pipeline_config, execution_results, metrics, output_path)

    except ImportError:
        # Fallback: return JSON report if reportlab unavailable
        return _build_json_report(pipeline_config, execution_results, metrics, output_path)


def _build_pdf(
    pipeline_config: dict,
    execution_results: list[dict],
    metrics: Optional[dict],
    output_path: Optional[str],
) -> bytes:
    """Build PDF report using reportlab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        textColor=HexColor("#00d4ff"),
    )
    elements.append(Paragraph("NeuroPipeline - Execution Report", title_style))
    elements.append(Spacer(1, 0.3 * inch))

    # Pipeline Configuration section
    elements.append(Paragraph("Pipeline Configuration", styles["Heading2"]))
    elements.append(Spacer(1, 0.1 * inch))

    config_data = [
        ["Parameter", "Value"],
        ["Modality", pipeline_config.get("modality", "N/A")],
        ["Task", pipeline_config.get("task", "N/A")],
        ["Model", pipeline_config.get("model", "N/A")],
        ["Dataset", pipeline_config.get("dataset", "N/A")],
    ]
    config_table = Table(config_data, colWidths=[2.5 * inch, 4 * inch])
    config_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a2235")),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#f1f5f9")),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#2d3f5e")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    elements.append(config_table)
    elements.append(Spacer(1, 0.3 * inch))

    # Execution Results section
    elements.append(Paragraph("Execution Results", styles["Heading2"]))
    elements.append(Spacer(1, 0.1 * inch))

    results_data = [["Stage", "Status", "Duration (s)"]]
    for result in execution_results:
        stage = result.get("stage_name", "Unknown")
        status = result.get("status", "unknown")
        duration = result.get("duration", None)
        dur_str = f"{duration:.2f}" if duration else "N/A"
        results_data.append([stage, status, dur_str])

    results_table = Table(results_data, colWidths=[3 * inch, 1.5 * inch, 2 * inch])
    results_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a2235")),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#f1f5f9")),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#2d3f5e")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    elements.append(results_table)
    elements.append(Spacer(1, 0.3 * inch))

    # Metrics section (if available)
    if metrics:
        elements.append(Paragraph("Model Metrics", styles["Heading2"]))
        elements.append(Spacer(1, 0.1 * inch))

        metrics_data = [["Metric", "Value"]]
        for key, value in metrics.items():
            if isinstance(value, float):
                metrics_data.append([key, f"{value:.4f}"])
            else:
                metrics_data.append([key, str(value)])

        metrics_table = Table(metrics_data, colWidths=[3 * inch, 3.5 * inch])
        metrics_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a2235")),
            ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#f1f5f9")),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#2d3f5e")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]))
        elements.append(metrics_table)

    # Build PDF
    doc.build(elements)
    pdf_bytes = buffer.getvalue()

    if output_path:
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)

    return pdf_bytes


def _build_json_report(
    pipeline_config: dict,
    execution_results: list[dict],
    metrics: Optional[dict],
    output_path: Optional[str],
) -> bytes:
    """Fallback: build JSON report if reportlab is unavailable."""
    report = {
        "title": "NeuroPipeline - Execution Report",
        "pipeline_config": pipeline_config,
        "execution_results": execution_results,
        "metrics": metrics or {},
    }

    content = json.dumps(report, indent=2, default=str).encode("utf-8")

    if output_path:
        with open(output_path, "wb") as f:
            f.write(content)

    return content
