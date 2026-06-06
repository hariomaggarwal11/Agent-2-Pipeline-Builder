"""Reusable UI components for NeuroPipeline."""


def pipeline_node_card(stage_name, status, stage_num):
    """Render a styled pipeline node card as HTML.

    Args:
        stage_name: Display name of the pipeline stage.
        status: One of 'done', 'running', 'failed', 'pending', 'skipped'.
        stage_num: Integer stage number.

    Returns:
        HTML string for the node card.
    """
    status_icons = {
        "done": "&#10003;",
        "running": "&#9654;",
        "failed": "&#10007;",
        "pending": "&#9679;",
        "skipped": "&#8212;",
    }
    icon = status_icons.get(status, "&#9679;")
    active_class = "active" if status == "running" else ""

    return f"""
<div class="pipeline-node {active_class}">
    <div class="node-number">Stage {stage_num}</div>
    <div class="node-name">{stage_name}</div>
    <div class="node-status">
        <span class="status-{status}">{icon} {status.capitalize()}</span>
    </div>
</div>
"""


def status_badge(status):
    """Render a status badge as HTML.

    Args:
        status: One of 'done', 'running', 'failed', 'pending', 'skipped'.

    Returns:
        HTML string for the badge.
    """
    labels = {
        "done": "Done",
        "running": "Running",
        "failed": "Failed",
        "pending": "Pending",
        "skipped": "Skipped",
    }
    label = labels.get(status, status.capitalize())
    return f'<span class="status-{status}">{label}</span>'


def code_block(code, language="python"):
    """Return formatted code display HTML.

    Args:
        code: The source code string.
        language: Programming language for syntax context.

    Returns:
        HTML string wrapping the code in a styled panel.
    """
    escaped_code = (
        code.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return f"""
<div class="code-panel">
<pre><code>{escaped_code}</code></pre>
</div>
"""


def metric_card(label, value, delta=None):
    """Render a styled metric display card as HTML.

    Args:
        label: Metric label text.
        value: Metric value (will be converted to string).
        delta: Optional delta value (positive or negative).

    Returns:
        HTML string for the metric card.
    """
    delta_html = ""
    if delta is not None:
        delta_class = "positive" if delta >= 0 else "negative"
        sign = "+" if delta >= 0 else ""
        delta_html = (
            f'<div class="metric-delta {delta_class}">'
            f"{sign}{delta}</div>"
        )

    return f"""
<div class="metric-card">
    <div class="metric-label">{label}</div>
    <div class="metric-value">{value}</div>
    {delta_html}
</div>
"""


def stage_connection_arrow():
    """Render a visual arrow connector between pipeline nodes.

    Returns:
        HTML string for the arrow.
    """
    return '<div class="stage-arrow">&#8595;</div>'


def progress_indicator(current, total):
    """Render a progress bar as HTML.

    Args:
        current: Current progress count.
        total: Total count.

    Returns:
        HTML string for the progress bar.
    """
    if total == 0:
        pct = 0
    else:
        pct = int((current / total) * 100)

    return f"""
<div style="display: flex; align-items: center; gap: 10px;">
    <div class="progress-container" style="flex: 1;">
        <div class="progress-fill" style="width: {pct}%;"></div>
    </div>
    <span style="color: #f1f5f9; font-family: 'JetBrains Mono', monospace; font-size: 0.85em;">
        {current}/{total}
    </span>
</div>
"""
