"""Results dashboard - display model evaluation metrics and visualizations."""


def render_results(results_dict):
    """Render the results dashboard with metrics and plots.

    Displays:
    - Key metrics (accuracy, F1, AUC-ROC)
    - Confusion matrix plot (plotly)
    - ROC curve
    - Per-subject accuracy bar chart
    - Training/validation loss curves
    - Classification report table
    - ECG clinical metrics panel (sensitivity, specificity, PPV, NPV)

    Args:
        results_dict: Dictionary containing evaluation results with keys like
            'accuracy', 'f1_score', 'auc_roc', 'confusion_matrix',
            'roc_data', 'per_subject_accuracy', 'loss_history',
            'classification_report', 'clinical_metrics'.
    """
    import streamlit as st

    from neuropipeline.ui.components import metric_card

    if not results_dict:
        st.info("No results available yet. Run the evaluation stage first.")
        return

    st.markdown('<h2>Evaluation Results</h2>', unsafe_allow_html=True)

    # Key metrics row
    _render_metrics_row(results_dict)

    st.markdown("---")

    # Plots in tabs
    tab_names = ["Confusion Matrix", "ROC Curve", "Loss Curves", "Per-Subject"]
    if results_dict.get("clinical_metrics"):
        tab_names.append("Clinical Metrics")

    tabs = st.tabs(tab_names)

    with tabs[0]:
        _render_confusion_matrix(results_dict.get("confusion_matrix"))

    with tabs[1]:
        _render_roc_curve(results_dict.get("roc_data"))

    with tabs[2]:
        _render_loss_curves(results_dict.get("loss_history"))

    with tabs[3]:
        _render_per_subject_accuracy(results_dict.get("per_subject_accuracy"))

    if results_dict.get("clinical_metrics"):
        with tabs[4]:
            _render_clinical_metrics(results_dict["clinical_metrics"])

    # Classification report table
    if results_dict.get("classification_report"):
        st.markdown("---")
        _render_classification_report(results_dict["classification_report"])


def _render_metrics_row(results_dict):
    """Render the top-level metrics as card row."""
    import streamlit as st

    from neuropipeline.ui.components import metric_card

    metrics = [
        ("Accuracy", results_dict.get("accuracy")),
        ("F1 Score", results_dict.get("f1_score")),
        ("AUC-ROC", results_dict.get("auc_roc")),
    ]

    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics):
        with col:
            if value is not None:
                display_value = f"{value:.4f}" if isinstance(value, float) else str(value)
                st.markdown(metric_card(label, display_value), unsafe_allow_html=True)
            else:
                st.markdown(metric_card(label, "N/A"), unsafe_allow_html=True)


def _render_confusion_matrix(cm_data):
    """Render confusion matrix as a plotly heatmap."""
    import streamlit as st

    if cm_data is None:
        st.info("Confusion matrix data not available.")
        return

    try:
        import numpy as np
        import plotly.figure_factory as ff

        matrix = np.array(cm_data.get("matrix", []))
        labels = cm_data.get("labels", [f"Class {i}" for i in range(len(matrix))])

        if matrix.size == 0:
            st.info("Confusion matrix is empty.")
            return

        fig = ff.create_annotated_heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale="Blues",
            showscale=True,
        )
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            paper_bgcolor="#0a0e1a",
            plot_bgcolor="#111827",
            font=dict(color="#f1f5f9"),
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.warning("Plotly not available for confusion matrix visualization.")
        st.json(cm_data)


def _render_roc_curve(roc_data):
    """Render ROC curve plot."""
    import streamlit as st

    if roc_data is None:
        st.info("ROC curve data not available.")
        return

    try:
        import plotly.graph_objects as go

        fig = go.Figure()

        # Handle multi-class ROC
        if isinstance(roc_data, dict) and "fpr" in roc_data:
            fig.add_trace(go.Scatter(
                x=roc_data["fpr"],
                y=roc_data["tpr"],
                mode="lines",
                name=f"ROC (AUC = {roc_data.get('auc', 'N/A'):.3f})",
                line=dict(color="#00d4ff", width=2),
            ))
        elif isinstance(roc_data, list):
            colors = ["#00d4ff", "#ff4d6d", "#22c55e", "#f59e0b", "#8b5cf6"]
            for i, curve in enumerate(roc_data):
                color = colors[i % len(colors)]
                fig.add_trace(go.Scatter(
                    x=curve.get("fpr", []),
                    y=curve.get("tpr", []),
                    mode="lines",
                    name=curve.get("label", f"Class {i}"),
                    line=dict(color=color, width=2),
                ))

        # Diagonal reference line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(color="#334155", width=1, dash="dash"),
        ))

        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            paper_bgcolor="#0a0e1a",
            plot_bgcolor="#111827",
            font=dict(color="#f1f5f9"),
            legend=dict(x=0.6, y=0.1),
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.warning("Plotly not available for ROC curve visualization.")
        st.json(roc_data)


def _render_loss_curves(loss_history):
    """Render training/validation loss curves."""
    import streamlit as st

    if loss_history is None:
        st.info("Loss history not available.")
        return

    try:
        import plotly.graph_objects as go

        fig = go.Figure()

        train_loss = loss_history.get("train_loss", [])
        val_loss = loss_history.get("val_loss", [])
        epochs = list(range(1, max(len(train_loss), len(val_loss)) + 1))

        if train_loss:
            fig.add_trace(go.Scatter(
                x=epochs[:len(train_loss)],
                y=train_loss,
                mode="lines",
                name="Training Loss",
                line=dict(color="#00d4ff", width=2),
            ))

        if val_loss:
            fig.add_trace(go.Scatter(
                x=epochs[:len(val_loss)],
                y=val_loss,
                mode="lines",
                name="Validation Loss",
                line=dict(color="#ff4d6d", width=2),
            ))

        fig.update_layout(
            title="Training & Validation Loss",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            paper_bgcolor="#0a0e1a",
            plot_bgcolor="#111827",
            font=dict(color="#f1f5f9"),
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.warning("Plotly not available for loss curve visualization.")
        st.json(loss_history)


def _render_per_subject_accuracy(subject_data):
    """Render per-subject accuracy bar chart."""
    import streamlit as st

    if subject_data is None:
        st.info("Per-subject accuracy data not available.")
        return

    try:
        import plotly.graph_objects as go

        subjects = subject_data.get("subjects", [])
        accuracies = subject_data.get("accuracies", [])

        if not subjects or not accuracies:
            st.info("Per-subject data is empty.")
            return

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=subjects,
            y=accuracies,
            marker_color="#2563eb",
            marker_line_color="#00d4ff",
            marker_line_width=1,
        ))

        fig.update_layout(
            title="Per-Subject Accuracy",
            xaxis_title="Subject",
            yaxis_title="Accuracy",
            yaxis_range=[0, 1],
            paper_bgcolor="#0a0e1a",
            plot_bgcolor="#111827",
            font=dict(color="#f1f5f9"),
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.warning("Plotly not available for per-subject chart.")
        st.json(subject_data)


def _render_clinical_metrics(clinical_metrics):
    """Render ECG clinical metrics panel."""
    import streamlit as st

    from neuropipeline.ui.components import metric_card

    st.markdown("#### Clinical Performance Metrics")

    metrics = [
        ("Sensitivity", clinical_metrics.get("sensitivity")),
        ("Specificity", clinical_metrics.get("specificity")),
        ("PPV", clinical_metrics.get("ppv")),
        ("NPV", clinical_metrics.get("npv")),
    ]

    cols = st.columns(4)
    for col, (label, value) in zip(cols, metrics):
        with col:
            if value is not None:
                display = f"{value:.4f}" if isinstance(value, float) else str(value)
                st.markdown(metric_card(label, display), unsafe_allow_html=True)
            else:
                st.markdown(metric_card(label, "N/A"), unsafe_allow_html=True)


def _render_classification_report(report_data):
    """Render classification report as a table."""
    import streamlit as st

    st.markdown("#### Classification Report")

    if isinstance(report_data, dict):
        try:
            import pandas as pd

            # Convert sklearn-style classification report dict to DataFrame
            rows = []
            for class_name, metrics in report_data.items():
                if isinstance(metrics, dict):
                    rows.append({
                        "Class": class_name,
                        "Precision": metrics.get("precision", ""),
                        "Recall": metrics.get("recall", ""),
                        "F1-Score": metrics.get("f1-score", ""),
                        "Support": metrics.get("support", ""),
                    })

            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.json(report_data)
        except ImportError:
            st.json(report_data)
    elif isinstance(report_data, str):
        st.code(report_data, language="text")
    else:
        st.json(report_data)
