"""Pipeline canvas - visual pipeline flow with connected stage nodes."""


def render_canvas(pipeline_config, stage_states):
    """Render the pipeline canvas with interactive stage nodes.

    Displays the pipeline as a series of connected node cards, each showing
    the stage number, name, and execution status. Supports EEG (8 stages)
    and ECG (9 stages) flows.

    Args:
        pipeline_config: Dict containing pipeline configuration with 'modality'
            and 'stages' keys.
        stage_states: Dict mapping stage keys to status strings
            ('pending', 'running', 'done', 'failed', 'skipped').
    """
    import streamlit as st

    from neuropipeline.ui.components import (
        pipeline_node_card,
        progress_indicator,
        stage_connection_arrow,
    )

    modality = pipeline_config.get("modality", "eeg")
    stages = pipeline_config.get("stages", [])

    # Header
    st.markdown(
        f'<h2>Pipeline Canvas - {modality.upper()}</h2>',
        unsafe_allow_html=True,
    )

    # Progress indicator
    done_count = sum(1 for s in stage_states.values() if s == "done")
    total_count = len(stages)
    st.markdown(
        progress_indicator(done_count, total_count),
        unsafe_allow_html=True,
    )

    # Run All button
    col_run, col_reset = st.columns([1, 1])
    with col_run:
        if st.button("Run All", type="primary", use_container_width=True):
            st.session_state["run_all_triggered"] = True
    with col_reset:
        if st.button("Reset Pipeline", use_container_width=True):
            for key in stage_states:
                stage_states[key] = "pending"
            st.session_state["stage_states"] = stage_states
            st.rerun()

    st.markdown("---")

    # Render pipeline nodes in a vertical flow
    # Use columns to display nodes with arrows between them
    for idx, stage_key in enumerate(stages):
        stage_name = _format_stage_name(stage_key)
        status = stage_states.get(stage_key, "pending")
        stage_num = idx + 1

        # Render node as a button-like element
        node_html = pipeline_node_card(stage_name, status, stage_num)

        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            st.markdown(node_html, unsafe_allow_html=True)

            # Stage selection button
            if st.button(
                f"Configure Stage {stage_num}",
                key=f"select_stage_{stage_key}",
                use_container_width=True,
            ):
                st.session_state["active_stage"] = stage_key
                st.session_state["active_stage_idx"] = idx

            # Arrow connector (except after last stage)
            if idx < len(stages) - 1:
                st.markdown(stage_connection_arrow(), unsafe_allow_html=True)


def _format_stage_name(stage_key):
    """Convert a stage key to a display-friendly name.

    Args:
        stage_key: Stage identifier like 's1_quality_check'.

    Returns:
        Formatted name like 'Quality Check'.
    """
    # Remove the sN_ prefix and convert underscores to spaces
    parts = stage_key.split("_", 1)
    if len(parts) > 1:
        name = parts[1]
    else:
        name = parts[0]
    return name.replace("_", " ").title()
