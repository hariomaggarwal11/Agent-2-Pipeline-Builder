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

    # Run All / Reset buttons
    col_run, col_reset = st.columns([1, 1])
    with col_run:
        run_all_clicked = st.button("Run All", type="primary", use_container_width=True)
    with col_reset:
        if st.button("Reset Pipeline", use_container_width=True):
            for key in stage_states:
                stage_states[key] = "pending"
            st.session_state["stage_states"] = stage_states
            st.session_state["generated_code"] = {}
            st.session_state["execution_results"] = {}
            st.rerun()

    # Handle Run All
    if run_all_clicked:
        _run_all_stages(stages, pipeline_config, stage_states)

    st.markdown("---")

    # Render pipeline nodes in a vertical flow
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


def _run_all_stages(stages, pipeline_config, stage_states):
    """Generate code for all stages and execute them sequentially."""
    import streamlit as st

    from neuropipeline.core.code_generator import generate_stage_code

    modality = pipeline_config.get("modality", "eeg")
    dataset_info = st.session_state.get("inspection_report", {})

    # Build dataset info dict for templates
    template_dataset_info = {
        "modality": modality.upper(),
        "dataset": dataset_info.get("known_dataset", {}).get("name", "Unknown")
            if dataset_info.get("known_dataset") else "Unknown",
        "n_channels": dataset_info.get("metadata", {}).get("Number of Channels", 14),
        "sfreq": dataset_info.get("metadata", {}).get("Sampling Frequency", "128 Hz"),
        "quality_score": dataset_info.get("quality_score", {}).get("score", 75)
            if dataset_info.get("quality_score") else 75,
    }

    with st.status("\u26a1 Running all pipeline stages...", expanded=True) as status:
        all_passed = True

        for idx, stage_key in enumerate(stages):
            stage_num = idx + 1
            stage_name = _format_stage_name(stage_key)
            st.write(f"\U0001f504 Stage {stage_num}: {stage_name}...")

            # Update state to running
            stage_states[stage_key] = "running"
            st.session_state["stage_states"] = stage_states

            # Get stage config (use defaults if none configured)
            stage_config = st.session_state.get("generated_code", {}).get(
                f"{stage_key}_config", {}
            )

            # Map stage key to template name
            template_name = _get_template_name(stage_key, modality)

            # Generate code
            try:
                code = generate_stage_code(template_name, stage_config, template_dataset_info)
                st.session_state.setdefault("generated_code", {})[stage_key] = code
            except Exception as e:
                code = f"# Code generation failed: {e}\nprint('[{stage_key}] Error: {e}')"
                st.session_state.setdefault("generated_code", {})[stage_key] = code

            # Execute the generated code
            success = _execute_stage_inline(stage_key, code)

            if success:
                stage_states[stage_key] = "done"
                st.write(f"\u2705 Stage {stage_num}: {stage_name} - Complete")
            else:
                stage_states[stage_key] = "failed"
                st.write(f"\u274c Stage {stage_num}: {stage_name} - Failed")
                all_passed = False
                # Mark remaining stages as skipped
                for remaining_key in stages[idx + 1:]:
                    stage_states[remaining_key] = "skipped"
                break

            st.session_state["stage_states"] = stage_states

        if all_passed:
            status.update(label="\u2705 All stages completed!", state="complete")
        else:
            status.update(label="\u274c Pipeline stopped due to failure", state="error")


def _execute_stage_inline(stage_key, code):
    """Execute stage code inline and capture output.

    Returns True if successful, False otherwise.
    """
    import subprocess
    import tempfile
    import os
    import time

    import streamlit as st

    start_time = time.time()

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp_file:
            tmp_file.write(code)
            tmp_path = tmp_file.name

        process = subprocess.Popen(
            ["python", tmp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        stdout_data, _ = process.communicate(timeout=120)
        elapsed = time.time() - start_time

        success = process.returncode == 0

        # Store result
        result = {
            "success": success,
            "returncode": process.returncode,
            "output": stdout_data,
            "elapsed_time": elapsed,
        }
        st.session_state.setdefault("execution_results", {})[stage_key] = result

        # Show output in expander
        if stdout_data.strip():
            with st.expander(f"Output ({elapsed:.1f}s)", expanded=not success):
                st.code(stdout_data, language="text")

        # Clean up
        os.unlink(tmp_path)
        return success

    except subprocess.TimeoutExpired:
        st.warning(f"Stage timed out after 120s")
        result = {"success": False, "returncode": -1, "output": "Timeout", "elapsed_time": 120}
        st.session_state.setdefault("execution_results", {})[stage_key] = result
        return False
    except Exception as e:
        st.warning(f"Execution error: {e}")
        result = {"success": False, "returncode": -1, "output": str(e), "elapsed_time": 0}
        st.session_state.setdefault("execution_results", {})[stage_key] = result
        return False


def _get_template_name(stage_key, modality):
    """Map a stage key to a template lookup name."""
    # Stage keys are like 's1_quality_check', 's2_preprocessing', etc.
    # Template registry keys are like 'eeg_quality_check', 'ecg_preprocessing'
    parts = stage_key.split("_", 1)
    if len(parts) > 1:
        stage_name = parts[1]
    else:
        stage_name = stage_key
    return f"{modality}_{stage_name}"


def _format_stage_name(stage_key):
    """Convert a stage key to a display-friendly name.

    Args:
        stage_key: Stage identifier like 's1_quality_check'.

    Returns:
        Formatted name like 'Quality Check'.
    """
    parts = stage_key.split("_", 1)
    if len(parts) > 1:
        name = parts[1]
    else:
        name = parts[0]
    return name.replace("_", " ").title()
