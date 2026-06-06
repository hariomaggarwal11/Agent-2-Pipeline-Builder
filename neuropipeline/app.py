"""NeuroPipeline - Streamlit application entry point.

Run with:
  streamlit run app.py          (from inside neuropipeline/)
  streamlit run neuropipeline/app.py  (from parent directory)
"""

import sys
from pathlib import Path

# Ensure the neuropipeline package is importable regardless of working directory
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st

from neuropipeline.config import ECG_STAGES, EEG_STAGES, MODEL_REGISTRY
from neuropipeline.ui.code_viewer import render_code_preview
from neuropipeline.ui.executor_panel import render_executor
from neuropipeline.ui.landing import render_landing
from neuropipeline.ui.pipeline_canvas import render_canvas
from neuropipeline.ui.results_dashboard import render_results
from neuropipeline.ui.stage_configurator import render_stage_config
from neuropipeline.ui.styles import get_custom_css


def main():
    """Main application entry point."""
    # Page configuration
    st.set_page_config(
        page_title="NeuroPipeline",
        layout="wide",
        page_icon="\U0001f52c",
        initial_sidebar_state="expanded",
    )

    # Inject custom CSS
    st.markdown(get_custom_css(), unsafe_allow_html=True)

    # Initialize session state
    _init_session_state()

    # Render sidebar
    _render_sidebar()

    # Route to appropriate screen
    current_screen = st.session_state.get("current_screen", "landing")

    if current_screen == "landing":
        render_landing()
    elif current_screen == "canvas":
        _render_canvas_screen()


def _init_session_state():
    """Initialize all session state keys with defaults."""
    defaults = {
        "current_screen": "landing",
        "pipeline_config": None,
        "stage_states": {},
        "generated_code": {},
        "execution_results": {},
        "active_stage": None,
        "active_stage_idx": 0,
        "inspection_report": None,
        "task_selection": "Emotion Recognition",
        "model_family": "auto",
        "output_language": "Python",
        "run_all_triggered": False,
        "save_code_triggered": False,
        "clipboard_code": None,
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def _render_sidebar():
    """Render the application sidebar with branding and navigation."""
    with st.sidebar:
        st.markdown(
            '<h2 style="margin-bottom: 0;">\U0001f52c NeuroPipeline</h2>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="color: #64748b; font-size: 0.85em; margin-top: 4px;">'
            "Agent 2 Pipeline Builder</p>",
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Task selection
        task_options = [
            "Emotion Recognition",
            "MDD Detection",
            "Motor Imagery",
            "Arrhythmia Detection",
            "ECG Classification",
            "Custom",
        ]
        st.session_state["task_selection"] = st.radio(
            "Task",
            options=task_options,
            index=task_options.index(st.session_state.get("task_selection", "Emotion Recognition")),
            key="sidebar_task",
        )

        st.markdown("---")

        # Model family selection
        model_options = ["auto"] + list(MODEL_REGISTRY.keys())
        model_labels = ["Auto-detect"] + [MODEL_REGISTRY[k]["name"] for k in MODEL_REGISTRY]
        st.session_state["model_family"] = st.radio(
            "Model Family",
            options=model_options,
            format_func=lambda x: "Auto-detect" if x == "auto" else MODEL_REGISTRY.get(x, {}).get("name", x),
            index=model_options.index(st.session_state.get("model_family", "auto")),
            key="sidebar_model",
        )

        st.markdown("---")

        # Output language selection
        lang_options = ["Python", "Python + PyTorch", "Python + TensorFlow"]
        st.session_state["output_language"] = st.radio(
            "Output Language",
            options=lang_options,
            index=lang_options.index(st.session_state.get("output_language", "Python")),
            key="sidebar_lang",
        )

        st.markdown("---")

        # Navigation
        if st.session_state.get("pipeline_config"):
            if st.button("Back to Landing", use_container_width=True):
                st.session_state["current_screen"] = "landing"
                st.rerun()


def _render_canvas_screen():
    """Render the main canvas screen with pipeline, configurator, and panels."""
    # Build pipeline config if not yet set
    if st.session_state.get("pipeline_config") is None:
        _build_pipeline_config()

    pipeline_config = st.session_state["pipeline_config"]
    stage_states = st.session_state["stage_states"]

    # Main layout: pipeline canvas on left, config/code on right
    col_canvas, col_panels = st.columns([1, 1])

    with col_canvas:
        render_canvas(pipeline_config, stage_states)

    with col_panels:
        active_stage = st.session_state.get("active_stage")

        if active_stage:
            # Tabs for config, code, executor, results
            tab_config, tab_code, tab_exec, tab_results = st.tabs(
                ["Configure", "Code", "Execute", "Results"]
            )

            modality = pipeline_config.get("modality", "eeg")

            with tab_config:
                stage_config = st.session_state.get("generated_code", {}).get(
                    f"{active_stage}_config", {}
                )
                updated_config = render_stage_config(
                    active_stage, modality, stage_config
                )
                st.session_state.setdefault("generated_code", {})[
                    f"{active_stage}_config"
                ] = updated_config

            with tab_code:
                code = st.session_state.get("generated_code", {}).get(active_stage, "")
                render_code_preview(code)

            with tab_exec:
                code = st.session_state.get("generated_code", {}).get(active_stage, "")
                render_executor(active_stage, code)

            with tab_results:
                results = st.session_state.get("execution_results", {}).get(
                    active_stage, {}
                )
                render_results(results)
        else:
            st.info("Select a stage from the pipeline canvas to configure it.")


def _build_pipeline_config():
    """Build the pipeline configuration from session state."""
    report = st.session_state.get("inspection_report", {})
    modality = report.get("modality", "eeg").lower()

    if modality == "ecg":
        stages = list(ECG_STAGES)
    else:
        stages = list(EEG_STAGES)

    pipeline_config = {
        "modality": modality,
        "stages": stages,
        "task": st.session_state.get("task_selection", "Emotion Recognition"),
        "model_family": st.session_state.get("model_family", "auto"),
        "output_language": st.session_state.get("output_language", "Python"),
    }

    st.session_state["pipeline_config"] = pipeline_config

    # Initialize stage states
    st.session_state["stage_states"] = {
        stage: "pending" for stage in stages
    }


if __name__ == "__main__":
    main()
