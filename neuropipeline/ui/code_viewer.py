"""Code viewer - live code preview with syntax highlighting."""


def render_code_preview(code_string):
    """Render a code preview panel with syntax highlighting.

    Displays the generated Python code with copy and save functionality.
    Updates live as the stage configuration changes.

    Args:
        code_string: The generated Python source code to display.
    """
    import streamlit as st

    st.markdown("#### Generated Code")

    if not code_string:
        st.info("No code generated yet. Configure a stage and click Generate.")
        return

    # Display code with syntax highlighting
    st.code(code_string, language="python")

    # Action buttons
    col_copy, col_save = st.columns(2)

    with col_copy:
        if st.button("Copy to Clipboard", key="btn_copy_code", use_container_width=True):
            st.session_state["clipboard_code"] = code_string
            st.success("Code copied to clipboard!")

    with col_save:
        if st.button("Save to File", key="btn_save_code", use_container_width=True):
            st.session_state["save_code_triggered"] = True

    # Download button for saving
    if st.session_state.get("save_code_triggered"):
        active_stage = st.session_state.get("active_stage", "stage")
        filename = f"{active_stage}.py"
        st.download_button(
            label=f"Download {filename}",
            data=code_string,
            file_name=filename,
            mime="text/x-python",
            key="download_code_btn",
        )

    # Show line count
    line_count = len(code_string.strip().split("\n"))
    st.caption(f"{line_count} lines of generated code")
