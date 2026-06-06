"""Executor panel - run generated code and display live output."""

import time


def render_executor(stage_name, code_string):
    """Render the execution panel for running generated stage code.

    Provides a Run Stage button, live stdout streaming, execution time
    display, and success/failure status indication.

    Args:
        stage_name: Name of the stage being executed.
        code_string: The generated Python code to execute.
    """
    import streamlit as st

    st.markdown(f"#### Execute: {_format_stage_name(stage_name)}")

    if not code_string:
        st.warning("No code to execute. Generate code first.")
        return

    col_run, col_status = st.columns([2, 1])

    with col_run:
        run_clicked = st.button(
            "Run Stage",
            key=f"run_{stage_name}",
            type="primary",
            use_container_width=True,
        )

    with col_status:
        if stage_name in st.session_state.get("execution_results", {}):
            result = st.session_state["execution_results"][stage_name]
            if result.get("success"):
                st.success("Passed")
            else:
                st.error("Failed")

    if run_clicked:
        _execute_stage(stage_name, code_string)


def _execute_stage(stage_name, code_string):
    """Execute the stage code using subprocess and stream output.

    Args:
        stage_name: Identifier for the stage.
        code_string: Python code to execute.
    """
    import subprocess
    import tempfile

    import streamlit as st

    output_container = st.empty()
    status_container = st.empty()

    status_container.info("Running...")
    st.session_state.setdefault("stage_states", {})[stage_name] = "running"

    start_time = time.time()
    stdout_lines = []

    try:
        # Write code to a temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp_file:
            tmp_file.write(code_string)
            tmp_path = tmp_file.name

        # Execute with subprocess
        process = subprocess.Popen(
            ["python", tmp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Stream output
        for line in iter(process.stdout.readline, ""):
            stdout_lines.append(line)
            output_container.code("".join(stdout_lines), language="text")

        process.wait()
        elapsed = time.time() - start_time

        # Store results
        success = process.returncode == 0
        result = {
            "success": success,
            "returncode": process.returncode,
            "output": "".join(stdout_lines),
            "elapsed_time": elapsed,
        }
        st.session_state.setdefault("execution_results", {})[stage_name] = result
        st.session_state["stage_states"][stage_name] = "done" if success else "failed"

        # Display final status
        if success:
            status_container.success(f"Completed in {elapsed:.2f}s")
        else:
            status_container.error(
                f"Failed (exit code {process.returncode}) after {elapsed:.2f}s"
            )

    except Exception as e:
        elapsed = time.time() - start_time
        result = {
            "success": False,
            "returncode": -1,
            "output": str(e),
            "elapsed_time": elapsed,
        }
        st.session_state.setdefault("execution_results", {})[stage_name] = result
        st.session_state.setdefault("stage_states", {})[stage_name] = "failed"
        status_container.error(f"Execution error: {e}")

    # Show execution time
    st.caption(f"Execution time: {elapsed:.2f} seconds")


def _format_stage_name(stage_key):
    """Convert a stage key to a display-friendly name."""
    parts = stage_key.split("_", 1)
    if len(parts) > 1:
        name = parts[1]
    else:
        name = parts[0]
    return name.replace("_", " ").title()
