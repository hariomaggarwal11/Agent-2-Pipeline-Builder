"""Executor - subprocess-based stage execution with stdout streaming."""

import subprocess
import time
import os
import tempfile
from typing import Callable, Optional
from enum import Enum


class StageStatus(Enum):
    """Status of a pipeline stage execution."""
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class StageResult:
    """Result of executing a pipeline stage."""

    def __init__(self, stage_name: str):
        self.stage_name = stage_name
        self.status = StageStatus.PENDING
        self.stdout = ""
        self.stderr = ""
        self.return_code = None
        self.start_time = None
        self.end_time = None
        self.duration = None
        self.error_message = None

    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return {
            "stage_name": self.stage_name,
            "status": self.status.value,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "return_code": self.return_code,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "error_message": self.error_message,
        }


class PipelineExecutor:
    """Execute pipeline stages as subprocesses with streaming output.

    Parameters
    ----------
    working_dir : str, optional
        Working directory for script execution.
    timeout : int, optional
        Default timeout per stage in seconds (default: 300).
    on_output : callable, optional
        Callback function called with each line of stdout.
    """

    def __init__(
        self,
        working_dir: Optional[str] = None,
        timeout: int = 300,
        on_output: Optional[Callable[[str, str], None]] = None,
    ):
        self.working_dir = working_dir or tempfile.mkdtemp(prefix="neuropipeline_")
        self.timeout = timeout
        self.on_output = on_output
        self.results: dict[str, StageResult] = {}
        self._stop_requested = False

    def execute_stage(self, stage_name: str, code: str, timeout: Optional[int] = None) -> StageResult:
        """Execute a single pipeline stage.

        Parameters
        ----------
        stage_name : str
            Name of the stage being executed.
        code : str
            Python code to execute.
        timeout : int, optional
            Timeout for this specific stage.

        Returns
        -------
        StageResult
            Execution result with status, stdout, stderr.
        """
        result = StageResult(stage_name)
        self.results[stage_name] = result

        stage_timeout = timeout or self.timeout

        # Write code to temporary file
        script_path = os.path.join(self.working_dir, f"{stage_name}.py")
        with open(script_path, "w") as f:
            f.write(code)

        result.status = StageStatus.RUNNING
        result.start_time = time.time()

        try:
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.working_dir,
                env=self._get_env(),
            )

            # Stream stdout
            stdout_lines = []
            stderr_lines = []

            try:
                stdout_data, stderr_data = process.communicate(timeout=stage_timeout)
                stdout_lines.append(stdout_data)
                stderr_lines.append(stderr_data)

                if self.on_output and stdout_data:
                    for line in stdout_data.splitlines():
                        self.on_output(stage_name, line)

            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                result.status = StageStatus.FAILED
                result.error_message = f"Stage timed out after {stage_timeout}s"
                result.end_time = time.time()
                result.duration = result.end_time - result.start_time
                return result

            result.stdout = "".join(stdout_lines)
            result.stderr = "".join(stderr_lines)
            result.return_code = process.returncode

            if process.returncode == 0:
                result.status = StageStatus.DONE
            else:
                result.status = StageStatus.FAILED
                result.error_message = result.stderr or f"Exit code: {process.returncode}"

        except Exception as e:
            result.status = StageStatus.FAILED
            result.error_message = str(e)

        result.end_time = time.time()
        result.duration = result.end_time - result.start_time
        return result

    def execute_pipeline(self, stages: list[tuple[str, str]], stop_on_failure: bool = True) -> list[StageResult]:
        """Execute a sequence of pipeline stages.

        Parameters
        ----------
        stages : list of (stage_name, code) tuples
            Ordered list of stages to execute.
        stop_on_failure : bool
            If True, skip remaining stages after a failure.

        Returns
        -------
        list of StageResult
            Results for all stages.
        """
        results = []
        failed = False

        for stage_name, code in stages:
            if self._stop_requested:
                result = StageResult(stage_name)
                result.status = StageStatus.SKIPPED
                result.error_message = "Execution stopped by user"
                results.append(result)
                continue

            if failed and stop_on_failure:
                result = StageResult(stage_name)
                result.status = StageStatus.SKIPPED
                result.error_message = "Skipped due to previous failure"
                results.append(result)
                continue

            result = self.execute_stage(stage_name, code)
            results.append(result)

            if result.status == StageStatus.FAILED:
                failed = True

        return results

    def stop(self):
        """Request execution to stop after the current stage."""
        self._stop_requested = True

    def reset(self):
        """Reset the executor state."""
        self._stop_requested = False
        self.results.clear()

    def get_status_summary(self) -> dict:
        """Get a summary of all stage statuses.

        Returns
        -------
        dict
            Summary with counts per status and list of results.
        """
        summary = {
            "total": len(self.results),
            "pending": 0,
            "running": 0,
            "done": 0,
            "failed": 0,
            "skipped": 0,
            "stages": {},
        }

        for name, result in self.results.items():
            summary[result.status.value] += 1
            summary["stages"][name] = result.to_dict()

        return summary

    def _get_env(self) -> dict:
        """Get environment variables for subprocess execution."""
        env = os.environ.copy()
        env["PYTHONPATH"] = self.working_dir
        return env
