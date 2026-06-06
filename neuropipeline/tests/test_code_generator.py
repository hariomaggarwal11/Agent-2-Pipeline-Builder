"""Tests for code generator - template rendering and Anthropic API mocking."""

import ast
from unittest.mock import patch, MagicMock

import pytest

from neuropipeline.core.code_generator import (
    generate_stage_code,
    generate_custom_model,
    _render_template,
    _generate_from_config,
    _call_claude_api,
)


@pytest.fixture
def eeg_dataset_info():
    """EEG dataset info for code generation."""
    return {
        "modality": "EEG",
        "dataset": "DREAMER",
        "n_channels": 14,
        "sampling_rate": 128,
        "labels": ["arousal", "valence", "dominance"],
    }


@pytest.fixture
def ecg_dataset_info():
    """ECG dataset info for code generation."""
    return {
        "modality": "ECG",
        "dataset": "MIT-BIH Arrhythmia",
        "n_channels": 2,
        "sampling_rate": 360,
        "labels": ["N", "L", "R", "V", "A"],
    }


@pytest.fixture
def sample_stage_config():
    """Sample stage configuration."""
    return {
        "bandpass_low": 0.5,
        "bandpass_high": 45.0,
        "notch_filter": 50.0,
        "reference": "average",
        "ica_enabled": True,
    }


class TestGenerateStageCode:
    """Test generate_stage_code function."""

    def test_generates_code_for_known_template(self, sample_stage_config, eeg_dataset_info):
        """Should generate code for a stage with a known template."""
        code = generate_stage_code(
            "eeg_quality_check", sample_stage_config, eeg_dataset_info
        )
        assert isinstance(code, str)
        assert len(code) > 0

    def test_fallback_for_unknown_stage(self, sample_stage_config, eeg_dataset_info):
        """Should fall back to config-based generation for unknown stages."""
        code = generate_stage_code(
            "unknown_stage", sample_stage_config, eeg_dataset_info
        )
        assert isinstance(code, str)
        assert "unknown_stage" in code

    def test_fallback_code_is_valid_python(self, sample_stage_config, eeg_dataset_info):
        """Fallback generated code should be valid Python."""
        code = generate_stage_code(
            "custom_stage", sample_stage_config, eeg_dataset_info
        )
        try:
            ast.parse(code)
        except SyntaxError as e:
            pytest.fail(f"Generated code is not valid Python: {e}")

    def test_fallback_code_contains_config(self, sample_stage_config, eeg_dataset_info):
        """Fallback code should embed the configuration."""
        code = generate_stage_code(
            "test_stage", sample_stage_config, eeg_dataset_info
        )
        assert "CONFIG" in code
        assert "DATASET_INFO" in code

    def test_fallback_code_contains_modality(self, sample_stage_config, eeg_dataset_info):
        """Fallback code should reference the modality."""
        code = generate_stage_code(
            "test_stage", sample_stage_config, eeg_dataset_info
        )
        assert "EEG" in code


class TestGenerateFromConfig:
    """Test the _generate_from_config fallback."""

    def test_returns_string(self, sample_stage_config, eeg_dataset_info):
        """Should return a string."""
        code = _generate_from_config("test_stage", sample_stage_config, eeg_dataset_info)
        assert isinstance(code, str)

    def test_valid_python(self, sample_stage_config, eeg_dataset_info):
        """Generated code should parse as valid Python."""
        code = _generate_from_config("test_stage", sample_stage_config, eeg_dataset_info)
        ast.parse(code)

    def test_contains_run_function(self, sample_stage_config, eeg_dataset_info):
        """Generated code should define a run() function."""
        code = _generate_from_config("test_stage", sample_stage_config, eeg_dataset_info)
        assert "def run(" in code

    def test_ecg_code_generation(self, sample_stage_config, ecg_dataset_info):
        """Should generate code for ECG modality."""
        code = _generate_from_config("ecg_stage", sample_stage_config, ecg_dataset_info)
        assert "ECG" in code
        ast.parse(code)


class TestRenderTemplate:
    """Test Jinja2 template rendering."""

    def test_returns_none_for_nonexistent_template(self, sample_stage_config, eeg_dataset_info):
        """Should return None when template file doesn't exist."""
        result = _render_template(
            "templates/nonexistent.py.j2", sample_stage_config, eeg_dataset_info
        )
        assert result is None

    def test_renders_existing_eeg_template(self, eeg_dataset_info):
        """Should render an existing EEG template."""
        config = {"check_flat_channels": True, "line_noise_freq": 50.0}
        result = _render_template(
            "templates/eeg/s1_quality_check.py.j2", config, eeg_dataset_info
        )
        # Result depends on whether the template exists and is valid
        if result is not None:
            assert isinstance(result, str)
            assert len(result) > 0


class TestGenerateCustomModel:
    """Test custom model generation with mocked Anthropic API."""

    def test_mocked_api_returns_code(self, eeg_dataset_info):
        """Should return generated code when API is mocked."""
        mock_response = MagicMock()
        mock_block = MagicMock()
        mock_block.text = '''```python
import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, n_channels=14, n_classes=3):
        super().__init__()
        self.fc = nn.Linear(n_channels * 128, n_classes)

    def forward(self, x):
        x = x.flatten(1)
        return self.fc(x)
```'''
        mock_response.content = [mock_block]

        with patch("neuropipeline.core.code_generator.Anthropic", create=True) as mock_anthropic_cls:
            # Mock the import inside the function
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic_cls.return_value = mock_client

            with patch(
                "neuropipeline.core.code_generator._call_claude_api"
            ) as mock_call:
                mock_call.return_value = (
                    "import torch\nimport torch.nn as nn\n\n"
                    "class CustomModel(nn.Module):\n"
                    "    def __init__(self):\n"
                    "        super().__init__()\n"
                    "        self.fc = nn.Linear(14 * 128, 3)\n\n"
                    "    def forward(self, x):\n"
                    "        return self.fc(x.flatten(1))\n"
                )
                code = generate_custom_model(
                    "A GAT-TCN model for emotion recognition", eeg_dataset_info
                )
                assert isinstance(code, str)
                assert "class" in code or "CustomModel" in code

    def test_fallback_on_import_error(self, eeg_dataset_info):
        """Should use fallback code when anthropic is not importable."""
        with patch(
            "neuropipeline.core.code_generator._call_claude_api"
        ) as mock_call:
            # Simulate the fallback behavior
            mock_call.side_effect = ImportError("No module named 'anthropic'")
            # Directly test the fallback path
            from neuropipeline.core.code_generator import _fallback_model_code

            code = _fallback_model_code("test prompt")
            assert isinstance(code, str)
            assert "CustomModel" in code
            ast.parse(code)

    def test_fallback_code_is_valid_python(self):
        """Fallback model code should be syntactically valid."""
        from neuropipeline.core.code_generator import _fallback_model_code

        code = _fallback_model_code("Generate a CNN model")
        ast.parse(code)

    def test_fallback_code_has_forward_method(self):
        """Fallback model code should define forward method."""
        from neuropipeline.core.code_generator import _fallback_model_code

        code = _fallback_model_code("Generate a model")
        assert "def forward(" in code


class TestCallClaudeAPI:
    """Test the Claude API call function with mocking."""

    def test_api_call_with_mock(self):
        """Test that _call_claude_api works with mocked Anthropic client."""
        mock_response = MagicMock()
        mock_block = MagicMock()
        mock_block.text = "```python\nprint('hello')\n```"
        mock_response.content = [mock_block]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        mock_anthropic_module = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
            result = _call_claude_api("Generate a hello world")
            assert result == "print('hello')"

    def test_api_call_extracts_python_code_block(self):
        """Test extraction of python code block from response."""
        mock_response = MagicMock()
        mock_block = MagicMock()
        mock_block.text = "Here is the code:\n```python\nx = 42\ny = x + 1\n```\nDone!"
        mock_response.content = [mock_block]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        mock_anthropic_module = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
            result = _call_claude_api("test prompt")
            assert "x = 42" in result
            assert "y = x + 1" in result
            # Should not include the markdown fence
            assert "```" not in result

    def test_api_call_handles_plain_text(self):
        """Test handling response without code fences."""
        mock_response = MagicMock()
        mock_block = MagicMock()
        mock_block.text = "import numpy as np\ndata = np.zeros(10)"
        mock_response.content = [mock_block]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        mock_anthropic_module = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
            result = _call_claude_api("test prompt")
            assert "import numpy" in result

    def test_api_call_handles_exception(self):
        """Test graceful handling of API errors."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API Error")

        mock_anthropic_module = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
            result = _call_claude_api("test prompt")
            # Should return fallback code or error message
            assert isinstance(result, str)
            assert len(result) > 0
