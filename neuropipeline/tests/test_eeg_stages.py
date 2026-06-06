"""Tests for all 8 EEG pipeline stages."""

import ast

import pytest

from neuropipeline.core.eeg.stages import (
    s1_quality_check,
    s2_preprocessing,
    s3_feature_extraction,
    s4_dataset_split,
    s5_model_definition,
    s6_training,
    s7_evaluation,
    s8_interpretation,
)

# All 8 EEG stage modules
EEG_STAGES = [
    ("s1_quality_check", s1_quality_check),
    ("s2_preprocessing", s2_preprocessing),
    ("s3_feature_extraction", s3_feature_extraction),
    ("s4_dataset_split", s4_dataset_split),
    ("s5_model_definition", s5_model_definition),
    ("s6_training", s6_training),
    ("s7_evaluation", s7_evaluation),
    ("s8_interpretation", s8_interpretation),
]


class TestEEGStageSchemas:
    """Test get_config_schema() for all EEG stages."""

    @pytest.mark.parametrize("stage_name,module", EEG_STAGES)
    def test_schema_returns_dict(self, stage_name, module):
        """Each stage's schema should return a non-empty dict."""
        schema = module.get_config_schema()
        assert isinstance(schema, dict)
        assert len(schema) > 0

    @pytest.mark.parametrize("stage_name,module", EEG_STAGES)
    def test_schema_entries_have_type(self, stage_name, module):
        """Each schema entry should have a 'type' key."""
        schema = module.get_config_schema()
        for key, value in schema.items():
            assert "type" in value, f"{stage_name}.{key} missing 'type'"

    @pytest.mark.parametrize("stage_name,module", EEG_STAGES)
    def test_schema_entries_have_description(self, stage_name, module):
        """Each schema entry should have a 'description' key."""
        schema = module.get_config_schema()
        for key, value in schema.items():
            assert "description" in value, f"{stage_name}.{key} missing 'description'"

    @pytest.mark.parametrize("stage_name,module", EEG_STAGES)
    def test_schema_entries_have_default(self, stage_name, module):
        """Each schema entry should have a 'default' key."""
        schema = module.get_config_schema()
        for key, value in schema.items():
            assert "default" in value, f"{stage_name}.{key} missing 'default'"


class TestEEGStageDefaults:
    """Test get_default_config() for all EEG stages."""

    @pytest.mark.parametrize("stage_name,module", EEG_STAGES)
    def test_default_config_returns_dict(self, stage_name, module, sample_agent1_eeg_report):
        """Each stage's default config should return a non-empty dict."""
        config = module.get_default_config(sample_agent1_eeg_report)
        assert isinstance(config, dict)
        assert len(config) > 0

    @pytest.mark.parametrize("stage_name,module", EEG_STAGES)
    def test_default_config_keys_match_schema(self, stage_name, module, sample_agent1_eeg_report):
        """Default config keys should be a subset of schema keys."""
        schema = module.get_config_schema()
        config = module.get_default_config(sample_agent1_eeg_report)
        for key in config:
            assert key in schema, (
                f"{stage_name}: default config key '{key}' not in schema"
            )


class TestEEGStageCodeGeneration:
    """Test generate_code() for all EEG stages."""

    @pytest.mark.parametrize("stage_name,module", EEG_STAGES)
    def test_generate_code_returns_string(self, stage_name, module, sample_agent1_eeg_report):
        """Each stage should generate a non-empty code string."""
        config = module.get_default_config(sample_agent1_eeg_report)
        code = module.generate_code(config)
        assert isinstance(code, str)
        assert len(code) > 0

    @pytest.mark.parametrize("stage_name,module", EEG_STAGES)
    def test_generate_code_is_valid_python(self, stage_name, module, sample_agent1_eeg_report):
        """Generated code should be syntactically valid Python (ast.parse)."""
        config = module.get_default_config(sample_agent1_eeg_report)
        code = module.generate_code(config)
        try:
            ast.parse(code)
        except SyntaxError as e:
            pytest.fail(
                f"{stage_name} generated invalid Python: {e}\n"
                f"First 200 chars: {code[:200]}"
            )

    @pytest.mark.parametrize("stage_name,module", EEG_STAGES)
    def test_generate_code_contains_imports(self, stage_name, module, sample_agent1_eeg_report):
        """Generated code should contain import statements."""
        config = module.get_default_config(sample_agent1_eeg_report)
        code = module.generate_code(config)
        assert "import" in code

    @pytest.mark.parametrize("stage_name,module", EEG_STAGES)
    def test_generate_code_contains_docstring(self, stage_name, module, sample_agent1_eeg_report):
        """Generated code should contain a module docstring."""
        config = module.get_default_config(sample_agent1_eeg_report)
        code = module.generate_code(config)
        assert '"""' in code


class TestEEGStageSpecific:
    """Specific tests for individual EEG stages."""

    def test_s1_quality_check_adjusts_for_high_quality(self, sample_agent1_eeg_report):
        """Quality check stage should adjust min_quality_score for high-quality data."""
        config = s1_quality_check.get_default_config(sample_agent1_eeg_report)
        # DREAMER has quality_score.score=85, should increase min_quality_score
        assert config["min_quality_score"] == 70

    def test_s2_preprocessing_ica_enabled(self, sample_agent1_eeg_report):
        """Preprocessing should enable ICA for high-quality 14-channel data."""
        config = s2_preprocessing.get_default_config(sample_agent1_eeg_report)
        assert config["ica_enabled"] is True
        assert config["ica_n_components"] == 13  # min(14-1, 20)

    def test_s3_feature_extraction_connectivity(self, sample_agent1_eeg_report):
        """Feature extraction should enable connectivity for 14 channels."""
        config = s3_feature_extraction.get_default_config(sample_agent1_eeg_report)
        assert config["compute_connectivity"] is True

    def test_s5_model_definition_gat_tcn(self, sample_agent1_eeg_report):
        """Model definition should default to gat_tcn for 14-channel data."""
        config = s5_model_definition.get_default_config(sample_agent1_eeg_report)
        assert config["model_type"] == "gat_tcn"
        assert config["n_channels"] == 14
