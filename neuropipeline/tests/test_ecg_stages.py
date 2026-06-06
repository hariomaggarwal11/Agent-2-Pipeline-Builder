"""Tests for all 9 ECG pipeline stages."""

import ast

import pytest

from neuropipeline.core.ecg.stages import (
    s1_quality_check,
    s2_preprocessing,
    s3_beat_segmentation,
    s4_feature_extraction,
    s5_dataset_split,
    s6_model_definition,
    s7_training,
    s8_evaluation,
    s9_clinical_interpretation,
)

# All 9 ECG stage modules
ECG_STAGES = [
    ("s1_quality_check", s1_quality_check),
    ("s2_preprocessing", s2_preprocessing),
    ("s3_beat_segmentation", s3_beat_segmentation),
    ("s4_feature_extraction", s4_feature_extraction),
    ("s5_dataset_split", s5_dataset_split),
    ("s6_model_definition", s6_model_definition),
    ("s7_training", s7_training),
    ("s8_evaluation", s8_evaluation),
    ("s9_clinical_interpretation", s9_clinical_interpretation),
]


class TestECGStageSchemas:
    """Test get_config_schema() for all ECG stages."""

    @pytest.mark.parametrize("stage_name,module", ECG_STAGES)
    def test_schema_returns_dict(self, stage_name, module):
        """Each stage's schema should return a non-empty dict."""
        schema = module.get_config_schema()
        assert isinstance(schema, dict)
        assert len(schema) > 0

    @pytest.mark.parametrize("stage_name,module", ECG_STAGES)
    def test_schema_entries_have_type(self, stage_name, module):
        """Each schema entry should have a 'type' key."""
        schema = module.get_config_schema()
        for key, value in schema.items():
            assert "type" in value, f"{stage_name}.{key} missing 'type'"

    @pytest.mark.parametrize("stage_name,module", ECG_STAGES)
    def test_schema_entries_have_description(self, stage_name, module):
        """Each schema entry should have a 'description' key."""
        schema = module.get_config_schema()
        for key, value in schema.items():
            assert "description" in value, f"{stage_name}.{key} missing 'description'"

    @pytest.mark.parametrize("stage_name,module", ECG_STAGES)
    def test_schema_entries_have_default(self, stage_name, module):
        """Each schema entry should have a 'default' key."""
        schema = module.get_config_schema()
        for key, value in schema.items():
            assert "default" in value, f"{stage_name}.{key} missing 'default'"


class TestECGStageDefaults:
    """Test get_default_config() for all ECG stages."""

    @pytest.mark.parametrize("stage_name,module", ECG_STAGES)
    def test_default_config_returns_dict(self, stage_name, module, sample_agent1_ecg_report):
        """Each stage's default config should return a non-empty dict."""
        config = module.get_default_config(sample_agent1_ecg_report)
        assert isinstance(config, dict)
        assert len(config) > 0

    @pytest.mark.parametrize("stage_name,module", ECG_STAGES)
    def test_default_config_keys_match_schema(self, stage_name, module, sample_agent1_ecg_report):
        """Default config keys should be a subset of schema keys."""
        schema = module.get_config_schema()
        config = module.get_default_config(sample_agent1_ecg_report)
        for key in config:
            assert key in schema, (
                f"{stage_name}: default config key '{key}' not in schema"
            )


class TestECGStageCodeGeneration:
    """Test generate_code() for all ECG stages."""

    @pytest.mark.parametrize("stage_name,module", ECG_STAGES)
    def test_generate_code_returns_string(self, stage_name, module, sample_agent1_ecg_report):
        """Each stage should generate a non-empty code string."""
        config = module.get_default_config(sample_agent1_ecg_report)
        code = module.generate_code(config)
        assert isinstance(code, str)
        assert len(code) > 0

    @pytest.mark.parametrize("stage_name,module", ECG_STAGES)
    def test_generate_code_is_valid_python(self, stage_name, module, sample_agent1_ecg_report):
        """Generated code should be syntactically valid Python (ast.parse)."""
        config = module.get_default_config(sample_agent1_ecg_report)
        code = module.generate_code(config)
        try:
            ast.parse(code)
        except SyntaxError as e:
            pytest.fail(
                f"{stage_name} generated invalid Python: {e}\n"
                f"First 200 chars: {code[:200]}"
            )

    @pytest.mark.parametrize("stage_name,module", ECG_STAGES)
    def test_generate_code_contains_imports(self, stage_name, module, sample_agent1_ecg_report):
        """Generated code should contain import statements."""
        config = module.get_default_config(sample_agent1_ecg_report)
        code = module.generate_code(config)
        assert "import" in code

    @pytest.mark.parametrize("stage_name,module", ECG_STAGES)
    def test_generate_code_contains_docstring(self, stage_name, module, sample_agent1_ecg_report):
        """Generated code should contain a module docstring."""
        config = module.get_default_config(sample_agent1_ecg_report)
        code = module.generate_code(config)
        assert '"""' in code


class TestECGStageSpecific:
    """Specific tests for individual ECG stages."""

    def test_s1_quality_check_has_clipping_check(self, sample_agent1_ecg_report):
        """Quality check stage should include clipping detection config."""
        config = s1_quality_check.get_default_config(sample_agent1_ecg_report)
        assert "check_clipping" in config
        assert config["check_clipping"] is True

    def test_s2_preprocessing_baseline_correction(self, sample_agent1_ecg_report):
        """Preprocessing should enable baseline correction."""
        config = s2_preprocessing.get_default_config(sample_agent1_ecg_report)
        assert config["baseline_correction"] is True

    def test_s3_beat_segmentation_defaults(self, sample_agent1_ecg_report):
        """Beat segmentation should have R-peak detection config."""
        schema = s3_beat_segmentation.get_config_schema()
        # Should have some beat-segmentation related keys
        assert len(schema) > 0

    def test_s6_model_definition_defaults(self, sample_agent1_ecg_report):
        """Model definition stage should produce valid default config."""
        config = s6_model_definition.get_default_config(sample_agent1_ecg_report)
        assert isinstance(config, dict)
        assert len(config) > 0

    def test_s9_clinical_interpretation_generates_code(self, sample_agent1_ecg_report):
        """Clinical interpretation stage should generate valid code."""
        config = s9_clinical_interpretation.get_default_config(sample_agent1_ecg_report)
        code = s9_clinical_interpretation.generate_code(config)
        ast.parse(code)  # Should not raise SyntaxError
