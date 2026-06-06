"""Tests for Agent 1 bridge - loading and parsing JSON reports."""

import json
import os
import tempfile

import pytest

from neuropipeline.core.agent1_bridge import (
    load_agent1_report,
    parse_agent1_report,
    get_channel_count,
    get_quality_score,
)


class TestLoadAgent1Report:
    """Tests for loading Agent 1 JSON reports from file."""

    def test_load_valid_eeg_report(self, sample_agent1_eeg_report):
        """Test loading a valid EEG JSON report from disk."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(sample_agent1_eeg_report, f)
            tmp_path = f.name

        try:
            report = load_agent1_report(tmp_path)
            assert report["modality"] == "EEG"
            assert "metadata" in report
            assert "quality_score" in report
        finally:
            os.unlink(tmp_path)

    def test_load_valid_ecg_report(self, sample_agent1_ecg_report):
        """Test loading a valid ECG JSON report from disk."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(sample_agent1_ecg_report, f)
            tmp_path = f.name

        try:
            report = load_agent1_report(tmp_path)
            assert report["modality"] == "ECG"
            assert "metadata" in report
        finally:
            os.unlink(tmp_path)

    def test_load_nonexistent_file(self):
        """Test FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_agent1_report("/nonexistent/path/report.json")

    def test_load_invalid_json(self):
        """Test ValueError for malformed JSON."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("not valid json {{{")
            tmp_path = f.name

        try:
            with pytest.raises(Exception):
                load_agent1_report(tmp_path)
        finally:
            os.unlink(tmp_path)


class TestParseAgent1Report:
    """Tests for parsing Agent 1 report dictionaries."""

    def test_parse_eeg_report(self, sample_agent1_eeg_report):
        """Test parsing a DREAMER EEG report."""
        parsed = parse_agent1_report(sample_agent1_eeg_report)
        assert parsed["modality"] == "EEG"
        assert parsed["known_dataset"]["name"] == "DREAMER"
        assert parsed["quality_score"]["score"] == 85
        assert parsed["metadata"]["Number of Channels"] == 14
        # EEG-specific fields
        assert "events" in parsed

    def test_parse_ecg_report(self, sample_agent1_ecg_report):
        """Test parsing a MIT-BIH ECG report."""
        parsed = parse_agent1_report(sample_agent1_ecg_report)
        assert parsed["modality"] == "ECG"
        assert parsed["known_dataset"]["name"] == "MIT-BIH Arrhythmia"
        # ECG-specific fields
        assert "annotations" in parsed

    def test_parse_missing_modality(self):
        """Test ValueError when modality key is missing."""
        with pytest.raises(ValueError, match="missing required key"):
            parse_agent1_report({"metadata": {}})

    def test_parse_unsupported_modality(self):
        """Test ValueError for unsupported modality."""
        with pytest.raises(ValueError, match="Unsupported modality"):
            parse_agent1_report({"modality": "MEG"})

    def test_parse_minimal_report(self):
        """Test parsing a minimal valid report."""
        parsed = parse_agent1_report({"modality": "EEG"})
        assert parsed["modality"] == "EEG"
        assert parsed["metadata"] == {}
        assert parsed["quality_score"]["score"] == 0


class TestHelperFunctions:
    """Tests for helper utility functions."""

    def test_get_channel_count_eeg(self, sample_agent1_eeg_report):
        """Test extracting channel count from EEG report."""
        parsed = parse_agent1_report(sample_agent1_eeg_report)
        assert get_channel_count(parsed) == 14

    def test_get_channel_count_ecg(self, sample_agent1_ecg_report):
        """Test extracting lead count from ECG report."""
        parsed = parse_agent1_report(sample_agent1_ecg_report)
        assert get_channel_count(parsed) == 2

    def test_get_channel_count_string_value(self):
        """Test handling string channel count."""
        report = {"metadata": {"Number of Channels": "32"}}
        assert get_channel_count(report) == 32

    def test_get_channel_count_invalid_string(self):
        """Test handling invalid string channel count."""
        report = {"metadata": {"Number of Channels": "unknown"}}
        assert get_channel_count(report) == 0

    def test_get_quality_score(self, sample_agent1_eeg_report):
        """Test extracting quality score."""
        parsed = parse_agent1_report(sample_agent1_eeg_report)
        assert get_quality_score(parsed) == 85

    def test_get_quality_score_missing(self):
        """Test quality score extraction with missing data."""
        assert get_quality_score({}) == 0
