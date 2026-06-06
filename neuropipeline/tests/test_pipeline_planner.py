"""Tests for pipeline planner - mapping Agent 1 reports to pipeline configs."""

import pytest

from neuropipeline.core.pipeline_planner import plan_pipeline


class TestPlanPipelineDREAMER:
    """Test pipeline planning for DREAMER dataset."""

    def test_dreamer_model_is_gat_tcn(self, sample_agent1_eeg_report):
        """DREAMER with 14 channels should recommend gat_tcn model."""
        config = plan_pipeline(sample_agent1_eeg_report)
        assert config["model"] == "gat_tcn"

    def test_dreamer_task_is_emotion_recognition(self, sample_agent1_eeg_report):
        """DREAMER should map to emotion_recognition task."""
        config = plan_pipeline(sample_agent1_eeg_report)
        assert config["task"] == "emotion_recognition"

    def test_dreamer_epoch_type_is_trial_based(self, sample_agent1_eeg_report):
        """DREAMER should use trial-based epochs."""
        config = plan_pipeline(sample_agent1_eeg_report)
        assert config["epoch_type"] == "trial-based"

    def test_dreamer_has_expected_labels(self, sample_agent1_eeg_report):
        """DREAMER should have arousal, valence, dominance labels."""
        config = plan_pipeline(sample_agent1_eeg_report)
        assert "arousal" in config["labels"]
        assert "valence" in config["labels"]

    def test_dreamer_faa_enabled(self, sample_agent1_eeg_report):
        """DREAMER emotion_recognition task should enable FAA features."""
        config = plan_pipeline(sample_agent1_eeg_report)
        assert config["features"]["faa"] is True

    def test_dreamer_ica_enabled(self, sample_agent1_eeg_report):
        """DREAMER with quality 85 and 14 channels should enable ICA."""
        config = plan_pipeline(sample_agent1_eeg_report)
        assert config["preprocessing"]["ica"]["enabled"] is True


class TestPlanPipelineMITBIH:
    """Test pipeline planning for MIT-BIH Arrhythmia ECG dataset."""

    def test_mitbih_model_is_cnn1d_lstm(self, sample_agent1_ecg_report):
        """MIT-BIH with 2 leads should recommend cnn1d_lstm."""
        config = plan_pipeline(sample_agent1_ecg_report)
        assert config["model"] == "cnn1d_lstm"

    def test_mitbih_task_is_arrhythmia_detection(self, sample_agent1_ecg_report):
        """MIT-BIH should map to arrhythmia_detection task."""
        config = plan_pipeline(sample_agent1_ecg_report)
        assert config["task"] == "arrhythmia_detection"

    def test_mitbih_epoch_type_is_beat_based(self, sample_agent1_ecg_report):
        """MIT-BIH should use beat-based epochs."""
        config = plan_pipeline(sample_agent1_ecg_report)
        assert config["epoch_type"] == "beat-based"

    def test_mitbih_ecg_stages(self, sample_agent1_ecg_report):
        """MIT-BIH should use ECG stage list."""
        config = plan_pipeline(sample_agent1_ecg_report)
        assert "s1_quality_check" in config["stages"]
        assert "s3_beat_segmentation" in config["stages"]
        assert "s9_clinical_interpretation" in config["stages"]


class TestPlanPipelineMotorImagery:
    """Test pipeline planning for Motor Imagery datasets."""

    def test_motor_imagery_event_based_epochs(self):
        """PhysioNet Motor Imagery should use event-based epochs."""
        report = {
            "modality": "EEG",
            "known_dataset": {"name": "PhysioNet Motor Imagery"},
            "quality_score": {"score": 75},
            "metadata": {"Number of Channels": 64},
            "artifacts": {},
        }
        config = plan_pipeline(report)
        assert config["epoch_type"] == "event-based"
        assert config["task"] == "motor_imagery"

    def test_bciciv_motor_imagery(self):
        """BCICIV should also be motor_imagery with event-based epochs."""
        report = {
            "modality": "EEG",
            "known_dataset": {"name": "BCICIV"},
            "quality_score": {"score": 80},
            "metadata": {"Number of Channels": 22},
            "artifacts": {},
        }
        config = plan_pipeline(report)
        assert config["epoch_type"] == "event-based"
        assert config["task"] == "motor_imagery"


class TestPlanPipelineLowChannel:
    """Test pipeline planning for low-channel-count EEG."""

    def test_low_channel_eeg_model(self):
        """EEG with fewer than 8 channels and motor_imagery should use eegnet."""
        report = {
            "modality": "EEG",
            "known_dataset": {"name": "BCICIV"},
            "quality_score": {"score": 70},
            "metadata": {"Number of Channels": 4},
            "artifacts": {},
        }
        config = plan_pipeline(report)
        # BCICIV is motor_imagery, which recommends eegnet
        assert config["model"] == "eegnet"

    def test_low_channel_unknown_dataset(self):
        """Low channel EEG without known dataset should use cnn1d."""
        report = {
            "modality": "EEG",
            "known_dataset": None,
            "quality_score": {"score": 60},
            "metadata": {"Number of Channels": 4},
            "artifacts": {},
        }
        config = plan_pipeline(report)
        assert config["model"] == "cnn1d"


class TestPlanPipelineICAConditions:
    """Test ICA enabling conditions."""

    def test_ica_enabled_high_quality_many_channels(self, sample_agent1_eeg_report):
        """ICA should be enabled when quality >= 70 and channels >= 8."""
        config = plan_pipeline(sample_agent1_eeg_report)
        assert config["preprocessing"]["ica"]["enabled"] is True

    def test_ica_disabled_low_quality(self):
        """ICA should be disabled when quality < 70."""
        report = {
            "modality": "EEG",
            "known_dataset": {"name": "DREAMER"},
            "quality_score": {"score": 50},
            "metadata": {"Number of Channels": 14},
            "artifacts": {},
        }
        config = plan_pipeline(report)
        assert config["preprocessing"]["ica"]["enabled"] is False

    def test_ica_disabled_few_channels(self):
        """ICA should be disabled when channels < 8."""
        report = {
            "modality": "EEG",
            "known_dataset": None,
            "quality_score": {"score": 85},
            "metadata": {"Number of Channels": 4},
            "artifacts": {},
        }
        config = plan_pipeline(report)
        assert config["preprocessing"]["ica"]["enabled"] is False


class TestPlanPipelineECGMultiLead:
    """Test ECG model selection based on lead count."""

    def test_ecg_12_lead_resnet1d(self):
        """ECG with 12 leads should recommend resnet1d."""
        report = {
            "modality": "ECG",
            "known_dataset": {"name": "PTB-XL"},
            "quality_score": {"score": 88},
            "metadata": {"Number of Leads": 12},
            "artifacts": {},
        }
        config = plan_pipeline(report)
        assert config["model"] == "resnet1d"

    def test_ecg_few_leads_cnn1d_lstm(self):
        """ECG with fewer than 12 leads should recommend cnn1d_lstm."""
        report = {
            "modality": "ECG",
            "known_dataset": {"name": "MIT-BIH Arrhythmia"},
            "quality_score": {"score": 90},
            "metadata": {"Number of Leads": 2},
            "artifacts": {},
        }
        config = plan_pipeline(report)
        assert config["model"] == "cnn1d_lstm"


class TestPlanPipelineFAA:
    """Test FAA feature inclusion for emotion recognition."""

    def test_faa_enabled_for_emotion(self, sample_agent1_eeg_report):
        """FAA should be enabled for emotion_recognition tasks."""
        config = plan_pipeline(sample_agent1_eeg_report)
        assert config["features"]["faa"] is True
        assert "asymmetry_pairs" in config["features"]

    def test_faa_disabled_for_motor_imagery(self):
        """FAA should be disabled for motor_imagery tasks."""
        report = {
            "modality": "EEG",
            "known_dataset": {"name": "PhysioNet Motor Imagery"},
            "quality_score": {"score": 80},
            "metadata": {"Number of Channels": 64},
            "artifacts": {},
        }
        config = plan_pipeline(report)
        assert config["features"]["faa"] is False
