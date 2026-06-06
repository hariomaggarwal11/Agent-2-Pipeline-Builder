"""ECG pipeline stage modules."""

from . import (
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

STAGES = {
    "s1_quality_check": s1_quality_check,
    "s2_preprocessing": s2_preprocessing,
    "s3_beat_segmentation": s3_beat_segmentation,
    "s4_feature_extraction": s4_feature_extraction,
    "s5_dataset_split": s5_dataset_split,
    "s6_model_definition": s6_model_definition,
    "s7_training": s7_training,
    "s8_evaluation": s8_evaluation,
    "s9_clinical_interpretation": s9_clinical_interpretation,
}
