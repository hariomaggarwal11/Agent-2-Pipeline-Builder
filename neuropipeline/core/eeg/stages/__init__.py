"""EEG pipeline stage modules."""

from . import (
    s1_quality_check,
    s2_preprocessing,
    s3_feature_extraction,
    s4_dataset_split,
    s5_model_definition,
    s6_training,
    s7_evaluation,
    s8_interpretation,
)

STAGES = {
    "s1_quality_check": s1_quality_check,
    "s2_preprocessing": s2_preprocessing,
    "s3_feature_extraction": s3_feature_extraction,
    "s4_dataset_split": s4_dataset_split,
    "s5_model_definition": s5_model_definition,
    "s6_training": s6_training,
    "s7_evaluation": s7_evaluation,
    "s8_interpretation": s8_interpretation,
}
