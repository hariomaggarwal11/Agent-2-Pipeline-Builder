"""Stage configurator - dynamic config panel for pipeline stages."""


def render_stage_config(stage_name, modality, current_config):
    """Render the configuration panel for a specific pipeline stage.

    Dynamically renders different configuration options based on the
    selected stage and modality using Streamlit widgets.

    Args:
        stage_name: The stage key (e.g., 's1_quality_check').
        modality: 'eeg' or 'ecg'.
        current_config: Dict of current configuration values for this stage.

    Returns:
        Updated configuration dict with user-selected values.
    """
    import streamlit as st

    config = dict(current_config) if current_config else {}

    st.markdown(
        f'<h3>Configure: {_format_stage_name(stage_name)}</h3>',
        unsafe_allow_html=True,
    )

    # Dispatch to stage-specific configurator
    if "quality_check" in stage_name:
        config = _config_quality_check(config, modality)
    elif "preprocessing" in stage_name:
        config = _config_preprocessing(config, modality)
    elif "beat_segmentation" in stage_name:
        config = _config_beat_segmentation(config)
    elif "feature_extraction" in stage_name:
        config = _config_feature_extraction(config, modality)
    elif "dataset_split" in stage_name:
        config = _config_dataset_split(config)
    elif "model_definition" in stage_name:
        config = _config_model_definition(config, modality)
    elif "training" in stage_name:
        config = _config_training(config)
    elif "evaluation" in stage_name:
        config = _config_evaluation(config, modality)
    elif "interpretation" in stage_name or "clinical_interpretation" in stage_name:
        config = _config_interpretation(config, modality)
    else:
        st.info(f"No specific configurator for stage: {stage_name}")

    return config


def _config_quality_check(config, modality):
    """Configuration for quality check stage."""
    import streamlit as st

    st.markdown("**Quality Thresholds**")

    config["quality_threshold"] = st.slider(
        "Minimum quality score",
        min_value=0.0,
        max_value=1.0,
        value=config.get("quality_threshold", 0.7),
        step=0.05,
        key="cfg_quality_threshold",
    )

    config["reject_bad_channels"] = st.checkbox(
        "Auto-reject bad channels",
        value=config.get("reject_bad_channels", True),
        key="cfg_reject_bad_channels",
    )

    if modality == "eeg":
        config["max_bad_channels"] = st.number_input(
            "Max bad channels allowed",
            min_value=0,
            max_value=20,
            value=config.get("max_bad_channels", 5),
            key="cfg_max_bad_channels",
        )
        config["amplitude_threshold_uv"] = st.number_input(
            "Amplitude threshold (uV)",
            min_value=50,
            max_value=500,
            value=config.get("amplitude_threshold_uv", 150),
            step=10,
            key="cfg_amplitude_threshold",
        )
    else:
        config["snr_threshold"] = st.number_input(
            "SNR threshold (dB)",
            min_value=0.0,
            max_value=30.0,
            value=config.get("snr_threshold", 10.0),
            step=0.5,
            key="cfg_snr_threshold",
        )

    return config


def _config_preprocessing(config, modality):
    """Configuration for preprocessing stage."""
    import streamlit as st

    st.markdown("**Filter Settings**")

    config["highpass_freq"] = st.number_input(
        "High-pass filter (Hz)",
        min_value=0.0,
        max_value=10.0,
        value=config.get("highpass_freq", 0.5),
        step=0.1,
        key="cfg_highpass",
    )

    config["lowpass_freq"] = st.number_input(
        "Low-pass filter (Hz)",
        min_value=10.0,
        max_value=500.0,
        value=config.get("lowpass_freq", 45.0 if modality == "eeg" else 150.0),
        step=1.0,
        key="cfg_lowpass",
    )

    config["notch_filter"] = st.checkbox(
        "Apply notch filter (50/60 Hz)",
        value=config.get("notch_filter", True),
        key="cfg_notch",
    )

    if config["notch_filter"]:
        config["notch_freq"] = st.radio(
            "Notch frequency",
            options=[50, 60],
            index=0 if config.get("notch_freq", 50) == 50 else 1,
            key="cfg_notch_freq",
        )

    if modality == "eeg":
        st.markdown("**Artifact Removal**")
        config["use_ica"] = st.checkbox(
            "Apply ICA for artifact removal",
            value=config.get("use_ica", True),
            key="cfg_ica",
        )
        if config.get("use_ica"):
            config["n_ica_components"] = st.slider(
                "Number of ICA components",
                min_value=5,
                max_value=64,
                value=config.get("n_ica_components", 20),
                key="cfg_n_ica",
            )
        config["reference"] = st.radio(
            "Re-reference method",
            options=["average", "Cz", "linked_mastoids"],
            index=["average", "Cz", "linked_mastoids"].index(
                config.get("reference", "average")
            ),
            key="cfg_reference",
        )

    return config


def _config_beat_segmentation(config):
    """Configuration for ECG beat segmentation stage."""
    import streamlit as st

    st.markdown("**Beat Detection Settings**")

    config["detector"] = st.radio(
        "R-peak detector algorithm",
        options=["pan_tompkins", "hamilton", "engzee", "christov"],
        index=["pan_tompkins", "hamilton", "engzee", "christov"].index(
            config.get("detector", "pan_tompkins")
        ),
        key="cfg_detector",
    )

    config["pre_beat_ms"] = st.slider(
        "Pre-beat window (ms)",
        min_value=50,
        max_value=400,
        value=config.get("pre_beat_ms", 200),
        step=10,
        key="cfg_pre_beat",
    )

    config["post_beat_ms"] = st.slider(
        "Post-beat window (ms)",
        min_value=100,
        max_value=600,
        value=config.get("post_beat_ms", 400),
        step=10,
        key="cfg_post_beat",
    )

    config["normalize_beats"] = st.checkbox(
        "Normalize beat amplitudes",
        value=config.get("normalize_beats", True),
        key="cfg_normalize_beats",
    )

    return config


def _config_feature_extraction(config, modality):
    """Configuration for feature extraction stage."""
    import streamlit as st

    st.markdown("**Feature Selection**")

    if modality == "eeg":
        eeg_features = [
            "bandpower",
            "spectral_entropy",
            "hjorth_parameters",
            "wavelet_coefficients",
            "connectivity",
            "asymmetry",
            "statistical",
        ]
        config["feature_types"] = st.multiselect(
            "Feature types to extract",
            options=eeg_features,
            default=config.get(
                "feature_types", ["bandpower", "spectral_entropy", "hjorth_parameters"]
            ),
            key="cfg_eeg_features",
        )

        st.markdown("**Epoch Parameters**")
        config["epoch_duration_s"] = st.slider(
            "Epoch duration (seconds)",
            min_value=0.5,
            max_value=10.0,
            value=config.get("epoch_duration_s", 2.0),
            step=0.5,
            key="cfg_epoch_duration",
        )
        config["epoch_overlap"] = st.slider(
            "Epoch overlap ratio",
            min_value=0.0,
            max_value=0.9,
            value=config.get("epoch_overlap", 0.5),
            step=0.1,
            key="cfg_epoch_overlap",
        )
    else:
        ecg_features = [
            "rr_intervals",
            "hrv_time_domain",
            "hrv_frequency_domain",
            "morphological",
            "wavelet",
            "statistical",
        ]
        config["feature_types"] = st.multiselect(
            "Feature types to extract",
            options=ecg_features,
            default=config.get(
                "feature_types", ["rr_intervals", "hrv_time_domain", "morphological"]
            ),
            key="cfg_ecg_features",
        )

    return config


def _config_dataset_split(config):
    """Configuration for dataset split stage."""
    import streamlit as st

    st.markdown("**Split Strategy**")

    config["split_method"] = st.radio(
        "Split method",
        options=["subject_independent", "stratified_kfold", "random"],
        index=["subject_independent", "stratified_kfold", "random"].index(
            config.get("split_method", "subject_independent")
        ),
        key="cfg_split_method",
    )

    if config["split_method"] == "stratified_kfold":
        config["n_folds"] = st.slider(
            "Number of folds",
            min_value=2,
            max_value=10,
            value=config.get("n_folds", 5),
            key="cfg_n_folds",
        )
    else:
        config["train_ratio"] = st.slider(
            "Training set ratio",
            min_value=0.5,
            max_value=0.9,
            value=config.get("train_ratio", 0.7),
            step=0.05,
            key="cfg_train_ratio",
        )
        config["val_ratio"] = st.slider(
            "Validation set ratio",
            min_value=0.05,
            max_value=0.3,
            value=config.get("val_ratio", 0.15),
            step=0.05,
            key="cfg_val_ratio",
        )

    config["random_seed"] = st.number_input(
        "Random seed",
        min_value=0,
        max_value=99999,
        value=config.get("random_seed", 42),
        key="cfg_random_seed",
    )

    return config


def _config_model_definition(config, modality):
    """Configuration for model definition stage."""
    import streamlit as st

    from neuropipeline.config import MODEL_REGISTRY

    st.markdown("**Model Architecture**")

    model_options = list(MODEL_REGISTRY.keys())
    model_names = [MODEL_REGISTRY[k]["name"] for k in model_options]

    current_model = config.get("model_key", "eegnet" if modality == "eeg" else "cnn1d")
    current_idx = model_options.index(current_model) if current_model in model_options else 0

    selected_idx = model_names.index(
        st.selectbox(
            "Model architecture",
            options=model_names,
            index=current_idx,
            key="cfg_model_arch",
        )
    )
    config["model_key"] = model_options[selected_idx]

    # Show model-specific hyperparameters
    model_info = MODEL_REGISTRY[config["model_key"]]
    st.markdown(f"*{model_info['description']}*")

    if model_info["hyperparameters"]:
        st.markdown("**Hyperparameters**")
        hyper_config = config.get("hyperparameters", dict(model_info["hyperparameters"]))

        for param_name, default_value in model_info["hyperparameters"].items():
            if isinstance(default_value, int):
                hyper_config[param_name] = st.number_input(
                    param_name,
                    value=hyper_config.get(param_name, default_value),
                    key=f"cfg_hp_{param_name}",
                )
            elif isinstance(default_value, float):
                hyper_config[param_name] = st.number_input(
                    param_name,
                    value=hyper_config.get(param_name, default_value),
                    format="%.4f",
                    key=f"cfg_hp_{param_name}",
                )
            elif isinstance(default_value, bool):
                hyper_config[param_name] = st.checkbox(
                    param_name,
                    value=hyper_config.get(param_name, default_value),
                    key=f"cfg_hp_{param_name}",
                )
            elif isinstance(default_value, list):
                hyper_config[param_name] = st.text_input(
                    f"{param_name} (comma-separated)",
                    value=str(hyper_config.get(param_name, default_value)),
                    key=f"cfg_hp_{param_name}",
                )

        config["hyperparameters"] = hyper_config

    return config


def _config_training(config):
    """Configuration for training stage."""
    import streamlit as st

    st.markdown("**Training Parameters**")

    config["epochs"] = st.slider(
        "Number of epochs",
        min_value=5,
        max_value=500,
        value=config.get("epochs", 100),
        step=5,
        key="cfg_epochs",
    )

    config["batch_size"] = st.selectbox(
        "Batch size",
        options=[8, 16, 32, 64, 128, 256],
        index=[8, 16, 32, 64, 128, 256].index(config.get("batch_size", 32)),
        key="cfg_batch_size",
    )

    config["learning_rate"] = st.number_input(
        "Learning rate",
        min_value=0.00001,
        max_value=0.1,
        value=config.get("learning_rate", 0.001),
        format="%.5f",
        step=0.0001,
        key="cfg_lr",
    )

    config["optimizer"] = st.radio(
        "Optimizer",
        options=["adam", "adamw", "sgd", "rmsprop"],
        index=["adam", "adamw", "sgd", "rmsprop"].index(
            config.get("optimizer", "adam")
        ),
        key="cfg_optimizer",
    )

    config["scheduler"] = st.radio(
        "Learning rate scheduler",
        options=["none", "cosine", "step", "plateau"],
        index=["none", "cosine", "step", "plateau"].index(
            config.get("scheduler", "cosine")
        ),
        key="cfg_scheduler",
    )

    config["early_stopping"] = st.checkbox(
        "Enable early stopping",
        value=config.get("early_stopping", True),
        key="cfg_early_stopping",
    )

    if config.get("early_stopping"):
        config["patience"] = st.slider(
            "Early stopping patience",
            min_value=3,
            max_value=50,
            value=config.get("patience", 10),
            key="cfg_patience",
        )

    return config


def _config_evaluation(config, modality):
    """Configuration for evaluation stage."""
    import streamlit as st

    st.markdown("**Evaluation Strategy**")

    config["metrics"] = st.multiselect(
        "Evaluation metrics",
        options=["accuracy", "f1_score", "auc_roc", "precision", "recall", "kappa"],
        default=config.get("metrics", ["accuracy", "f1_score", "auc_roc"]),
        key="cfg_metrics",
    )

    config["average_method"] = st.radio(
        "Multi-class averaging",
        options=["macro", "weighted", "micro"],
        index=["macro", "weighted", "micro"].index(
            config.get("average_method", "macro")
        ),
        key="cfg_average_method",
    )

    config["generate_confusion_matrix"] = st.checkbox(
        "Generate confusion matrix",
        value=config.get("generate_confusion_matrix", True),
        key="cfg_confusion_matrix",
    )

    config["generate_roc_curve"] = st.checkbox(
        "Generate ROC curve",
        value=config.get("generate_roc_curve", True),
        key="cfg_roc_curve",
    )

    if modality == "ecg":
        st.markdown("**Clinical Metrics**")
        config["compute_clinical_metrics"] = st.checkbox(
            "Compute clinical metrics (Sensitivity, Specificity, PPV, NPV)",
            value=config.get("compute_clinical_metrics", True),
            key="cfg_clinical_metrics",
        )

    return config


def _config_interpretation(config, modality):
    """Configuration for interpretation/clinical interpretation stage."""
    import streamlit as st

    st.markdown("**Interpretation Settings**")

    if modality == "ecg":
        config["generate_clinical_report"] = st.checkbox(
            "Generate clinical report",
            value=config.get("generate_clinical_report", True),
            key="cfg_clinical_report",
        )
        config["confidence_threshold"] = st.slider(
            "Confidence threshold for clinical flagging",
            min_value=0.5,
            max_value=0.99,
            value=config.get("confidence_threshold", 0.9),
            step=0.01,
            key="cfg_confidence_threshold",
        )
    else:
        config["generate_topomaps"] = st.checkbox(
            "Generate topographic maps",
            value=config.get("generate_topomaps", True),
            key="cfg_topomaps",
        )
        config["feature_importance"] = st.checkbox(
            "Compute feature importance",
            value=config.get("feature_importance", True),
            key="cfg_feature_importance",
        )

    config["export_format"] = st.radio(
        "Export format",
        options=["pdf", "html", "both"],
        index=["pdf", "html", "both"].index(config.get("export_format", "pdf")),
        key="cfg_export_format",
    )

    return config


def _format_stage_name(stage_key):
    """Convert a stage key to a display-friendly name."""
    parts = stage_key.split("_", 1)
    if len(parts) > 1:
        name = parts[1]
    else:
        name = parts[0]
    return name.replace("_", " ").title()
