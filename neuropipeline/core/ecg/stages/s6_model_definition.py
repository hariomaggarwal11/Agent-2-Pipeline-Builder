"""ECG Stage 6: Model Definition - define neural network for ECG classification."""


def get_config_schema() -> dict:
    """Return the configuration schema for this stage."""
    return {
        "model_type": {
            "type": "str",
            "default": "resnet1d",
            "description": "Model architecture",
            "options": ["resnet1d", "cnn1d_lstm", "cnn1d", "lstm", "transformer", "custom"],
        },
        "n_classes": {
            "type": "int",
            "default": 5,
            "description": "Number of output classes",
        },
        "n_leads": {
            "type": "int",
            "default": 12,
            "description": "Number of ECG leads (input channels)",
        },
        "seq_length": {
            "type": "int",
            "default": 256,
            "description": "Sequence length (samples per beat)",
        },
        "base_filters": {
            "type": "int",
            "default": 64,
            "description": "Base number of convolutional filters",
        },
        "dropout": {
            "type": "float",
            "default": 0.2,
            "description": "Dropout rate",
        },
    }


def get_default_config(dataset_info: dict) -> dict:
    """Return default configuration based on dataset info."""
    metadata = dataset_info.get("metadata", {})
    n_leads = metadata.get("Number of Leads", 12)
    if isinstance(n_leads, str):
        n_leads = int(n_leads) if n_leads.isdigit() else 12

    model_type = "resnet1d" if n_leads >= 12 else "cnn1d_lstm"

    config = {
        "model_type": model_type,
        "n_classes": 5,
        "n_leads": n_leads,
        "seq_length": 256,
        "base_filters": 64,
        "dropout": 0.2,
    }
    return config


def generate_code(config: dict) -> str:
    """Generate Python code for the ECG model definition stage."""
    model_type = config.get("model_type", "resnet1d")

    if model_type == "resnet1d":
        return _generate_resnet1d(config)
    elif model_type == "cnn1d_lstm":
        return _generate_cnn1d_lstm(config)
    else:
        return _generate_cnn1d(config)


def _generate_resnet1d(config: dict) -> str:
    """Generate ResNet-1D model code."""
    return f'''"""Stage 6: ECG Model Definition - ResNet-1D"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

N_CLASSES = {config.get("n_classes", 5)}
N_LEADS = {config.get("n_leads", 12)}
SEQ_LENGTH = {config.get("seq_length", 256)}
BASE_FILTERS = {config.get("base_filters", 64)}
DROPOUT = {config.get("dropout", 0.2)}


class ResidualBlock(nn.Module):
    """1D Residual block."""

    def __init__(self, in_channels, out_channels, kernel_size=15, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(DROPOUT)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)


class ResNet1D(nn.Module):
    """1D ResNet for multi-lead ECG classification."""

    def __init__(self):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv1d(N_LEADS, BASE_FILTERS, kernel_size=15, padding=7),
            nn.BatchNorm1d(BASE_FILTERS),
            nn.ReLU(),
        )
        self.layer1 = self._make_layer(BASE_FILTERS, BASE_FILTERS, 2, stride=2)
        self.layer2 = self._make_layer(BASE_FILTERS, BASE_FILTERS * 2, 2, stride=2)
        self.layer3 = self._make_layer(BASE_FILTERS * 2, BASE_FILTERS * 4, 2, stride=2)
        self.layer4 = self._make_layer(BASE_FILTERS * 4, BASE_FILTERS * 8, 2, stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(BASE_FILTERS * 8, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, N_CLASSES),
        )
        self._init_weights()

    def _make_layer(self, in_channels, out_channels, n_blocks, stride):
        layers = [ResidualBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, n_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """Input: (batch, n_leads, seq_length) -> Output: (batch, n_classes)"""
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


def create_model():
    """Create and return the model."""
    model = ResNet1D()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: ResNet1D, Parameters: {{n_params:,}}")
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = create_model()
    x = torch.randn(2, N_LEADS, SEQ_LENGTH)
    out = model(x)
    logger.info(f"Input: {{x.shape}}, Output: {{out.shape}}")
'''


def _generate_cnn1d_lstm(config: dict) -> str:
    """Generate CNN-LSTM hybrid model code."""
    return f'''"""Stage 6: ECG Model Definition - CNN-LSTM"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

N_CLASSES = {config.get("n_classes", 5)}
N_LEADS = {config.get("n_leads", 12)}
SEQ_LENGTH = {config.get("seq_length", 256)}
DROPOUT = {config.get("dropout", 0.3)}


class CNNLSTM(nn.Module):
    """Hybrid CNN-LSTM for ECG classification."""

    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(N_LEADS, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT,
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, N_CLASSES),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """Input: (batch, n_leads, seq_length) -> Output: (batch, n_classes)"""
        x = self.cnn(x)  # (batch, 64, seq_length/4)
        x = x.permute(0, 2, 1)  # (batch, seq_length/4, 64)
        lstm_out, _ = self.lstm(x)  # (batch, seq_length/4, 128)
        x = lstm_out[:, -1, :]  # Last timestep
        return self.fc(x)


def create_model():
    """Create and return the model."""
    model = CNNLSTM()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: CNN-LSTM, Parameters: {{n_params:,}}")
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = create_model()
    x = torch.randn(2, N_LEADS, SEQ_LENGTH)
    out = model(x)
    logger.info(f"Input: {{x.shape}}, Output: {{out.shape}}")
'''


def _generate_cnn1d(config: dict) -> str:
    """Generate basic 1D CNN model code."""
    return f'''"""Stage 6: ECG Model Definition - 1D CNN"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

N_CLASSES = {config.get("n_classes", 5)}
N_LEADS = {config.get("n_leads", 12)}
SEQ_LENGTH = {config.get("seq_length", 256)}
DROPOUT = {config.get("dropout", 0.3)}


class CNN1D(nn.Module):
    """1D CNN for ECG classification."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(N_LEADS, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, N_CLASSES),
        )

    def forward(self, x):
        """Input: (batch, n_leads, seq_length) -> Output: (batch, n_classes)"""
        x = self.features(x)
        x = x.squeeze(-1)
        return self.classifier(x)


def create_model():
    model = CNN1D()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: CNN1D, Parameters: {{n_params:,}}")
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = create_model()
    x = torch.randn(2, N_LEADS, SEQ_LENGTH)
    out = model(x)
    logger.info(f"Input: {{x.shape}}, Output: {{out.shape}}")
'''
