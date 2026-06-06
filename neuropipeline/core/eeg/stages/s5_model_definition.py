"""EEG Stage 5: Model Definition - define and configure the neural network architecture."""


def get_config_schema() -> dict:
    """Return the configuration schema for this stage."""
    return {
        "model_type": {
            "type": "str",
            "default": "gat_tcn",
            "description": "Model architecture to use",
            "options": ["gat_tcn", "eegnet", "cnn1d", "lstm", "transformer", "custom"],
        },
        "n_classes": {
            "type": "int",
            "default": 3,
            "description": "Number of output classes",
        },
        "n_channels": {
            "type": "int",
            "default": 14,
            "description": "Number of EEG channels (input dimension)",
        },
        "seq_length": {
            "type": "int",
            "default": 128,
            "description": "Sequence length (time samples per epoch)",
        },
        "hidden_dim": {
            "type": "int",
            "default": 64,
            "description": "Hidden layer dimension",
        },
        "num_layers": {
            "type": "int",
            "default": 3,
            "description": "Number of layers",
        },
        "dropout": {
            "type": "float",
            "default": 0.3,
            "description": "Dropout rate",
        },
        "weight_init": {
            "type": "str",
            "default": "kaiming",
            "description": "Weight initialization method",
            "options": ["kaiming", "xavier", "normal"],
        },
    }


def get_default_config(dataset_info: dict) -> dict:
    """Return default configuration based on dataset info."""
    metadata = dataset_info.get("metadata", {})
    n_ch = metadata.get("Number of Channels", 14)
    if isinstance(n_ch, str):
        n_ch = int(n_ch) if n_ch.isdigit() else 14

    known = dataset_info.get("known_dataset", None)
    dataset_name = known.get("name", "") if known and isinstance(known, dict) else ""

    # Default model selection
    model_type = "gat_tcn" if n_ch >= 8 else "cnn1d"

    config = {
        "model_type": model_type,
        "n_classes": 3,
        "n_channels": n_ch,
        "seq_length": 128,
        "hidden_dim": 64,
        "num_layers": 3,
        "dropout": 0.3,
        "weight_init": "kaiming",
    }

    return config


def generate_code(config: dict) -> str:
    """Generate Python code for the model definition stage."""
    model_type = config.get("model_type", "gat_tcn")

    if model_type == "gat_tcn":
        return _generate_gat_tcn(config)
    elif model_type == "eegnet":
        return _generate_eegnet(config)
    else:
        return _generate_generic_cnn(config)


def _generate_gat_tcn(config: dict) -> str:
    """Generate GAT-TCN model code."""
    return f'''"""Stage 5: Model Definition - GAT-TCN"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

N_CLASSES = {config.get("n_classes", 3)}
N_CHANNELS = {config.get("n_channels", 14)}
SEQ_LENGTH = {config.get("seq_length", 128)}
HIDDEN_DIM = {config.get("hidden_dim", 64)}
NUM_LAYERS = {config.get("num_layers", 3)}
DROPOUT = {config.get("dropout", 0.3)}


class TemporalConvBlock(nn.Module):
    """Temporal Convolutional Network block."""

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        return self.dropout(F.relu(self.bn(self.conv(x))))


class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer for EEG channel relationships."""

    def __init__(self, in_features, out_features, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = out_features // n_heads
        self.W = nn.Linear(in_features, out_features)
        self.a = nn.Linear(2 * self.head_dim, 1)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        # x shape: (batch, n_channels, features)
        h = self.W(x)
        batch, n_nodes, _ = h.shape
        h = h.view(batch, n_nodes, self.n_heads, self.head_dim)
        # Simplified attention (full connectivity between channels)
        attn = torch.mean(h, dim=-1, keepdim=True)
        attn = F.softmax(attn, dim=1)
        out = (h * attn).view(batch, n_nodes, -1)
        return self.dropout(out)


class GATTCN(nn.Module):
    """GAT-TCN: Graph Attention + Temporal Convolution Network for EEG."""

    def __init__(self):
        super().__init__()
        # Graph attention layers
        self.gat1 = GraphAttentionLayer(SEQ_LENGTH, HIDDEN_DIM)
        self.gat2 = GraphAttentionLayer(HIDDEN_DIM, HIDDEN_DIM)

        # Temporal convolution layers
        self.tcn = nn.Sequential(
            TemporalConvBlock(N_CHANNELS, 64, kernel_size=3, dilation=1),
            TemporalConvBlock(64, 128, kernel_size=3, dilation=2),
            TemporalConvBlock(128, HIDDEN_DIM, kernel_size=3, dilation=4),
        )

        # Classifier
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, N_CLASSES),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        """Forward pass.
        Input: (batch, n_channels, seq_length)
        Output: (batch, n_classes)
        """
        # GAT branch: treat channels as graph nodes
        gat_out = self.gat1(x)  # (batch, n_channels, hidden_dim)
        gat_out = self.gat2(gat_out)
        gat_feat = gat_out.mean(dim=1)  # (batch, hidden_dim)

        # TCN branch: temporal convolutions
        tcn_out = self.tcn(x)  # (batch, hidden_dim, seq_length)
        tcn_feat = self.pool(tcn_out).squeeze(-1)  # (batch, hidden_dim)

        # Combine branches
        combined = torch.cat([gat_feat, tcn_feat], dim=1)
        return self.fc(combined)


def create_model():
    """Create and return the model."""
    model = GATTCN()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created: GATTCN with {{n_params:,}} trainable parameters")
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = create_model()
    x = torch.randn(2, N_CHANNELS, SEQ_LENGTH)
    out = model(x)
    logger.info(f"Input shape: {{x.shape}}, Output shape: {{out.shape}}")
'''


def _generate_eegnet(config: dict) -> str:
    """Generate EEGNet model code."""
    return f'''"""Stage 5: Model Definition - EEGNet"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

N_CLASSES = {config.get("n_classes", 3)}
N_CHANNELS = {config.get("n_channels", 14)}
SEQ_LENGTH = {config.get("seq_length", 128)}
DROPOUT = {config.get("dropout", 0.5)}


class EEGNet(nn.Module):
    """EEGNet: Compact CNN for EEG-based BCIs."""

    def __init__(self, F1=8, D=2, F2=16, kernel_length=64):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (N_CHANNELS, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(DROPOUT),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(DROPOUT),
        )
        self.classifier = nn.Linear(F2 * (SEQ_LENGTH // 32), N_CLASSES)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """Input: (batch, n_channels, seq_length) -> Output: (batch, n_classes)"""
        x = x.unsqueeze(1)  # (batch, 1, n_channels, seq_length)
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(1)
        return self.classifier(x)


def create_model():
    """Create and return the model."""
    model = EEGNet()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created: EEGNet with {{n_params:,}} trainable parameters")
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = create_model()
    x = torch.randn(2, N_CHANNELS, SEQ_LENGTH)
    out = model(x)
    logger.info(f"Input shape: {{x.shape}}, Output shape: {{out.shape}}")
'''


def _generate_generic_cnn(config: dict) -> str:
    """Generate generic 1D CNN model code."""
    return f'''"""Stage 5: Model Definition - 1D CNN"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

N_CLASSES = {config.get("n_classes", 3)}
N_CHANNELS = {config.get("n_channels", 14)}
SEQ_LENGTH = {config.get("seq_length", 128)}
HIDDEN_DIM = {config.get("hidden_dim", 64)}
DROPOUT = {config.get("dropout", 0.3)}


class CNN1D(nn.Module):
    """1D Convolutional Neural Network for EEG classification."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(N_CHANNELS, 32, kernel_size=7, padding=3),
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
            nn.Linear(128, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, N_CLASSES),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """Input: (batch, n_channels, seq_length) -> Output: (batch, n_classes)"""
        x = self.features(x)
        x = x.squeeze(-1)
        return self.classifier(x)


def create_model():
    """Create and return the model."""
    model = CNN1D()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created: CNN1D with {{n_params:,}} trainable parameters")
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = create_model()
    x = torch.randn(2, N_CHANNELS, SEQ_LENGTH)
    out = model(x)
    logger.info(f"Input shape: {{x.shape}}, Output shape: {{out.shape}}")
'''
