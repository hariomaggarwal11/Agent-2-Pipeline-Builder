"""EEG Stage 6: Training - model training with early stopping and logging."""


def get_config_schema() -> dict:
    """Return the configuration schema for this stage."""
    return {
        "epochs": {
            "type": "int",
            "default": 100,
            "description": "Maximum number of training epochs",
        },
        "batch_size": {
            "type": "int",
            "default": 32,
            "description": "Training batch size",
        },
        "learning_rate": {
            "type": "float",
            "default": 0.001,
            "description": "Initial learning rate",
        },
        "optimizer": {
            "type": "str",
            "default": "adam",
            "description": "Optimizer",
            "options": ["adam", "adamw", "sgd", "radam"],
        },
        "scheduler": {
            "type": "str",
            "default": "cosine",
            "description": "LR scheduler",
            "options": ["cosine", "step", "plateau", "none"],
        },
        "early_stopping_patience": {
            "type": "int",
            "default": 10,
            "description": "Early stopping patience (epochs)",
        },
        "weight_decay": {
            "type": "float",
            "default": 1e-4,
            "description": "L2 regularization weight decay",
        },
        "class_weights": {
            "type": "str",
            "default": "balanced",
            "description": "Class weighting strategy",
            "options": ["balanced", "none", "custom"],
        },
        "mixed_precision": {
            "type": "bool",
            "default": False,
            "description": "Enable mixed precision training (FP16)",
        },
    }


def get_default_config(dataset_info: dict) -> dict:
    """Return default configuration based on dataset info."""
    config = {
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "scheduler": "cosine",
        "early_stopping_patience": 10,
        "weight_decay": 1e-4,
        "class_weights": "balanced",
        "mixed_precision": False,
    }
    return config


def generate_code(config: dict) -> str:
    """Generate Python code for the training stage."""
    return f'''"""Stage 6: EEG Model Training"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
import logging

logger = logging.getLogger(__name__)

# Configuration
EPOCHS = {config.get("epochs", 100)}
BATCH_SIZE = {config.get("batch_size", 32)}
LEARNING_RATE = {config.get("learning_rate", 0.001)}
OPTIMIZER = "{config.get("optimizer", "adam")}"
SCHEDULER = "{config.get("scheduler", "cosine")}"
EARLY_STOPPING_PATIENCE = {config.get("early_stopping_patience", 10)}
WEIGHT_DECAY = {config.get("weight_decay", 1e-4)}
CLASS_WEIGHTS = "{config.get("class_weights", "balanced")}"
MIXED_PRECISION = {config.get("mixed_precision", False)}


def get_optimizer(model):
    """Create optimizer."""
    if OPTIMIZER == "adam":
        return torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == "sgd":
        return torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == "radam":
        return torch.optim.RAdam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


def get_scheduler(optimizer, n_steps):
    """Create learning rate scheduler."""
    if SCHEDULER == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)
    elif SCHEDULER == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif SCHEDULER == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    return None


def train_model(model, X_train, y_train, X_val, y_val, device="cpu"):
    """Train the model with early stopping.

    Parameters
    ----------
    model : nn.Module
        Model to train
    X_train, y_train : np.ndarray
        Training data and labels
    X_val, y_val : np.ndarray
        Validation data and labels
    device : str
        Device (cpu or cuda)

    Returns
    -------
    dict
        Training history with loss and metrics
    """
    model = model.to(device)

    # Prepare dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Loss function with class weights
    if CLASS_WEIGHTS == "balanced":
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer, EPOCHS)

    # Training loop
    history = {{"train_loss": [], "val_loss": [], "val_acc": []}}
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation phase
        model.eval()
        val_losses = []
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_losses.append(loss.item())
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_acc = correct / total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if scheduler and SCHEDULER != "plateau":
            scheduler.step()
        elif scheduler and SCHEDULER == "plateau":
            scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            logger.info(f"Epoch {{epoch}}/{{EPOCHS}} - Loss: {{train_loss:.4f}} - Val Loss: {{val_loss:.4f}} - Val Acc: {{val_acc:.4f}}")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping at epoch {{epoch}}")
            break

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)

    logger.info(f"Training complete. Best val loss: {{best_val_loss:.4f}}")
    return history


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Training stage - requires model, features, and labels")
'''
