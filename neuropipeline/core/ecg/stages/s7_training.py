"""ECG Stage 7: Training - model training with early stopping."""


def get_config_schema() -> dict:
    """Return the configuration schema for this stage."""
    return {
        "epochs": {
            "type": "int",
            "default": 100,
            "description": "Maximum training epochs",
        },
        "batch_size": {
            "type": "int",
            "default": 64,
            "description": "Training batch size",
        },
        "learning_rate": {
            "type": "float",
            "default": 0.001,
            "description": "Initial learning rate",
        },
        "optimizer": {
            "type": "str",
            "default": "adamw",
            "description": "Optimizer",
            "options": ["adam", "adamw", "sgd"],
        },
        "scheduler": {
            "type": "str",
            "default": "cosine",
            "description": "LR scheduler",
            "options": ["cosine", "step", "plateau", "none"],
        },
        "early_stopping_patience": {
            "type": "int",
            "default": 15,
            "description": "Early stopping patience",
        },
        "weight_decay": {
            "type": "float",
            "default": 1e-4,
            "description": "Weight decay",
        },
        "class_weights": {
            "type": "str",
            "default": "balanced",
            "description": "Class weighting",
            "options": ["balanced", "none"],
        },
        "label_smoothing": {
            "type": "float",
            "default": 0.1,
            "description": "Label smoothing factor",
        },
    }


def get_default_config(dataset_info: dict) -> dict:
    """Return default configuration based on dataset info."""
    config = {
        "epochs": 100,
        "batch_size": 64,
        "learning_rate": 0.001,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "early_stopping_patience": 15,
        "weight_decay": 1e-4,
        "class_weights": "balanced",
        "label_smoothing": 0.1,
    }
    return config


def generate_code(config: dict) -> str:
    """Generate Python code for the ECG training stage."""
    return f'''"""Stage 7: ECG Model Training"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
import logging

logger = logging.getLogger(__name__)

# Configuration
EPOCHS = {config.get("epochs", 100)}
BATCH_SIZE = {config.get("batch_size", 64)}
LEARNING_RATE = {config.get("learning_rate", 0.001)}
OPTIMIZER = "{config.get("optimizer", "adamw")}"
SCHEDULER = "{config.get("scheduler", "cosine")}"
EARLY_STOPPING_PATIENCE = {config.get("early_stopping_patience", 15)}
WEIGHT_DECAY = {config.get("weight_decay", 1e-4)}
CLASS_WEIGHTS = "{config.get("class_weights", "balanced")}"
LABEL_SMOOTHING = {config.get("label_smoothing", 0.1)}


def train_model(model, X_train, y_train, X_val, y_val, device="cpu"):
    """Train ECG classification model.

    Parameters
    ----------
    model : nn.Module
        Model to train
    X_train, y_train : np.ndarray
        Training data
    X_val, y_val : np.ndarray
        Validation data
    device : str
        Device

    Returns
    -------
    dict
        Training history
    """
    model = model.to(device)

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Loss with class weights
    if CLASS_WEIGHTS == "balanced":
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        criterion = nn.CrossEntropyLoss(
            weight=torch.FloatTensor(weights).to(device),
            label_smoothing=LABEL_SMOOTHING
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # Optimizer
    if OPTIMIZER == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # Scheduler
    if SCHEDULER == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    elif SCHEDULER == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    else:
        scheduler = None

    history = {{"train_loss": [], "val_loss": [], "val_acc": []}}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses, correct, total = [], 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                out = model(X_batch)
                loss = criterion(out, y_batch)
                val_losses.append(loss.item())
                _, pred = torch.max(out, 1)
                total += y_batch.size(0)
                correct += (pred == y_batch).sum().item()

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_acc = correct / total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if scheduler and SCHEDULER != "plateau":
            scheduler.step()
        elif scheduler:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            logger.info(f"Epoch {{epoch}}/{{EPOCHS}} - Loss: {{train_loss:.4f}} - Val Acc: {{val_acc:.4f}}")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping at epoch {{epoch}}")
            break

    model.load_state_dict(best_state)
    logger.info(f"Training complete. Best val loss: {{best_val_loss:.4f}}")
    return history


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Training stage - requires model and data")
'''
