"""Group Phi M3: feedforward neural network on TF-IDF features.

Uses the same build_features() pipeline and group-aware split from M2.
"""

from time import perf_counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from data_pipeline import build_features

SEED = 47
LABELS = ("Negative", "Neutral", "Positive")
LABEL_MAP = {"Negative": 0, "Neutral": 1, "Positive": 2}


class SentimentFNN(nn.Module):
    """Two hidden-layer feedforward network with dropout."""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def main():
    bundle = build_features()

    # encode string labels to integers for pytorch
    y_train_enc = np.array([LABEL_MAP[l] for l in bundle.y_train])
    y_test_enc = np.array([LABEL_MAP[l] for l in bundle.y_test])

    # hold out 15% of training for validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        bundle.X_train_tfidf, y_train_enc,
        test_size=0.15, random_state=SEED, stratify=y_train_enc,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # convert sparse matrices to tensors
    X_tr_t = torch.tensor(X_tr.toarray(), dtype=torch.float32)
    X_val_t = torch.tensor(X_val.toarray(), dtype=torch.float32)
    X_test_t = torch.tensor(bundle.X_test_tfidf.toarray(), dtype=torch.float32)

    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    y_test_t = torch.tensor(y_test_enc, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t), batch_size=256, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t), batch_size=512
    )

    input_dim = X_tr_t.shape[1]
    model = SentimentFNN(input_dim, len(LABELS)).to(device)
    print(f"\n{model}\n")

    # class weights to handle imbalance (Neutral is underrepresented)
    class_counts = np.bincount(y_tr)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights_t)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # training loop
    num_epochs = 30
    train_losses = []
    val_losses = []

    start_time = perf_counter()

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:2d}/{num_epochs}  "
                f"Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}"
            )

    training_time = perf_counter() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")

    # evaluate on held-out test set
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t.to(device))
        fnn_predictions = logits.argmax(dim=1).cpu().numpy()

    fnn_accuracy = accuracy_score(y_test_enc, fnn_predictions)
    fnn_f1 = f1_score(y_test_enc, fnn_predictions, average="macro", zero_division=0)

    print(f"\nFeedforward NN (TF-IDF) Accuracy: {fnn_accuracy:.2%} | Macro-F1: {fnn_f1:.4f}")
    print(f"Training time: {training_time:.4f} seconds")
    print(
        classification_report(
            y_test_enc, fnn_predictions,
            target_names=list(LABELS), zero_division=0,
        )
    )
    fnn_cm = confusion_matrix(
        y_test_enc, fnn_predictions, labels=[0, 1, 2]
    )
    print(fnn_cm)

    # loss curves
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, num_epochs + 1), train_losses, label="Training Loss", linewidth=2)
    ax.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Feedforward NN — Training vs. Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("fnn_loss_curves.png", dpi=150)
    plt.close(fig)
    print("Saved fnn_loss_curves.png")

    torch.save(model.state_dict(), "outputs/models/fnn_model.pt")
    print("Saved outputs/models/fnn_model.pt")


if __name__ == "__main__":
    main()