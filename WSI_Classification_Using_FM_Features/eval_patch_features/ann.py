import torch
import torch.nn.functional as F
import numpy as np
import random, os
import time
from collections import defaultdict
from typing import Tuple, Dict, Any, List
from warnings import simplefilter
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from .metrics import get_eval_metrics
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# Define Focal Loss
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        BCE_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.float()
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

# ANN Binary Classifier
class ANNBinaryClassifier:
    def __init__(self, input_dim=512, hidden_dim=512, max_iter=100, C=1.0, verbose=True):
        self.C = C
        self.loss_func = FocalLoss()  # Class imbalance handling
        self.max_iter = max_iter
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Define the model
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(self.hidden_dim),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(self.hidden_dim, 1)  # Output layer for binary classification
        ).to(self.device)

    def compute_loss(self, preds, labels):
        loss = self.loss_func(preds, labels)
        wreg = 0.5 * sum((param.norm(p=2) for param in self.model.parameters()))  # L2 regularization
        return loss.mean() + (1.0 / self.C) * wreg

    def predict_proba(self, feats):
        feats = feats.to(self.device)
        self.model.eval()
        with torch.no_grad():
            return torch.sigmoid(self.model(feats))

    def fit(self, train_feats, train_labels, val_feats=None, val_labels=None, combine_trainval=False):
        train_feats = train_feats.to(self.device)
        train_labels = train_labels.to(self.device)
        if val_feats is not None:
            val_feats = val_feats.to(self.device)
            val_labels = val_labels.to(self.device)
        if combine_trainval and val_feats is not None:
            train_feats = torch.cat([train_feats, val_feats], dim=0)
            train_labels = torch.cat([train_labels, val_labels], dim=0)
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)

        train_loss_history = []
        val_loss_history = []

        best_val_loss = float("inf")
        patience = 10
        epochs_no_improve = 0

        for epoch in range(self.max_iter):
            # Training phase
            self.model.train()
            preds = self.model(train_feats).squeeze(-1)
            loss = self.compute_loss(preds, train_labels.float())
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss_history.append(loss.item())

            # Validation phase
            val_loss = None
            if val_feats is not None and combine_trainval is not True:
                self.model.eval()
                with torch.no_grad():
                    val_preds = self.model(val_feats).squeeze(-1)
                    val_loss = self.compute_loss(val_preds, val_labels.float())
                val_loss_history.append(val_loss.item())

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch}")
                        break

            scheduler.step()
            if self.verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss: {loss:.3f}, Val Loss: {val_loss:.3f}" if val_loss else f"Epoch {epoch}: Loss: {loss:.3f}")

        return train_loss_history, val_loss_history

# Training and Evaluation Functions
def eval_ANN(
    fold: int,
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    valid_feats: torch.Tensor,
    valid_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    input_dim: int = 512,
    hidden_dim: int = 512,
    max_iter: int = 1000,
    combine_trainval: bool = False,
    model_save_path: str="",
    verbose: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if verbose:
        print(f"Train Shape: {train_feats.shape}, Validation Shape: {valid_feats.shape}, Test Shape: {test_feats.shape}")

    classifier = ANNBinaryClassifier(input_dim=input_dim,hidden_dim=hidden_dim, max_iter=max_iter, verbose=verbose)
    train_loss, val_loss = classifier.fit(train_feats, train_labels, valid_feats, valid_labels, combine_trainval)

    model_path = os.path.join(model_save_path, f"fold{fold}_trained_ann_model_{input_dim}.pth")
    torch.save(classifier.model.state_dict(), model_path)
    print(f"Model saved at: {model_path}")

    # Testing phase
    probs_all = classifier.predict_proba(test_feats).squeeze(-1).cpu().numpy()
    preds_all = (probs_all > 0.5).astype(int)
    targets_all = test_labels.cpu().numpy()

    # Metrics and Plots
    eval_metrics = get_eval_metrics(targets_all, preds_all, probs_all)
    dump = {"preds_all": preds_all, "probs_all": probs_all, "targets_all": targets_all}
    # print confusion matrix
    cm = confusion_matrix(targets_all, preds_all)
    print("Confusion Matrix:")
    print(cm)
    if verbose:
        plot_training_logs({"train_loss": train_loss, "valid_loss": val_loss})
        plot_roc_auc(targets_all, probs_all)

    return eval_metrics, dump


def test_saved_ann_model(input_dim: int,hidden_dim:int ,test_feats: torch.Tensor,test_labels: torch.Tensor ,  model_path="best_ann_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ Define the ANN model structure (Must match saved model)
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(hidden_dim),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(hidden_dim, 1)  # Output layer for binary classification
    ).to(device)

    # ✅ Load trained model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode

    # ✅ Convert test features to tensor
    test_feats = test_feats.to(device)

    # ✅ Get predictions
    with torch.no_grad():
        probabilities = torch.sigmoid(model(test_feats)).cpu().numpy()
    
    # Convert probabilities to class labels (binary classification)
    predictions = (probabilities > 0.5).astype(int)

    # Get test labels
    test_labels = test_labels.cpu().numpy()

    # Get evaluation metrics
    eval_metrics = get_eval_metrics(test_labels, predictions, probabilities)
    dump = {"preds_all": predictions, "probs_all": probabilities, "targets_all": test_labels}
    # print confusion matrix
    cm = confusion_matrix(test_labels, predictions)
    print("Confusion Matrix:")
    print(cm)

    return eval_metrics, dump

def plot_training_logs(training_logs):
    plt.figure(figsize=(10, 6))
    plt.plot(training_logs["train_loss"], label="Train Loss", marker="o")
    if "valid_loss" in training_logs and training_logs["valid_loss"]:
        plt.plot(training_logs["valid_loss"], label="Validation Loss", marker="x")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_roc_auc(targets, probs):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(targets, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
