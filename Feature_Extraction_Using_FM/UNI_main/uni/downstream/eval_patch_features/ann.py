import torch
import torch.nn.functional as F
import numpy as np
import random
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

class ANNBinaryClassifier:
    def __init__(self, input_dim=512, hidden_dim=512, C=1.0, max_iter=100, verbose=True, random_state=42):
        self.C = C
        self.loss_func = FocalLoss()  # Use Focal Loss for class imbalance
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.model = None

    def compute_loss(self, preds, labels):
        loss = self.loss_func(preds, labels)
        wreg = 0.5 * sum((param.norm(p=2) for param in self.model.parameters()))  # L2 regularization for weights
        return loss.mean() + (1.0 / self.C) * wreg

    def predict_proba(self, feats):
        assert self.model is not None, "Model must be trained before making predictions."
        feats = feats.to(self.device)
        with torch.no_grad():
            return torch.sigmoid(self.model(feats))

    def fit(self, feats, labels, val_feats=None, val_labels=None):
        # Set random seed
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Define the ANN model architecture
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.hidden_dim),  # Input to first hidden layer
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(self.hidden_dim),
            torch.nn.Dropout(0.5),  # Increased dropout rate for better regularization
            torch.nn.Linear(self.hidden_dim, self.hidden_dim * 2),  # Second hidden layer
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(self.hidden_dim * 2),
            torch.nn.Dropout(0.5),  # Increased dropout rate for better regularization
            torch.nn.Linear(self.hidden_dim * 2, 1)  # Output layer (binary classification)
        ).to(self.device)

        feats = feats.to(self.device)
        labels = labels.to(self.device)

        # Define optimizer
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)

        # Training loop
        for epoch in range(self.max_iter):
            def loss_closure():
                opt.zero_grad()
                preds = self.model(feats).squeeze(-1)
                loss = self.compute_loss(preds, labels.float())
                loss.backward()
                return loss

            opt.step(loss_closure)
            scheduler.step()  # Step the learning rate scheduler

            if self.verbose and epoch % 10 == 0:  # Log loss every 10 epochs
                preds = self.model(feats).squeeze(-1)
                loss = self.compute_loss(preds, labels.float())
                # print(f"Epoch {epoch}: Loss: {loss:.3f}")

                # Validation loss if provided
                if val_feats is not None and val_labels is not None:
                    val_feats = val_feats.to(self.device)
                    val_labels = val_labels.to(self.device)
                    val_preds = self.model(val_feats).squeeze(-1)
                    val_loss = self.compute_loss(val_preds, val_labels.float())
                    # print(f"Epoch {epoch} and Validation Loss: {val_loss:.3f}")

        if self.verbose:
            preds = self.model(feats).squeeze(-1)
            loss = self.compute_loss(preds, labels.float())
            print(f"(After Training) Loss: {loss:.3f}")


def eval_ANN_probe(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    valid_feats: torch.Tensor,
    valid_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    input_dim: int = 512,
    max_iter: int = 1000,
    combine_trainval: bool = True,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if verbose:
        print("ANN Probe Evaluation: Train shape", train_feats.shape)
    if valid_feats is not None:
        if verbose:
            print("ANN Probe Evaluation: Valid shape", valid_feats.shape)
    if verbose:
        print("ANN Probe Evaluation: Test shape", test_feats.shape)
    start = time.time()

    classifier = train_ANN_probe(
        train_feats, train_labels, valid_feats, valid_labels,input_dim, max_iter, combine_trainval, verbose
    )

    results, dump = test_ANN_probe(classifier, test_feats, test_labels, prefix="ann_", verbose=verbose)
    classifier.model = classifier.model.to(torch.device("cpu"))
    dump["model"] = classifier.model.state_dict()
    del classifier
    torch.cuda.empty_cache()
    
    if verbose:
        print(f"ANN Probe Evaluation: Time taken {time.time() - start:.2f}")
    
    return results, dump

def train_ANN_probe(
    train_feats,
    train_labels,
    valid_feats,
    valid_labels,
    input_dim,
    max_iter=1000,
    combine_trainval=True,
    verbose=True,
):
    NUM_C = len(set(train_labels.cpu().numpy()))
    cost = (train_feats.shape[1] * NUM_C) / 100
    if verbose:
        print(f"ANN Probe Evaluation (Train Time): Best cost = {cost:.3f}")

    # Combine train and validation sets for final training
    if combine_trainval and valid_feats is not None:
        trainval_feats = torch.cat([train_feats, valid_feats], dim=0)
        trainval_labels = torch.cat([train_labels, valid_labels], dim=0)
        if verbose:
            print("ANN Probe Evaluation (Train Time): Combining train and validation sets for final training. Trainval Shape: ", trainval_feats.shape)
        classifier = ANNBinaryClassifier(input_dim=input_dim,max_iter=max_iter, verbose=verbose)
        classifier.fit(trainval_feats, trainval_labels)
    else:
        if verbose:
            print("ANN Probe Evaluation (Train Time): Using only train set for evaluation. Train Shape: ", train_feats.shape)
        classifier = ANNBinaryClassifier(input_dim=input_dim,max_iter=max_iter, verbose=verbose)
        classifier.fit(train_feats, train_labels)
    
    return classifier

def test_ANN_probe(
    classifier: ANNBinaryClassifier,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    prefix: str = "ann_",
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if verbose:
        print(f"ANN Probe Evaluation (Test Time): Test Shape {test_feats.shape}")

    probs_all = classifier.predict_proba(test_feats).squeeze(-1).cpu().numpy()
    preds_all = (probs_all > 0.5).astype(int)
    targets_all = test_labels.cpu().numpy()

    eval_metrics = get_eval_metrics(targets_all, preds_all, probs_all, True, prefix)
    dump = {"preds_all": preds_all, "probs_all": probs_all, "targets_all": targets_all}

    conf_matrix = confusion_matrix(targets_all, preds_all)
    print("Confusion Matrix:")
    print(conf_matrix)

    return eval_metrics, dump

