import logging
from typing import Any, List, Tuple, Dict

import numpy as np
import pandas as pd
import sklearn.neighbors
import torch
from torch.nn.functional import normalize
from torch.utils.data import Sampler
from tqdm import tqdm

from .metrics import get_eval_metrics

def eval_knn(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    val_feats: torch.Tensor,
    val_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    n_neighbors: int = 5,
    normalize_feats: bool = True,
    prefix: str = "knn_",
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluate using K-Nearest Neighbors (KNN).

    Args:
        train_feats (torch.Tensor): Training features.
        train_labels (torch.Tensor): Training labels.
        test_feats (torch.Tensor): Test features.
        test_labels (torch.Tensor): Test labels.
        n_neighbors (int): Number of neighbors. Defaults to 5.
        normalize_feats (bool): Whether to normalize features. Defaults to True.
        prefix (str): Prefix for metrics. Defaults to "knn_".

    Returns:
        Tuple containing metrics and evaluation dump.
    """
    if verbose:
        print(f"Train Features Shape: {train_feats.shape}, Train Label Shape: {train_labels.shape}")
        if val_feats is not None:
            print(f"Validation Features Shape: {val_feats.shape}, Validation Label Shape: {val_labels.shape}")
        print(f"Test Features Shape: {test_feats.shape}, Test Label Shape: {test_labels.shape}")
    # Combine train and validation data
    train_feats = torch.cat((train_feats, val_feats), dim=0)
    train_labels = torch.cat((train_labels, val_labels), dim=0)
    # Normalize features
    if normalize_feats:
        train_feats = normalize(train_feats, dim=-1, p=2)
        test_feats = normalize(test_feats, dim=-1, p=2)

    # Convert tensors to numpy for sklearn
    train_feats_np = train_feats.numpy()
    test_feats_np = test_feats.numpy()
    train_labels_np = train_labels.numpy()
    test_labels_np = test_labels.numpy()

    # Train KNN classifier
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(train_feats_np, train_labels_np)

    # Predict and evaluate
    predicted_labels = knn.predict(test_feats_np)
    metrics = get_eval_metrics(test_labels_np, predicted_labels, prefix=prefix)
    dump = {
        "predicted_labels": predicted_labels,
        "targets": test_labels_np,
    }

    return metrics, dump
