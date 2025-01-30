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


def eval_protonet(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    val_feats: torch.Tensor,
    val_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    normalize_feats: bool = True,
    prefix: str = "proto_",
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluate using Prototypical Networks.

    Args:
        train_feats (torch.Tensor): Training features.
        train_labels (torch.Tensor): Training labels.
        test_feats (torch.Tensor): Test features.
        test_labels (torch.Tensor): Test labels.
        normalize_feats (bool): Whether to normalize features. Defaults to True.
        prefix (str): Prefix for metrics. Defaults to "proto_".

    Returns:
        Tuple containing metrics and evaluation dump.
    """
    if verbose:
        print(f"Train Features Shape: {train_feats.shape}, Train Label Shape: {train_labels.shape}")
        if val_feats is not None:
            print(f"Validation Features Shape: {val_feats.shape}, Validation Label Shape: {val_labels.shape}")
        print(f"Test Features Shape: {test_feats.shape}, Test Label Shape: {test_labels.shape}")
    # becuase in this model validation not used so combine train and validation data
    train_feats = torch.cat((train_feats, val_feats), dim=0)
    train_labels = torch.cat((train_labels, val_labels), dim=0)

    # Normalize features
    if normalize_feats:
        train_feats = normalize(train_feats, dim=-1, p=2)
        test_feats = normalize(test_feats, dim=-1, p=2)
    # Compute class prototypes
    class_ids = sorted(np.unique(train_labels))
    prototypes = torch.stack(
        [train_feats[train_labels == class_id].mean(dim=0) for class_id in class_ids]
    )
    labels_proto = torch.tensor(class_ids)

    # Compute pairwise distances
    pairwise_distances = (test_feats[:, None] - prototypes[None, :]).norm(dim=-1, p=2)

    # Predict labels based on closest prototype
    predicted_labels = labels_proto[pairwise_distances.argmin(dim=1)]

    # Evaluate metrics
    metrics = get_eval_metrics(test_labels, predicted_labels, prefix=prefix)
    dump = {
        "predicted_labels": predicted_labels.numpy(),
        "targets": test_labels.numpy(),
        "pairwise_distances": pairwise_distances.cpu().numpy(),
        "prototypes": prototypes.cpu().numpy(),
    }

    return metrics, dump
