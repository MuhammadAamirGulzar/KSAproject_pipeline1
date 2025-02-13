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
import joblib,os


def eval_protonet(
    fold: int,
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    val_feats: torch.Tensor,
    val_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    normalize_feats: bool = True,
    prefix: str = "proto_",
    model_save_path: str="",
    verbose: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    
    if verbose:
        print(f"Train Features Shape: {train_feats.shape}, Train Label Shape: {train_labels.shape}")
        if val_feats is not None:
            print(f"Validation Features Shape: {val_feats.shape}, Validation Label Shape: {val_labels.shape}")
        print(f"Test Features Shape: {test_feats.shape}, Test Label Shape: {test_labels.shape}")
    # becuase in this model validation not used so combine train and validation data
    if val_feats is not None:
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
    model_path = os.path.join(model_save_path, f"fold{fold}_protonet_model.pkl")
    joblib.dump({"prototypes": prototypes.cpu(), "labels_proto": labels_proto.cpu()}, model_path)
    print(f"✅ Protonet model saved at: {model_path}")
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


def test_saved_protonet_model(test_feats: torch.Tensor, test_labels: torch.Tensor, model_path="protonet_model.pkl"):
    # Load trained class prototypes & labels
    saved_model = joblib.load(model_path)
    prototypes = saved_model["prototypes"]
    labels_proto = saved_model["labels_proto"]

    # Convert prototypes to tensors
    prototypes = torch.tensor(prototypes)
    labels_proto = torch.tensor(labels_proto)

    # Compute pairwise distances
    pairwise_distances = (test_feats[:, None] - prototypes[None, :]).norm(dim=-1, p=2)

    # Predict labels based on the closest prototype
    predicted_labels = labels_proto[pairwise_distances.argmin(dim=1)]

    # Compute evaluation metrics
    eval_metrics = get_eval_metrics(test_labels.numpy(), predicted_labels.numpy(), prefix="proto_")

    # Print confusion matrix
    from sklearn.metrics import confusion_matrix
    conf_matrix = confusion_matrix(test_labels.numpy(), predicted_labels.numpy())
    print("Confusion Matrix:")
    print(conf_matrix)

    return eval_metrics

