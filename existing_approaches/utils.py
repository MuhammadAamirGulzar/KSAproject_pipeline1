import torch
import csv
import numpy as np


def encode_onehot(labels, n_classes):
    onehot = torch.FloatTensor(labels.size()[0], n_classes)
    labels = labels.data
    if labels.is_cuda:
        onehot = onehot.cuda()
    onehot.zero_()
    onehot.scatter_(1, labels.view(-1, 1), 1)
    return onehot


class CSVLogger():
    def __init__(self, args, filename='log.csv', fieldnames=['epoch']):

        self.filename = filename
        self.csv_file = open(filename, 'w')

        # Write model configuration at top of csv
        writer = csv.writer(self.csv_file)
        for arg in vars(args):
            writer.writerow([arg, getattr(args, arg)])
        writer.writerow([''])

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()


class Cutout(object):
    """Randomly mask out one or more patches from an image.
       https://arxiv.org/abs/1708.04552
    Args:
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        if np.random.choice([0, 1]):
            mask = np.ones((h, w), np.float32)

            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = int(np.clip(y - self.length / 2, 0, h))
            y2 = int(np.clip(y + self.length / 2, 0, h))
            x1 = int(np.clip(x - self.length / 2, 0, w))
            x2 = int(np.clip(x + self.length / 2, 0, w))

            mask[y1: y2, x1: x2] = 0.

            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img = img * mask

        return img


import numpy as np
from sklearn.utils import column_or_1d, check_array
from sklearn.utils.multiclass import type_of_target, unique_labels
from sklearn.utils.validation import _num_samples
import scipy.sparse as sp

def custom_label_binarize(y, *, classes, neg_label=0, pos_label=1, sparse_output=False):
    if not isinstance(y, list):
        y = check_array(
            y, input_name="y", accept_sparse="csr", ensure_2d=False, dtype=None
        )
    else:
        if _num_samples(y) == 0:
            raise ValueError("y has 0 samples: %r" % y)

    if neg_label >= pos_label:
        raise ValueError(
            "neg_label={0} must be strictly less than pos_label={1}.".format(
                neg_label, pos_label
            )
        )

    if sparse_output and (pos_label == 0 or neg_label != 0):
        raise ValueError(
            "Sparse binarization is only supported with non "
            "zero pos_label and zero neg_label, got "
            "pos_label={0} and neg_label={1}".format(pos_label, neg_label)
        )

    pos_switch = pos_label == 0
    if pos_switch:
        pos_label = -neg_label

    y_type = type_of_target(y)
    if "multioutput" in y_type:
        raise ValueError(
            "Multioutput target data is not supported with label binarization"
        )
    if y_type == "unknown":
        raise ValueError("The type of target data is not known")

    n_samples = y.shape[0] if sp.issparse(y) else len(y)
    n_classes = len(classes)
    classes = np.asarray(classes)

    if y_type == "binary":
        if n_classes == 1:
            if sparse_output:
                return sp.csr_matrix((n_samples, 1), dtype=int)
            else:
                Y = np.zeros((n_samples, 2), dtype=int)  # Ensure 2 columns for binary
                Y[:, 0] = neg_label
                Y[:, 1] = pos_label
                return Y
        elif len(classes) >= 3:
            y_type = "multiclass"

    sorted_class = np.sort(classes)
    if y_type == "multilabel-indicator":
        y_n_classes = y.shape[1] if hasattr(y, "shape") else len(y[0])
        if classes.size != y_n_classes:
            raise ValueError(
                "classes {0} mismatch with the labels {1} found in the data".format(
                    classes, unique_labels(y)
                )
            )

    if y_type in ("binary", "multiclass"):
        y = column_or_1d(y)

        y_in_classes = np.isin(y, classes)
        y_seen = y[y_in_classes]
        indices = np.searchsorted(sorted_class, y_seen)
        indptr = np.hstack((0, np.cumsum(y_in_classes)))

        data = np.empty_like(indices)
        data.fill(pos_label)
        Y = sp.csr_matrix((data, indices, indptr), shape=(n_samples, n_classes))
    elif y_type == "multilabel-indicator":
        Y = sp.csr_matrix(y)
        if pos_label != 1:
            data = np.empty_like(Y.data)
            data.fill(pos_label)
            Y.data = data
    else:
        raise ValueError(
            "%s target data is not supported with label binarization" % y_type
        )

    if not sparse_output:
        Y = Y.toarray()
        Y = Y.astype(int, copy=False)

        if neg_label != 0:
            Y[Y == 0] = neg_label

        if pos_switch:
            Y[Y == pos_label] = 0
    else:
        Y.data = Y.data.astype(int, copy=False)

    if np.any(classes != sorted_class):
        indices = np.searchsorted(sorted_class, classes)
        Y = Y[:, indices]

    if y_type == "binary":
        if sparse_output:
            Y = Y.getcol(-1)
        else:
            Y = np.zeros((n_samples, 2), dtype=int)  # Ensure 2 columns for binary
            for i in range(n_samples):
                if y[i] == classes[0]:
                    Y[i, 0] = pos_label
                    Y[i, 1] = neg_label
                else:
                    Y[i, 0] = neg_label
                    Y[i, 1] = pos_label

    return Y
