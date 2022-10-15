import random
import datetime
import os
import string

import torch
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
)
from scipy.interpolate import interp1d
from scipy.optimize import brentq


class Struct:
    """
    Struct class, s.t. a nested dictionary is transformed
    into a nested object
    """

    def __init__(self, **entries):
        self.entries = entries
        for k, v in entries.items():
            if isinstance(v, dict):
                self.__dict__.update({k: Struct(**v)})
            else:
                self.__dict__.update({k: v})

    def get_true_key(self):
        """
        Return the only key in the Struct s.t. its value is True
        """
        true_types = [k for k, v in self.__dict__.items() if v == True]
        assert len(true_types) == 1
        return true_types[0]

    def get_true_keys(self):
        """
        Return all the keys in the Struct s.t. its value is True
        """
        return [k for k, v in self.__dict__.items() if v == True]

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)


def get_random_filename(length=10):
    """
    Return a random sequence of letters, to be used as unique filenames
    """
    symbols = string.ascii_lowercase
    return "".join(random.choice(symbols) for _ in range(length))


def plot_spectrogram(spectrogram, figsize=(12, 3)):
    """
    Plot the given spectrogram as an image having frequency
    on the y-axis and time on the x-axis
    """
    # If a batch of spectrograms is given, select a random one
    if len(spectrogram.shape) > 2:
        if spectrogram.size(0) > 1:
            spectrogram = spectrogram[random.randint(0, len(spectrogram) - 1)]
        else:
            spectrogram = spectrogram.squeeze(0)

    # Convert from torch to numpy
    spectrogram = to_numpy(spectrogram.squeeze(0)).astype(np.float32)

    # Plot the spectrogram
    _, ax = plt.subplots(figsize=figsize)
    img = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(img, ax=ax)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def play_audio(waveform, sample_rate):
    """
    Spawn an audio player in a Jupyter notebook,
    to listen to the given waveform
    """
    waveform = to_numpy(waveform)
    num_channels, _ = waveform.shape
    if num_channels == 1:
        ipd.display(ipd.Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
        ipd.display(ipd.Audio((waveform[0], waveform[1]), rate=sample_rate))
    else:
        raise ValueError("Waveform with more than 2 channels are not supported")


def to_numpy(arr):
    """
    Convert the given array to the numpy format
    """
    if isinstance(arr, np.ndarray):
        return arr
    elif isinstance(arr, list):
        return np.array(arr)
    elif isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return None


def now():
    """
    Returns the current date and time
    """
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_device():
    """
    Return a CUDA device, if available, or a standard CPU device otherwise
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def flatten(arr):
    """
    Flatten the given 2D array
    """
    return [item for sublist in arr for item in sublist]


def overlap(a1, a2):
    """
    Check if the given arrays have common elements
    """
    return len(set(a1).intersection(set(a2))) > 0


def set_seed(seed):
    """
    Fix all possible sources of randomness
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def compute_eer(scores, labels):
    """
    Compute the equal error rate score
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return eer


def compute_error_rates(scores, labels, eps=1e-6):
    """
    Creates a list of false negative rates, a list of false positive rates
    and a list of decision thresholds that give those error rates
    (see https://github.com/clovaai/voxceleb_trainer)
    """
    # Sort the scores from smallest to largest, and also get the corresponding
    # indexes of the sorted scores.  We will treat the sorted scores as the
    # thresholds at which the the error-rates are evaluated.
    sorted_indexes, _ = zip(
        *sorted(
            [(index, threshold) for index, threshold in enumerate(scores)],
            key=lambda t: t[1],
        )
    )
    labels = [labels[i] for i in sorted_indexes]

    # At the end of this loop, fnrs[i] is the number of errors made by
    # incorrectly rejecting scores less than thresholds[i] and fprs[i]
    # is the total number of times that we have correctly accepted
    # scores greater than thresholds[i]
    fnrs, fprs = [], []
    for i in range(0, len(labels)):
        if i == 0:
            fnrs.append(labels[i])
            fprs.append(1 - labels[i])
        else:
            fnrs.append(fnrs[i - 1] + labels[i])
            fprs.append(fprs[i - 1] + 1 - labels[i])
    fnrs_norm = sum(labels)
    fprs_norm = len(labels) - fnrs_norm

    # Now divide by the total number of false negative errors to
    # obtain the false positive rates across all thresholds
    fnrs = [x / (float(fnrs_norm) + eps) for x in fnrs]

    # Divide by the total number of correct positives to get the
    # true positive rate and subtract these quantities from 1 to
    # get the false positive rates
    fprs = [1 - x / (float(fprs_norm) + eps) for x in fprs]

    return fnrs, fprs


def compute_mindcf(scores, labels, p_target=1e-2, c_fa=1, c_miss=1, eps=1e-6):
    """
    Computes the minimum of the detection cost function
    (see https://github.com/clovaai/voxceleb_trainer)
    """
    # Extract false negative and false positive rates
    fnrs, fprs = compute_error_rates(scores, labels)

    # Compute the minimum detection cost
    min_c_det = float("inf")
    for i in range(0, len(fnrs)):
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det

    # Compute default cost and use it to normalize the
    # minimum detection cost
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / (c_def + eps)

    return min_dcf


def get_train_val_metrics(y_true, y_pred, prefix=None):
    """
    Return a dictionary of classification metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall": recall_score(y_true, y_pred, average="macro"),
        "f1": f1_score(y_true, y_pred, average="macro"),
    }
    if prefix is not None:
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
    return metrics


def get_test_metrics(
    scores, labels, mindcf_p_target=1e-2, mindcf_c_fa=1, mindcf_c_miss=1, prefix=None
):
    """
    Return EER and minDCF metrics
    """
    metrics = {
        "eer": compute_eer(scores, labels),
        "mindcf": compute_mindcf(
            scores,
            labels,
            p_target=mindcf_p_target,
            c_fa=mindcf_c_fa,
            c_miss=mindcf_c_miss,
        ),
    }
    if prefix is not None:
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
    return metrics


def optimizer_to(optimizer, device="cpu"):
    """
    Transfer the given optimizer to device
    """
    for param in optimizer.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
    return optimizer


def scheduler_to(scheduler, device="cpu"):
    """
    Transfer the given LR scheduler to device
    """
    for param in scheduler.__dict__.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
    return scheduler


def chart_dependencies(model, n_mels=80, device="cpu"):
    """
    Use backprop to chart dependencies
    (see http://karpathy.github.io/2019/04/25/recipe/)
    """
    model.eval()
    batch_size, time_steps = random.randint(2, 10), random.randint(10, 100)
    inputs = torch.randn((batch_size, n_mels, time_steps)).to(device)
    inputs.requires_grad = True
    outputs = model(inputs)
    random_index = random.randint(0, batch_size)
    loss = outputs[random_index].sum()
    loss.backward()
    assert (
        torch.cat([inputs.grad[i] == 0 for i in range(batch_size) if i != random_index])
    ).all() and (
        inputs.grad[random_index] != 0
    ).any(), f"Only index {random_index} should have non-zero gradients"
