import torch
import torch.nn as nn
import torch.nn.functional as F

def NLL(y_hat, y, sigma=1.0):
    """
    Negative log likelihood of Gaussian distribution
    """
    return torch.mean((y - y_hat) ** 2) / (2 * sigma**2)

def Cross_entropy(y_hat, y):
    criterion = nn.BCEWithLogitsLoss()
    return criterion(y_hat, y)
