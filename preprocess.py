import torch

def preprocess(x):
    mean, std, y = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = (x[:, i, :, :] - mean[i]) / std[i]
    return y

def undo_preprocess(x):
    mean, std, y = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    return y