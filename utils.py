import os, json, random, pickle, copy
import time

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Random seed for torch
def seed_torch(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# Model inference
def get_preds(test_loader, model, mode='test'):
    """
    Inference for test set.

    :param test_loader: DataLoader for test data
    :param model: model

    :return: ndarray predictions
    """
    assert mode in ('val', 'test')
    model.eval()  # eval mode disables dropout

    predictions = []
    with torch.no_grad():
        # Batches
        for i, images in enumerate(test_loader):
            if mode=='val': images = images[0]

            # Move to default device
            images = images.to(device)  # (N, 3, 224, 224)

            # Forward prop.
            pred_scores = model(images)  # (N, 1)

            predictions.append(torch.sigmoid(pred_scores[:,0]).cpu().numpy())

    predictions = np.concatenate(predictions)

    return predictions


def TTA(test_datasets, model, beta=0.4):
    """
    Test time augmentation prediction

    Inputs:
    - test_datasets: list of original/transformed datasets
    - model: model
    - beta: param for ratio of pred of original dataset in final results

    Return:
    - ndarray predictions
    """
    original_pred = 0
    transformed_pred = 0
    for i in range(len(test_datasets)):
        test_loader = DataLoader(test_datasets[i], batch_size=batch_size,
                                 shuffle=False)
        preds = get_preds(test_loader, model)
        if i == 0:
            original_pred += preds
        else:
            transformed_pred += preds / (len(test_datasets)-1)

    predictions = beta * original_pred + (1-beta) * transformed_pred

    return predictions


# Forward model to get stats of batch norm (for weight avg model)
def forward_model(model, train_loader):
    model.train()
    with torch.no_grad():
        for x,y in train_loader:
            x = x.to(device=device, dtype=torch.float32)
            tmp = model(x)
    torch.cuda.empty_cache()


# Gradient clipping
def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


# For metrics log
class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Plot log of loss and auc during training
def plot_history(history, fname):
    aucs, val_aucs, losses, val_losses = history
    epochs = range(len(aucs))
    fig = plt.figure(figsize=(14,5))
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(epochs, aucs, '-o')
    ax1.plot(epochs, val_aucs, '-o')
    #ax1.set_ylim(0.8, 1.0)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Auc')
    ax1.legend(['train', 'val'], loc='lower right')
    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(epochs, losses, '-o')
    ax2.plot(epochs, val_losses, '-o')
    #ax2.set_ylim(bottom=-0.1)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(['train', 'val'], loc='upper right')
    fig.savefig(fname)

