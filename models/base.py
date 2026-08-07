"""Minimal base learner: device bookkeeping and evaluation helpers."""

import numpy as np
import torch
from torch import nn

from utils.toolkit import accuracy


class BaseLearner(object):
    def __init__(self, args):
        self.args = args
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None

        self.topk = args["top_k"]
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        return self._network.feature_dim

    def incremental_train(self, data_manager):
        raise NotImplementedError

    def after_task(self):
        pass

    def eval_task(self):
        """Evaluate on all classes seen so far.

        Returns a dict with ``top1``, ``top{topk}`` and per-class-group accuracy.
        """
        y_pred, y_true = self._eval_cnn(self.test_loader)
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes, self.args["increment"])
        return {
            "top1": grouped["total"],
            "top{}".format(self.topk): np.around(
                (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true), decimals=2
            ),
            "grouped": grouped,
        }

    @torch.no_grad()
    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for _, inputs, targets in loader:
            outputs = model(inputs.to(self._device))["logits"]
            correct += int((outputs.argmax(dim=1).cpu() == targets).sum())
            total += len(targets)
        return np.around(correct * 100 / max(1, total), decimals=2)

    @torch.no_grad()
    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, inputs, targets in loader:
            outputs = self._network(inputs.to(self._device))["logits"]
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk], [N]
