import copy
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from modules.backbone import make_backbone
import trainer.utils.train as tutils


def match_logits_loss(s_logits, t_logits, fn, reduction="mean"):
    if fn == "mse":
        return F.mse_loss(s_logits, t_logits, reduction=reduction)
    elif fn == "bce":
        # adapted from icarl
        t_logits = torch.sigmoid(t_logits)
        loss = 0
        for i in range(1, t_logits.shape[-1]):
            loss += F.binary_cross_entropy_with_logits(
                s_logits[..., :i], t_logits[..., :i], reduction=reduction
            )
        return loss
    else:
        raise NotImplementedError


class ExpertModel(nn.Module):
    def __init__(self, config, head_size):
        super(ExpertModel, self).__init__()
        self.config = config
        self.backbone = make_backbone(n_classes=head_size)
        self.old_backbone = None
        self.ewc = None
        self.regularization_coef = config.regularization_coef
        self.regularization_loss_func = "mse"
        self.num_class_learned = 0

        self.moving_average = None
        self.std_average = None

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def expert_train_loss(self, logits, y):
        task_loss = self.task_loss(logits, y)
        reg_loss = 0
        if self.ewc is not None:
            reg_loss = self.ewc.penalty(self.backbone)
        loss = task_loss + self.regularization_coef * reg_loss
        return loss

    def task_loss(self, logits, y):
        logits = logits.clone()[:, self.num_class_learned:]
        return F.cross_entropy(logits, y)

    def forward(self, x, y=None, **ignore_kwargs):
        logits = self.backbone(x)
        loss = None
        if y is not None:
            loss = self.expert_train_loss(logits, y)
        return logits, loss

    def end_task(self, num_classes):
        self.num_class_learned += num_classes

    def surprise_check(self, loss):
        params = [p for p in list(self.parameters()) if p.requires_grad]
        grads = torch.autograd.grad(
            loss, params, retain_graph=True
        )
        l2_norms = torch.stack([(g ** 2).norm() for g in grads])
        if self.moving_average is None:
            self.moving_average = l2_norms
            self.std_average = torch.zeros_like(l2_norms)
        else:
            decay = 0.1
            delta = l2_norms - self.moving_average
            delta = delta.clone()
            _std = self.std_average
            self.moving_average += decay * delta
            self.std_average = (1 - decay) * (
                self.std_average + decay * delta ** 2
            )
            z_scores = delta / (_std) ** 0.5
            # TODO find other context switch
            if z_scores.mean() > 20:
                self.moving_average = None
                self.std_average = None
                return True
        return False

    @torch.no_grad()
    def get_preds(self, x, y, **ignore_kwargs):
        assert not self.training
        logits = self.backbone(x)
        return logits[:, :self.num_class_learned]

    def update(self, ewc):
        self.ewc = ewc

    def checkpoint(self):
        self.old_backbone = copy.deepcopy(self.backbone)

    # def _reg_loss(
    #     self, logits, base_logits, reduction="mean"
    # ):
    #     return match_logits_loss(logits, base_logits, self.regularization_loss_func, reduction=reduction)
