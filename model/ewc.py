from copy import deepcopy

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from model.base import CLModel
from utils import batch_to_device


def variable(t: torch.Tensor, device, **kwargs):
    t = t.to(device)
    return Variable(t, **kwargs)


'''
    Elastic Weight Consolidation
'''
class EWC(CLModel):
    def __init__(self, config):
        super().__init__(config)
        self.reg_coef = config.reg_coef
        self.params = None
        self._means = None
        self._precision_matrices = None

    def observe(self, x, y, attn_mask):
        logits, _, loss = self.forward(x, y, attn_mask)
        penalty = self.penalty()
        loss = loss + penalty
        return logits, loss

    def end_task(self, dataloader, label_offset):
        self._update_fisher(dataloader, label_offset)

    def _update_fisher(self, dataloader, label_offset):
        self.params = {n: p for n, p in self.backbone.named_parameters() if p.requires_grad}
        # fisher
        _precision_matrices = self._diag_fisher(dataloader, label_offset)
        if self._means is not None:
            for k, v in self._precision_matrices.items():
                self._precision_matrices[k] = 0.5 * self._precision_matrices[k] + 0.5 * v
        else:
            self._precision_matrices = _precision_matrices

        self._means = {}
        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data, self.device)

    def _diag_fisher(self, dataloader, label_offset):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data, self.device)

        self.backbone.eval()
        for batch in dataloader:
            self.backbone.zero_grad()
            batch = batch_to_device(batch, self.device)
            x, y, attn_mask = batch['input_ids'], batch['labels'], batch['attention_mask']
            y = y.long() + label_offset
            output, _, _ = self.forward(x, y, attn_mask)

            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.backbone.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(dataloader)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        self.backbone.zero_grad()
        return precision_matrices

    def penalty(self):
        if self._means is None:
            return 0
        loss = 0
        for n, p in self.backbone.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss * self.reg_coef