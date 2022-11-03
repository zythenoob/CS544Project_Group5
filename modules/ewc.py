from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from utils import batch_to_device


def variable(t: torch.Tensor, device, **kwargs):
    t = t.to(device)
    return Variable(t, **kwargs)


class EWC(object):
    def __init__(self, device):
        self.device = device

    def update(self, model, dataloader, label_offset):
        model = deepcopy(model)
        self.params = {n: p for n, p in model.backbone.named_parameters() if p.requires_grad}
        self._means = {}
        # fisher
        _precision_matrices = self._diag_fisher(model, dataloader, self.device, label_offset)
        if hasattr(self, '_precision_matrices'):
            for k, v in self._precision_matrices.items():
                self._precision_matrices[k] = 0.5 * self._precision_matrices[k] + 0.5 * v
        else:
            self._precision_matrices = _precision_matrices

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data, self.device)
        # release memory
        del model

    def _diag_fisher(self, model, dataloader, device, label_offset):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data, device)

        model.eval()
        for batch in dataloader:
            model.zero_grad()
            batch = batch_to_device(batch, device)
            x, y, attn_mask = batch['input_ids'], batch['labels'], batch['attention_mask']
            y = y + label_offset
            output, _ = model(x, y, attn_mask=attn_mask)

            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in model.backbone.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(dataloader)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, backbone: nn.Module):
        if not hasattr(self, '_precision_matrices'):
            return 0
        loss = 0
        for n, p in backbone.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss
