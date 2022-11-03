import torch
import torch.nn.functional as F
from torch import nn

from modules.backbone import make_backbone


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
    def __init__(self, config, seq_len, head_size):
        super(ExpertModel, self).__init__()
        self.config = config
        self.backbone = make_backbone(name=config.backbone, seq_len=seq_len, n_classes=head_size)

        self.ewc = None
        self.regularization_coef = config.regularization_coef
        self.regularization_loss_func = "mse"

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def expert_train_loss(self, logits, y):
        task_loss = F.cross_entropy(logits, y)
        reg_loss = 0
        if self.ewc is not None:
            reg_loss = self.ewc.penalty(self.backbone)
        loss = task_loss + self.regularization_coef * reg_loss
        return loss

    def forward(self, x, y, attn_mask):
        output = self.backbone(input_ids=x, labels=y, attention_mask=attn_mask)
        loss = output.loss
        logits = output.logits
        return logits, loss

    @torch.no_grad()
    def get_preds(self, x, y, attn_mask):
        assert not self.training
        output = self.backbone(input_ids=x, labels=y, attention_mask=attn_mask)
        return output.logits

    def update(self, ewc):
        self.ewc = ewc



