from abc import abstractmethod
import torch
from torch import nn

from modules.backbone import make_backbone


class CLModel(nn.Module):
    def __init__(self, config):
        super(CLModel, self).__init__()
        self.config = config
        self.backbone = make_backbone(name=config.backbone, seq_len=config.seq_len, n_classes=config.head_size)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @abstractmethod
    def observe(self, x, y, attn_mask):
        pass

    @abstractmethod
    def end_task(self, dl, offset):
        pass

    def forward(self, x, y, attn_mask):
        output = self.backbone(input_ids=x, labels=y, attention_mask=attn_mask, output_hidden_states=True)
        loss = output.loss
        logits = output.logits
        features = output.hidden_states
        return logits, features, loss

    @torch.no_grad()
    def get_preds(self, x, y, attn_mask):
        assert not self.training
        logits, _, _ = self.forward(x=x, y=y, attn_mask=attn_mask)
        return logits


def get_model(config):
    name = config.method
    if name == 'ewc':
        from model.ewc import EWC
        model_class = EWC
    elif name == 'er':
        from model.er import ER
        model_class = ER
    elif name == 'derpp':
        from model.derpp import DERPP
        model_class = DERPP
    else:
        raise NotImplementedError
    return model_class(config)
