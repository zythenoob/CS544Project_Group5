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

    def forward(self, x=None, attn_mask=None, inputs_embeds=None):
        if inputs_embeds is not None:
            output = self.backbone(inputs_embeds=inputs_embeds, attention_mask=attn_mask, output_hidden_states=True)
        else:
            output = self.backbone(input_ids=x, attention_mask=attn_mask, output_hidden_states=True)
        logits = output.logits
        features = output.hidden_states
        return logits, features

    @torch.no_grad()
    def get_preds(self, x, attn_mask):
        assert not self.training
        logits, _ = self.forward(x=x, attn_mask=attn_mask)
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
    elif name == 'ersyn':
        from model.ersyn import ERSyn
        model_class = ERSyn
    else:
        raise NotImplementedError
    return model_class(config)
