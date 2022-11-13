from copy import deepcopy

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from model.base import CLModel
from modules.memory import Buffer
from utils import batch_to_device

'''
    Dark Experience Replay ++
'''
class DERPP(CLModel):
    def __init__(self, config):
        super().__init__(config)
        self.buffer = Buffer(buffer_size=config.buffer_size, device='cpu')
        self.replay_size = config.batch_size
        self.alpha = 0.5
        self.beta = 1.0

    def observe(self, x, y, attn_mask):
        logits, _, loss = self.forward(x, y, attn_mask)
        if not self.buffer.is_empty():
            aux_loss = 0
            # knowledge distillation
            buffer_x, buffer_y, buffer_logits, buffer_attn_mask = self.buffer.get_data(self.replay_size, self.device)
            _, _, buffer_loss = self.forward(buffer_x, buffer_y, buffer_attn_mask)
            aux_loss += self.alpha * buffer_loss
            # er
            buffer_x, buffer_y, buffer_logits, buffer_attn_mask = self.buffer.get_data(self.replay_size, self.device)
            s_logits, _, _ = self.forward(buffer_x, buffer_y, buffer_attn_mask)
            aux_loss += self.beta * F.mse_loss(buffer_logits, s_logits)
            loss = loss + aux_loss

        self.buffer.add_data(x=x, y=y, logits=logits, attn_mask=attn_mask)
        return logits, loss

    def end_task(self, dataloader, label_offset):
        pass
