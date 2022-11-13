from copy import deepcopy

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from model.base import CLModel
from modules.memory import Buffer
from utils import batch_to_device

'''
    Experience Replay
'''
class ER(CLModel):
    def __init__(self, config):
        super().__init__(config)
        self.buffer = Buffer(buffer_size=config.buffer_size, device='cpu')
        self.replay_size = config.batch_size

    def observe(self, x, y, attn_mask):
        logits, _, loss = self.forward(x, y, attn_mask)
        if not self.buffer.is_empty():
            buffer_x, buffer_y, buffer_attn_mask = self.buffer.get_data(self.replay_size, self.device)
            _, _, aux_loss = self.forward(buffer_x, buffer_y, buffer_attn_mask)
            loss = loss + aux_loss
        self.buffer.add_data(x=x, y=y, attn_mask=attn_mask)
        return logits, loss

    def end_task(self, dataloader, label_offset):
        pass
