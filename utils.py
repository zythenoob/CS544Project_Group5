from typing import Dict

import torch


def batch_to_device(batch, device):
    if isinstance(batch, Dict):
        for k, v in batch.items():
            batch[k] = v.to(device)
    else:
        raise ValueError('Batch type not supported')
    return batch


def mask_logits_class(logits, label_offset, task):
    mask = torch.zeros_like(logits)
    mask[:, label_offset[task]:label_offset[task + 1]] = 1
    return logits * mask
