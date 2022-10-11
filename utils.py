from typing import Dict


def batch_to_device(batch, device):
    if isinstance(batch, Dict):
        for k, v in batch.items():
            batch[k] = v.to(device)
    else:
        raise ValueError('Batch type not supported')
    return batch
