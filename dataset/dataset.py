import random

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST, SVHN

from dataset.transform.transforms import make_transform_mnist

SURPRISE = ['raw', 'rot', 'perm', 'svhn']
# SURPRISE = ['svhn']


def create_dataset(configs):
    dataset_name = configs['dataset']
    dataset_args = {
        "root": configs['root'],
        "download": True
    }
    if dataset_name == 'mnist':
        dataset_class = MNIST
        transform = make_transform_mnist(configs['task_id'])
        dataset_args["train"] = configs['split'] == 'train'
    else:
        raise NotImplementedError
    dataset_args["transform"] = transform
    return dataset_class(**dataset_args)


class BufferDataset(Dataset):
    def __init__(self, x, y, logits):
        self.x = x
        self.y = y
        self.logits = logits

    def __getitem__(self, idx):
        return {
            'x': self.x[idx],
            'y': self.y[idx],
            'logits': self.logits[idx]
        }

    def __len__(self):
        return len(self.x)


class TaskStream:
    def __init__(self, config):
        self.train_loader = None
        self.val_loaders = []
        self.config = config
        self.task_id = 0
        # task
        self.task_class = SplitMNIST
        self.task_type = config.task_type
        assert self.task_type in ["til", "cil"]
        self.n_classes = self.task_class.HEAD_SIZE * self.task_class.N_TASKS
        self.n_tasks = self.task_class.N_TASKS
        self.seq_len = self.task_class.SEQ_LENGTH

        if self.config.dataset_root_path is None:
            self.config.dataset_root_path = './data'

    def new_task(self):
        # create surprise task
        train_data = self.task_class(root=self.config.dataset_root_path, task_id=self.task_id, split='train')
        val_data = self.task_class(root=self.config.dataset_root_path, task_id=self.task_id, split='test')

        self.train_loader = DataLoader(
            train_data,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=None,
        )
        val_loader = DataLoader(
            val_data,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=None,
        )
        self.val_loaders.append(val_loader)

        self.task_id += 1

    def __len__(self):
        return len(self.val_loaders)


class SplitMNIST(Dataset):
    HEAD_SIZE = 2
    N_TASKS = 5
    SEQ_LENGTH = 784

    def __init__(self, root, task_id, split):
        super().__init__()
        self.root = root
        self.split = split
        self.task_id = task_id
        # dataset
        task_args = {'dataset': 'mnist',
                     'task_id': task_id,
                     'split': self.split,
                     'root': self.root}
        self.dataset = create_dataset(task_args)
        # split dataset: use data within certain classes for split task
        train_mask = np.logical_and(np.array(self.dataset.targets) >= task_id * self.HEAD_SIZE,
                                    np.array(self.dataset.targets) < (task_id + 1) * self.HEAD_SIZE)

        self.dataset.data = self.dataset.data[train_mask]
        self.dataset.targets = np.array(self.dataset.targets)[train_mask]

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return {'x': x,
                'y': y,
                'task_id': self.task_id}

    def __len__(self):
        return len(self.dataset)
