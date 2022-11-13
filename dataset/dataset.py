import math

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from transformers import DistilBertTokenizer
from datasets import load_dataset

from dataset.transform.transforms import make_transform_mnist

glue_task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def create_dataset(configs):
    dataset_name = configs['dataset']
    if dataset_name == 'mnist':
        dataset_args = {
            "root": configs['root'],
            "download": True
        }
        dataset_class = MNIST
        dataset_args["train"] = configs['split'] == 'train'
        dataset_args["transform"] = make_transform_mnist(configs['task_id'])
        ds = dataset_class(**dataset_args)
    elif dataset_name in glue_task_to_keys.keys():
        ds = load_dataset("glue", configs["dataset"])[configs['split']]
    else:
        raise NotImplementedError
    return ds


class TaskStream:
    def __init__(self, config):
        self.train_loader = None
        self.val_loaders = []
        self.config = config
        self.task_id = 0
        # task
        self.task_class = task_class_names[config.dataset]
        self.n_classes = self.task_class.HEAD_SIZE
        self.n_tasks = self.task_class.N_TASKS
        self.seq_len = self.task_class.SEQ_LENGTH
        self.task_names = self.task_class.task_names
        self.label_offset = self.task_class.label_offset

        if self.config.dataset_root_path is None:
            self.config.dataset_root_path = './data'

    def new_task(self):
        # create surprise task
        train_data = self.task_class(root=self.config.dataset_root_path, task_id=self.task_id,
                                     split='train').get_dataset()
        val_data = self.task_class(root=self.config.dataset_root_path, task_id=self.task_id,
                                   split='validation').get_dataset()

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


class SplitGLUE:
    HEAD_SIZE = 17
    N_TASKS = 7
    SEQ_LENGTH = 128
    task_names = ['cola', 'sst2', 'mrpc', 'qqp', 'rte', 'qnli', 'stsb']
    label_offset = [
        0,
        2,
        4,
        6,
        8,
        10,
        12,
        17
    ]

    def __init__(self, root, task_id, split, padding=True):
        self.root = root
        self.split = split
        self.task_id = task_id
        self.task_name = self.task_names[self.task_id]
        # dataset
        task_args = {'dataset': self.task_name,
                     'split': self.split,
                     'root': self.root}
        self.dataset = create_dataset(task_args)
        sentence1_key, sentence2_key = glue_task_to_keys[self.task_name]
        # tokenize
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        max_seq_length = min(self.SEQ_LENGTH, tokenizer.model_max_length)
        if padding:
            padding = "max_length"
        else:
            padding = False

        def preprocess(examples):
            args = (
                (examples[sentence1_key],) if sentence2_key is None else (
                    examples[sentence1_key], examples[sentence2_key])
            )
            result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
            if self.task_name == 'stsb':
                result['label'] = [int(y) if y != 5.0 else 4 for y in examples['label']]
            return result

        tokenized_datasets = self.dataset.map(preprocess, batched=True)
        tokenized_datasets = tokenized_datasets.map()
        tokenized_datasets = tokenized_datasets.remove_columns(
            ['idx'] + [x for x in glue_task_to_keys[self.task_name] if x is not None])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")

        self.dataset = tokenized_datasets

    def get_dataset(self):
        return self.dataset


class SplitMNIST(Dataset):
    HEAD_SIZE = 10
    N_TASKS = 5
    SEQ_LENGTH = 784
    label_offset = [
        0,
        2,
        4,
        6,
        8,
        10
    ]

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


task_class_names = {
    'mnist': SplitMNIST,
    'glue': SplitGLUE,
}
