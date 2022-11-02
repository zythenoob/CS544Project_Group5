import numpy as np
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset


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
        self.task_class = SplitGLUE
        self.n_classes = sum(self.task_class.dataset_class)
        self.n_tasks = self.task_class.N_TASKS
        self.seq_len = self.task_class.SEQ_LENGTH

        if self.config.dataset_root_path is None:
            self.config.dataset_root_path = './data'

    def new_task(self):
        # create surprise task
        train_data = self.task_class(root=self.config.dataset_root_path, task_id=self.task_id, split='train').get_dataset()
        val_data = self.task_class(root=self.config.dataset_root_path, task_id=self.task_id, split='validation').get_dataset()

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


task_to_keys = {
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

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def create_dataset(configs):
    return load_dataset("glue", configs["dataset"])[configs['split']]


class SplitGLUE:
    HEAD_SIZE = 2
    N_TASKS = 10
    SEQ_LENGTH = 128
    dataset_list = ['cola', 'sst2', 'mrpc', 'qqp', 'rte']
    dataset_class = [2, 2, 2, 2, 2]
    dataset_class_begin = {'cola': 0, 'sst2': 2, 'mrpc': 4, 'qqp': 6, 'rte': 8}

    def __init__(self, root, task_id, split, padding=True):
        self.root = root
        self.split = split
        self.task_id = task_id
        self.task_name = self.dataset_list[self.task_id]
        # dataset
        task_args = {'dataset': self.task_name,
                     'split': self.split,
                     'root': self.root}
        self.dataset = create_dataset(task_args)
        sentence1_key, sentence2_key = task_to_keys[self.task_name]
        max_seq_length = min(128, tokenizer.model_max_length)

        if padding:
            padding = "max_length"
        else:
            padding = False

        def preprocess_function(examples):
            args = (
                (examples[sentence1_key],) if sentence2_key is None else (
                examples[sentence1_key], examples[sentence2_key])
            )
            result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

            return result

        tokenized_datasets = self.dataset.map(preprocess_function, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(
            ['idx'] + [x for x in task_to_keys[self.task_name] if x != None])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")

        small_train_dataset = tokenized_datasets.shuffle(seed=69)
        self.datasets = small_train_dataset

    def get_dataset(self):
        return self.datasets