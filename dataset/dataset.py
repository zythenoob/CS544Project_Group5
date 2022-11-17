import math

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from transformers import DistilBertTokenizer
from datasets import load_dataset
import pandas as pd
from datasets import Dataset as dts
import os
from dataset.transform.transforms import make_transform_mnist
from datasets import load_from_disk
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

label_choice = {
    0: ['hep-ph', 'math.CO', 'physics.gen-ph', 'math.CA', 'cond-mat.mes-hall', 'gr-qc', 'cond-mat.mtrl-sci',
        'astro-ph', 'math.NT', 'hep-th', 'math.PR', 'hep-ex', 'nlin.PS', 'math.NA', 'cond-mat.str-el', 'math.RA',
        'physics.optics', 'q-bio.PE', 'q-bio.QM', 'math.OA', 'math.QA', 'cond-mat.stat-mech', 'quant-ph', 'cs.NE',
        'physics.ed-ph', 'math.DG', 'cond-mat.soft', 'physics.pop-ph', 'nucl-th', 'math.FA', 'cs.DS'],
    1: ['math.AG', 'math.DS', 'physics.soc-ph', 'math-ph', 'cond-mat.other', 'physics.data-an', 'cs.CE', 'math.GR',
        'hep-lat', 'cond-mat.supr-con', 'nlin.SI', 'cs.IT', 'math.AC', 'math.SG', 'cs.CC', 'math.GT', 'nlin.CD',
        'math.CV', 'math.AP', 'math.RT', 'q-bio.OT', 'physics.plasm-ph', 'physics.bio-ph', 'nlin.CG', 'cs.DM',
        'nucl-ex', 'physics.flu-dyn', 'physics.comp-ph', 'math.MG', 'physics.atom-ph', 'math.ST'],
    2: ['physics.chem-ph', 'math.AT', 'physics.geo-ph', 'q-bio.NC', 'q-fin.RM', 'cond-mat.dis-nn', 'q-bio.SC',
        'q-bio.BM', 'math.OC', 'cs.CR', 'math.LO', 'cs.NI', 'q-fin.PR', 'physics.class-ph', 'q-fin.GN', 'q-fin.ST',
        'cs.PF', 'stat.ME', 'q-fin.CP', 'math.GM', 'math.KT', 'physics.atm-clus', 'physics.acc-ph', 'math.SP',
        'physics.hist-ph', 'cs.LG', 'cs.CY', 'q-bio.GN', 'cs.CG', 'cs.CV', 'math.HO'],
    3: ['cs.SE', 'physics.ins-det', 'cs.OH', 'cs.PL', 'q-bio.CB', 'cs.AI', 'physics.space-ph', 'nlin.AO',
        'q-bio.MN', 'cs.IR', 'cs.GT', 'cs.LO', 'stat.AP', 'cs.SC', 'cs.DC', 'cs.CL', 'math.CT', 'q-fin.PM',
        'physics.med-ph', 'cs.HC', 'physics.ao-ph', 'cs.AR', 'cs.DL', 'cs.MS', 'cs.RO', 'cs.DB', 'math.GN',
        'q-bio.TO', 'cs.GL', 'cs.MA', 'cs.MM'],
    4: ['stat.ML', 'cs.OS', 'q-fin.TR', 'cs.NA', 'cs.SD', 'stat.CO', 'cs.GR', 'cs.FL', 'cond-mat.quant-gas',
        'astro-ph.HE', 'astro-ph.SR', 'astro-ph.GA', 'astro-ph.CO', 'astro-ph.IM', 'astro-ph.EP', 'cs.SI',
        'stat.OT', 'cs.SY', 'eess.SY', 'cs.ET', 'eess.SP', 'q-fin.EC', 'q-fin.MF', 'physics.app-ph', 'econ.GN',
        'eess.AS', 'econ.TH', 'eess.IV', 'econ.EM', 'cond-mat', 'supr-con'],
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


class SplitArxiv(Dataset):
    HEAD_SIZE = 155
    N_TASKS = 5
    SEQ_LENGTH = 512

    label_offset = [
        0,
        31,
        62,
        93,
        124
    ]

    task_names = ['0', '1', '2', '3', '4']

    def __init__(self, root, task_id, split, padding=True):
        super().__init__()
        self.root = root
        self.split = split
        self.task_id = task_id
        self.path_name = split + str(task_id) + ".hf"
        if(os.path.exists(self.path_name)):
            self.dataset = load_from_disk(self.path_name)
        else:
            df = pd.read_csv("papers_dataset.csv",
                             low_memory=False)
            #         df = csv_data
            train_data = df.sample(frac=0.8, random_state=0, axis=0)
            test_data = df[~df.index.isin(train_data.index)]
            df = train_data if split == "train" else test_data
            self.data = df.loc[df['categories'].isin(label_choice[task_id])]
            self.dataset = dts.from_pandas(self.data)

            tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            max_seq_length = min(self.SEQ_LENGTH, tokenizer.model_max_length)
            if padding:
                padding = "max_length"
            else:
                padding = False

            def preprocess(examples):
                args = ((examples["title"], examples["abstract"]))
                result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
                result['label'] = [label_choice[task_id].index(xx) for xx in examples["categories"]]
                return result

            tokenized_datasets = self.dataset.map(preprocess, batched=True)
            tokenized_datasets = tokenized_datasets.remove_columns(
                ["Unnamed: 0", "id", "title", "abstract", "categories", "task", "__index_level_0__"])
            tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
            tokenized_datasets.set_format("torch")

            self.dataset = tokenized_datasets
            self.dataset.save_to_disk(self.path_name)

    def get_dataset(self):
        return self.dataset


task_class_names = {
    'mnist': SplitMNIST,
    'glue': SplitGLUE,
    'arxiv': SplitArxiv,
}
