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
    0: ['econ.GN', 'econ.EM', 'econ.TH'],
    1: ['math.GT', 'math.KT', 'math.AC', 'math.AP', 'math.CT', 'math.QA', 'math.PR', 'math.FA', 'math.OC', 'math.NT', 'math.RA', 'math.AT', 'math.DS', 'math.SG', 'math.MG', 'math.SP', 'math.GN', 'math.CA', 'math.CO', 'math.RT', 'math.HO', 'math.AG', 'math.OA', 'math.ST', 'math.GM', 'math.DG', 'math.LO', 'math.CV', 'math.GR', 'math.NA'],
    2: ['q-bio.PE', 'q-bio.SC', 'q-bio.QM', 'q-bio.TO', 'q-bio.GN', 'q-bio.NC', 'q-bio.OT', 'q-bio.MN', 'q-bio.BM', 'q-bio.CB'],
    3: ['cs.CR', 'cs.DS', 'cs.NI', 'cs.HC', 'cs.MA', 'cs.DM', 'cs.RO', 'cs.PL', 'cs.AI', 'cs.CV', 'cs.SE', 'cs.SY', 'cs.ET', 'cs.CG', 'cs.CY', 'cs.CE', 'cs.CL', 'cs.LO', 'cs.GR', 'cs.AR', 'cs.PF', 'cs.GL', 'cs.OS', 'cs.OH', 'cs.IT', 'cs.DL', 'cs.SD', 'cs.GT', 'cs.LG', 'cs.NE', 'cs.DB', 'cs.MM', 'cs.SC', 'cs.SI', 'cs.MS', 'cs.NA', 'cs.FL', 'cs.IR', 'cs.CC', 'cs.DC'],
    4: ['eess.SY', 'eess.SP', 'eess.IV', 'eess.AS'],
    5: ['cond-mat.supr-con', 'physics.atom-ph', 'cond-mat.quant-gas', 'physics.atm-clus', 'physics.plasm-ph', 'physics.space-ph', 'physics.pop-ph', 'physics.chem-ph', 'astro-ph.HE', 'astro-ph.IM', 'astro-ph.EP', 'nucl-ex', 'cond-mat.soft', 'cond-mat.other', 'nlin.AO', 'astro-ph.GA', 'hep-ex', 'physics.soc-ph', 'supr-con', 'quant-ph', 'physics.data-an', 'cond-mat', 'math-ph', 'hep-ph', 'nucl-th', 'physics.hist-ph', 'cond-mat.mtrl-sci', 'astro-ph', 'physics.ao-ph', 'physics.ed-ph', 'cond-mat.stat-mech', 'physics.class-ph', 'hep-th', 'hep-lat', 'physics.bio-ph', 'physics.flu-dyn', 'astro-ph.SR', 'physics.app-ph', 'physics.ins-det', 'astro-ph.CO', 'nlin.PS', 'nlin.CD', 'physics.gen-ph', 'physics.optics', 'cond-mat.dis-nn', 'nlin.SI', 'gr-qc', 'nlin.CG', 'physics.comp-ph', 'physics.geo-ph', 'physics.acc-ph', 'physics.med-ph', 'cond-mat.str-el', 'cond-mat.mes-hall'],
    6: ['q-fin.TR', 'q-fin.EC', 'q-fin.ST', 'q-fin.CP', 'q-fin.MF', 'q-fin.RM', 'q-fin.GN', 'q-fin.PM', 'q-fin.PR'],
    7: ['stat.ML', 'stat.ME', 'stat.OT', 'stat.CO', 'stat.AP'],
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
                result['label'] = [int(y) if y != 5.0 else 4 for y in examples['categories']]
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
    N_TASKS = 8
    SEQ_LENGTH = 512

    label_offset = [
        0,
        3,
        33,
        43,
        83,
        87,
        141,
        150,
        155
    ]
    task_names = ['0', '1', '2', '3', '4', '5', '6', '7']

    def __init__(self, root, task_id, split, padding=True):
        super().__init__()
        self.root = root
        self.split = split
        self.task_id = task_id
        self.path_name = split + str(task_id) + ".hf"
        # if(os.path.exists(self.path_name)):
        #     self.dataset = load_from_disk(self.path_name)
        if 0:
            pass
        else:
            df = pd.read_csv("papers_dataset.csv",
                             low_memory=False)
            #         df = csv_data
            train_data = df.sample(frac=0.8, random_state=0, axis=0)
            test_data = df[~df.index.isin(train_data.index)]
            df = train_data.sample(frac=0.1, random_state=0, axis=0) if split == "train" else test_data.sample(frac=0.1, random_state=0, axis=0)
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
                print([label_choice[task_id].index(xx) for xx in examples["categories"]])
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
