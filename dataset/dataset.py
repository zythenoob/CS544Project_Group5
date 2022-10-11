import random

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST, SVHN

from dataset.transform.transforms import make_transform_mnist, make_transform_svhn, make_transform
from dataset.utils import get_random_seed


SURPRISE = ['rot', 'perm', 'svhn_perm', 'svhn_rot']

SURPRISE_MAP = {
    "rot->perm": "abrupt",                  # from mnist rotation to mnist permutation
    "perm->rot": "distribution",            # add rotation to mnist
    "svhn_rot->svhn_perm": "abrupt",        # from svhn rotation to svhn permutation
    "svhn_perm->svhn_rot": "distribution",  # add rotation to svhn
    "rot->rot": "distribution",             # add rotation to mnist
    "svhn_rot->svhn_rot": "distribution",   # add rotation to svhn
    "rot->svhn_rot": "domain",              # from mnist to svhn with the same rotation
    "svhn_rot->rot": "domain",              # from svhn to mnist with the same rotation
    "svhn_perm->perm": "domain",            # from svhn to mnist with the same permutation
    "perm->svhn_perm": "domain",            # from mnist to svhn with the same permutation
    "perm->svhn_rot": "abrupt",             # from mnist permutation to svhn rotation
    "svhn_rot->perm": "abrupt",             # from svhn rotation to mnist permutation
    "rot->svhn_perm": "abrupt",             # from mnist rotation to svhn permutation
    "svhn_perm->rot": "abrupt",             # from svhn permutation to mnist rotation
    "perm->perm": "abrupt",                 # switch mnist permutation
    "svhn_perm->svhn_perm": "abrupt",       # switch svhn permutation
}


def get_surprise_type(prev, cur):
    return SURPRISE_MAP["->".join([prev, cur])]


def create_surpriseMNIST_subtask(configs, seed):
    task_name = configs['dataset']
    prev_task = configs['prev_dataset']
    prev_trans = configs['prev_transform']
    dataset_args = {
        "root": configs['root'],
        "download": True
    }
    if 'svhn' in task_name:
        dataset_class = SVHN
        dataset_args["split"] = configs['split']
    else:
        dataset_class = MNIST
        dataset_args["train"] = configs['split'] == 'train'
    transform, task_transform = make_transform(task_name, seed, prev_task, prev_trans)
    dataset_args["transform"] = transform
    return dataset_class(**dataset_args), task_transform


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


class SurpriseScheduler:
    def __init__(self, config):
        self.train_loader = None
        self.val_loaders = []
        self.config = config
        # task
        self.task_class = SurpriseMNIST
        self.n_tasks = config.n_tasks
        self.task_type = config.task_type
        assert self.task_type in ["til", "cil", "dil"]
        if self.task_type == "dil":
            self.head_size = self.task_class.HEAD_SIZE
        else:
            self.head_size = self.n_tasks * self.task_class.HEAD_SIZE

        # every n step change the dataset abruptly
        self.surprise_interval = config.surprise_interval
        self.surprises = []
        self.interval = 0
        self.prev_task = ""
        self.prev_transform = None

        if self.config.dataset_root_path is None:
            self.config.dataset_root_path = './data'

        self.init_dataset()

    def init_dataset(self):
        # create surprise task
        seed = get_random_seed()
        task = self.random_task()
        if len(self.surprises) > 0:
            prev = self.surprises[-1][0]
            print(f'Surprise: {prev}->{task} ({get_surprise_type(prev, task)})')
        self.surprises.append((task, seed))
        train_data = self.task_class(root=self.config.dataset_root_path,
                                     task=task, seed=seed,
                                     prev_task=self.prev_task,
                                     prev_transform=self.prev_transform,
                                     split='train')
        val_data = self.task_class(root=self.config.dataset_root_path,
                                   task=task, seed=seed,
                                   prev_task=self.prev_task,
                                   prev_transform=self.prev_transform,
                                   split='test')
        # label offset for cil, til
        self.offset_targets(val_data)

        self.prev_task = task
        self.prev_transform = train_data.task_transform

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

    def offset_targets(self, data):
        if self.task_type != "dil":
            offset = self.task_class.HEAD_SIZE * len(self.val_loaders)
            if hasattr(data.dataset, 'targets'):
                data.dataset.targets += offset
            elif hasattr(data.dataset, 'labels'):
                data.dataset.labels += offset
            else:
                raise ValueError('Cannot locate labels in dataset')

    def step(self):
        self.interval += 1
        # surprise by epoch intervals
        if self.interval == self.surprise_interval and len(self) < self.n_tasks:
            self.init_dataset()
            self.interval = 0
            return 1
        return 0

    def random_task(self):
        return random.choice(SURPRISE)

    def __len__(self):
        return len(self.val_loaders)


class SurpriseMNIST(Dataset):
    HEAD_SIZE = 10

    def __init__(self, root, task, seed, prev_task, prev_transform, split):
        super().__init__()
        self.root = root
        self.split = split
        # dataset
        task_args = {'dataset': task,
                     'prev_dataset': prev_task,
                     'prev_transform': prev_transform,
                     'split': self.split,
                     'root': self.root}
        self.dataset, self.task_transform = create_surpriseMNIST_subtask(task_args, seed=seed)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return {'x': x, 'y': y}

    def __len__(self):
        return len(self.dataset)
