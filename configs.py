from dataclasses import dataclass
from omegaconf import OmegaConf


@dataclass
class TrainConfig:
    # data
    dataset: str = "mnist"
    dataset_root_path: str = None
    save_dir: str = 'save'
    # cl
    method: str = 'ewc'
    backbone: str = 'distilbert'
    # train
    epochs: int = 3              # training epoch per task
    lr: float = 0.1
    batch_size: int = 128
    device: str = 'cuda:0'
    n_tasks: int = 7
    seq_len: int = 256
    head_size: int = 10
    # other args
    buffer_size: float = 100
    reg_coef: float = 100
    syn_iter: int = 200
    syn_lr: float = 0.001
    # debug
    debug: bool = False

    def __init__(self, values):
        super(TrainConfig, self).__init__()
        for k, v in values.items():
            setattr(self, k, v)


def load_config(path):
    config = OmegaConf.load(path)
    return TrainConfig(config['train'])
