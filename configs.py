from trainer import ModelConfigBase
from trainer import TrainConfigBase
from trainer import OptimizerConfig
from trainer import SchedulerConfig
from typing import Literal


class SCModelConfig:
    regularization_coef: float = 25


class SCTrainConfig:
    # data
    dataset: str = "mnist"
    surprise_interval: int = 2
    dataset_root_path: str = None
    # train
    n_tasks: int = 5
    task_type: Literal["cil", "dil"] = "cil"
    epochs: int = 3              # max epoch for one task
    surprise_check_epochs: int = 1       # epochs starting to check surprises
    lr: float = 0.03
    batch_size: int = 128
    device: str = 'cuda:0'
    # buffer-memory
    buffer_size: float = 5000  # Number of examples to send back from the expert
    memory_size: float = 1000
