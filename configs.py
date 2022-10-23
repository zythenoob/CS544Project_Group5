from typing import Literal


class ModelConfig:
    regularization_coef: float = 25

class TrainConfig:
    # data
    dataset: str = "mnist"
    dataset_root_path: str = None
    # train
    epochs: int = 3              # training epoch per task
    lr: float = 0.1
    batch_size: int = 32
    device: str = 'cuda:0'
    # buffer
    buffer_size: float = 1000
