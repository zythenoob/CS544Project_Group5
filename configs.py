

class ModelConfig:
    backbone: str = 'distilbert'
    regularization_coef: float = 25

class TrainConfig:
    # data
    dataset: str = "glue"
    dataset_root_path: str = None
    # train
    epochs: int = 3              # training epoch per task
    lr: float = 2e-5
    batch_size: int = 32
    device: str = 'cuda:0'
    # buffer
    buffer_size: float = 1000
