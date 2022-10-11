import torch
from sklearn.metrics import accuracy_score
from torch.optim import SGD
from tqdm import tqdm

from configs import ModelConfig, TrainConfig
from dataset.dataset import TaskStream
from model import ExpertModel
from modules.ewc import EWC
from modules.memory import Buffer
from utils import batch_to_device


class ContinualNLP:
    def __init__(self, train_config, model_config):
        self.train_config = train_config
        self.device = train_config.device
        self.task_stream = TaskStream(train_config)
        # model
        self.model = ExpertModel(
            model_config,
            seq_len=self.task_stream.seq_len,
            head_size=self.task_stream.n_classes
        ).to(self.device)
        self.buffer = Buffer(buffer_size=train_config.buffer_size, device='cpu')
        self.ewc = EWC(device=self.device)
        self.optimizer = SGD(self.model.parameters(), lr=train_config.lr)

    def run(self):
        for t in range(self.task_stream.n_tasks):
            self.task_stream.new_task()
            train_loader = self.task_stream.train_loader
            self.model.update(self.ewc)
            self.train_loop(train_loader, task=t)
            # ewc
            self.ewc.update(self.model, train_loader)
            # # memory
            # self.memory.add_data(*buffer.get_all_data())
            # evaluate
            self.evaluate(val_loaders=self.task_stream.val_loaders)

    def train_loop(self, train_dataloader, task=0):
        model = self.model
        device = self.device
        epochs = self.train_config.epochs
        optimizer = self.optimizer

        tqdm_bar = tqdm(range(epochs), desc=f'Training task {task}')
        for e in tqdm_bar:
            for batch in train_dataloader:
                batch = batch_to_device(batch, device)
                logits, loss = model(**batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tqdm_bar.set_postfix(loss=loss.item())

    def evaluate(self, val_loaders):
        print()
        print('Evaluating')
        total_acc = 0
        for i, vl in enumerate(val_loaders):
            y_pred, y_true = self.get_preds(self.model, val_loader=vl)
            task_acc = accuracy_score(y_true, y_pred)
            total_acc += task_acc
            print(f'Task {i}, acc {round(task_acc * 100, 2)}')
        avg_acc = total_acc / len(val_loaders)
        print(f'Average acc {round(avg_acc * 100, 2)}')
        print()

    @torch.no_grad()
    def get_preds(self, model, val_loader):
        model.eval()
        y_pred, y_true = [], []
        for batch in val_loader:
            batch = batch_to_device(batch, self.device)
            logits = model.get_preds(**batch)
            y_pred.extend(list(logits.detach().argmax(-1).cpu().numpy()))
            y_true.extend(list(batch['y'].cpu().numpy()))
        model.train()
        return y_pred, y_true


if __name__ == "__main__":
    model_config = ModelConfig()
    train_config = TrainConfig()
    sc = ContinualNLP(train_config, model_config)
    sc.run()
