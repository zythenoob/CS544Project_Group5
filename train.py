import argparse
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score
from torch.optim import SGD, Adam
import trainer.utils.train as tutils
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import SCModelConfig, SCTrainConfig
from dataset.dataset import SurpriseScheduler, SurpriseMNIST
from model import ExpertModel
from modules.ewc import EWC
from modules.memory import Buffer


class SurpriseConsolidation:
    def __init__(self, train_config, model_config):
        self.train_config = train_config
        self.device = train_config.device
        self.data_scheduler = SurpriseScheduler(train_config)
        self.model = ExpertModel(model_config, head_size=self.data_scheduler.head_size).to(self.device)
        self.memory = Buffer(buffer_size=train_config.memory_size, device='cpu')
        self.ewc = EWC(device=self.device)
        self.optimizer = SGD(self.model.parameters(), lr=train_config.lr)

        self.boundary_detected = []

    def run(self):
        while len(self.data_scheduler) < self.train_config.n_tasks:
            self.model.update(self.ewc)
            buffer = self.train_loop(self.data_scheduler)
            # ewc
            buffer_dataloader = DataLoader(buffer.to_dataset(), shuffle=False, batch_size=self.train_config.batch_size)
            self.ewc.update(self.model, buffer_dataloader)
            # # memory
            # self.memory.add_data(*buffer.get_all_data())
            # evaluate
            self.evaluate(val_loaders=self.data_scheduler.val_loaders[:-1])

    def train_loop(self, data_scheduler):
        model = self.model
        device = self.device
        optimizer = self.optimizer
        # surprise
        epochs = self.train_config.epochs
        switch_epochs = self.train_config.surprise_interval
        check_epoch = self.train_config.surprise_check_epochs
        context_switch = False
        manual_switch = False
        # reservoir buffer for expert training
        buffer = Buffer(buffer_size=train_config.buffer_size, device='cpu')
        print('surprises:', data_scheduler.surprises)
        tqdm_bar = tqdm(range(epochs), desc='Training')
        for e in tqdm_bar:
            if data_scheduler.interval == data_scheduler.surprise_interval:
                data_scheduler.make_surprise()
                manual_switch = True

            for batch in data_scheduler.train_loader:
                batch = tutils.iter_to_device(batch, device)
                logits, loss, task_loss = model(**batch)
                # check for surprise
                if e > check_epoch:
                    context_switch, z_scores = model.surprise_check(task_loss)

                optimizer.zero_grad()
                if not context_switch and not manual_switch:
                    loss.backward()
                    optimizer.step()
                    buffer.add_data(batch['x'], batch['y'], logits.detach())
                    tqdm_bar.set_postfix(loss=loss.item())
                elif context_switch:
                    print(f'Context switch! z_scores={z_scores}')
                    break
                elif manual_switch:
                    print(f'Manual switch! z_scores={z_scores}')
                    break

            if context_switch or manual_switch:
                model.end_task(SurpriseMNIST.HEAD_SIZE)
                self.boundary_detected.append(context_switch == manual_switch)
                if not manual_switch:
                    data_scheduler.make_surprise()
                break
            data_scheduler.interval += 1
        print(f'Seen {len(data_scheduler.surprises)} tasks, detected {sum(self.boundary_detected)} boundaries. '
              f'Acc: {sum(self.boundary_detected) / (len(data_scheduler.surprises) - 1)}')
        return buffer

    def evaluate(self, val_loaders):
        print()
        print(f'Evaluating {len(val_loaders)} tasks')
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
            batch = tutils.iter_to_device(batch, self.device)
            logits = model.get_preds(**batch)
            y_pred.extend(list(logits.detach().argmax(-1).cpu().numpy()))
            y_true.extend(list(batch['y'].cpu().numpy()))
        model.train()
        return y_pred, y_true


if __name__ == "__main__":
    model_config = SCModelConfig()
    train_config = SCTrainConfig()
    sc = SurpriseConsolidation(train_config, model_config)
    sc.run()
