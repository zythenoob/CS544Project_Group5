import os
import numpy as np
from transformers import AdamW

import torch
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

from configs import ModelConfig, TrainConfig
from dataset.dataset import TaskStream
from model import ExpertModel
from modules.ewc import EWC
from modules.memory import Buffer
from utils import batch_to_device, mask_logits_class

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


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
        self.optimizer = AdamW(self.model.parameters(), lr=train_config.lr)
        self.train_tqdm = tqdm(range(self.task_stream.n_tasks))

    def run(self):
        for t in range(self.task_stream.n_tasks):
            self.task_stream.new_task()
            train_loader = self.task_stream.train_loader
            self.model.update(self.ewc)
            self.train_loop(train_loader, task=t)
            # ewc
            self.ewc.update(self.model, train_loader, label_offset=self.task_stream.label_offset[t])
            # # memory
            # self.memory.add_data(*buffer.get_all_data())
            # evaluate
            self.evaluate(val_loaders=self.task_stream.val_loaders)

    def train_loop(self, train_dataloader, task=0):
        model = self.model
        device = self.device
        epochs = self.train_config.epochs
        optimizer = self.optimizer
        task_name = self.task_stream.task_names[task]
        label_offset = self.task_stream.label_offset
        # train statistics
        y_pred = []
        y_true = []
        losses = []
        mean_loss = 0
        mean_acc = 0

        model.backbone.train()
        self.train_tqdm = tqdm(range(epochs * len(train_dataloader)), position=0, leave=True)
        for e in range(epochs):
            for batch in train_dataloader:
                batch = batch_to_device(batch, device)
                x, y, attn_mask = batch['input_ids'], batch['labels'], batch['attention_mask']
                y = y + label_offset[task]
                logits, loss = model(x, y, attn_mask=attn_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                y_pred.append(logits.detach().cpu().argmax(-1).numpy())
                y_true.append(y.detach().cpu().numpy())
                # progress bar
                self.train_tqdm.update(1)
                if len(losses) > 0:
                    mean_loss = np.mean(losses[-30:])
                self.train_tqdm.set_description(
                    f"Training on {task_name} - [Ep: {e}/{epochs} ] Training: loss: {mean_loss:.4f} Mean acc: {mean_acc:.2f}"
                )
            # train accuracy
            mean_acc = accuracy_score(np.concatenate(y_pred), np.concatenate(y_true))

    def evaluate(self, val_loaders):
        print()
        print('Evaluating')
        total_acc = [0, 0]
        for i, vl in enumerate(val_loaders):
            cil_pred, til_pred, y_true = self.get_preds(self.model, val_loader=vl, task=i)
            cil_acc = accuracy_score(y_true, cil_pred)
            til_acc = accuracy_score(y_true, til_pred)
            total_acc[0] += cil_acc
            total_acc[1] += til_acc
            print(f'Task {i}: cil acc {round(cil_acc * 100, 2)}, til acc {round(til_acc * 100, 2)}')
        print(f'Average cil acc {round((total_acc[0] / len(val_loaders)) * 100, 2)}, '
              f'til acc {round((total_acc[1] / len(val_loaders)) * 100, 2)}')
        print()

    @torch.no_grad()
    def get_preds(self, model, val_loader, task):
        label_offset = self.task_stream.label_offset
        model.eval()
        y_pred_cil, y_pred_til, y_true = [], [], []
        for batch in val_loader:
            batch = batch_to_device(batch, model.device)
            x, y, attn_mask = batch['input_ids'], batch['labels'], batch['attention_mask']
            y = y + label_offset[task]
            logits = model.get_preds(x, y, attn_mask=attn_mask)
            y_pred_cil.append(logits.detach().argmax(-1).cpu().numpy())
            y_pred_til.append(mask_logits_class(logits.detach(), label_offset, task).argmax(-1).cpu().numpy())
            y_true.append(batch['labels'].cpu().numpy())
        model.train()
        return np.concatenate(y_pred_cil), np.concatenate(y_pred_til), np.concatenate(y_true)


if __name__ == "__main__":
    model_config = ModelConfig()
    train_config = TrainConfig()
    trainer = ContinualNLP(train_config, model_config)
    trainer.run()
