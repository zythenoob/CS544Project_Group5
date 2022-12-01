import os
import numpy as np
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from transformers import AdamW

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

from configs import load_config
from dataset.dataset import TaskStream
from modules.backbone import LinearProbe, make_backbone
from utils import batch_to_device

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class ProbeTrainer:
    def __init__(self, train_config):
        self.train_config = train_config
        self.device = train_config.device
        self.task_stream = TaskStream(train_config)
        self.train_config.seq_len = self.task_stream.seq_len
        self.train_config.head_size = self.task_stream.n_classes
        self.train_config.n_tasks = self.task_stream.n_tasks
        # model
        self.pretrained = make_backbone(
            name='distilbert',
            seq_len=self.train_config.seq_len,
            n_classes=self.train_config.head_size
        )
        self.pretrained.load_state_dict(
            torch.load(os.path.join(train_config.save_dir, train_config.method, 'model.pt'))
        )
        self.pretrained.to(self.device)
        self.model = LinearProbe(n_tasks=self.task_stream.n_tasks, n_hidden=7).to(self.device)
        # freeze CL model
        self.pretrained.eval()
        for p in self.pretrained.parameters():
            p.requires_grad = False

        self.optimizer = AdamW(self.model.parameters(), lr=train_config.lr)
        self.train_tqdm = tqdm(range(1))
        self.save_dir = os.path.join(self.train_config.save_dir, self.train_config.method, 'probing')
        os.makedirs(self.save_dir, exist_ok=True)

    def run(self):
        self.task_stream.new_task()
        train_loader = self.task_stream.train_loader
        self.train_probe(train_loader)
        # evaluate
        self.evaluate(val_loader=self.task_stream.val_loaders[0])
        # save
        self.save_model()

    def train_probe(self, train_dataloader):
        pretrained = self.pretrained
        model = self.model
        device = self.device
        epochs = self.train_config.epochs
        optimizer = self.optimizer
        task_name = self.task_stream.task_names[0]
        # train statistics
        losses = [[] for _ in range(7)]

        model.train()
        self.train_tqdm = tqdm(range(epochs * len(train_dataloader)), position=0, leave=True)
        for e in range(epochs):
            for i, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                batch = batch_to_device(batch, device)
                x, y, attn_mask = batch['input_ids'], batch['labels'], batch['attention_mask']

                output = pretrained(x, attention_mask=attn_mask, output_hidden_states=True)
                # take cls token for probing
                hiddens = [h[:, 0, :] for h in output.hidden_states]
                hs_logits = model(hiddens)
                for idx, logits in enumerate(hs_logits):
                    loss = F.cross_entropy(logits, y)
                    losses[idx].append(loss.item())
                    loss.backward()
                optimizer.step()

                # progress bar
                self.train_tqdm.update(1)
                mean_loss = [round(np.mean(probe_loss[-30:]), 3) if len(probe_loss) > 0 else 0 for probe_loss in losses]
                self.train_tqdm.set_description(
                    f"Training on {task_name} - [Ep: {e}/{epochs} ] Training: loss: {mean_loss}"
                )

                if self.train_config.debug and i == 3:
                    break

    @torch.no_grad()
    def evaluate(self, val_loader):
        print()
        self.model.eval()
        y_preds = [[] for _ in range(7)]
        y_true = []
        for batch in tqdm(val_loader, desc='Probing', position=0, leave=True):
            batch = batch_to_device(batch, self.device)
            x, y, attn_mask = batch['input_ids'], batch['labels'], batch['attention_mask']

            output = self.pretrained(x, attention_mask=attn_mask, output_hidden_states=True)
            # take cls token for probing
            hiddens = [h[:, 0, :] for h in output.hidden_states]
            hs_logits = self.model(hiddens)
            for idx, logits in enumerate(hs_logits):
                y_preds[idx].append(logits.argmax(-1).cpu().numpy())
            y_true.append(y.cpu().numpy())
        y_preds = [np.concatenate(pred) for pred in y_preds]
        y_true = np.concatenate(y_true)
        layer_results = []
        # output result
        for idx, pred in enumerate(y_preds):
            print()
            totol_acc = accuracy_score(y_true, pred)
            print(f'Layer {idx}: total acc: {totol_acc}')
            unique, counts = np.unique(pred, return_counts=True)
            bin_count = dict(zip(unique, counts))
            print(f'bin count: {bin_count}')
            # task-wise acc
            task_wise_score = []
            for t in range(self.task_stream.n_tasks):
                t_idx = np.where(y_true == t)[0]
                if t in bin_count:
                    bin_count[t] = float(bin_count[t]) / len(t_idx)
                task_wise_score.append(accuracy_score(y_true[t_idx], pred[t_idx]))
            print(f'task-wise f1: {list(task_wise_score)}')
            layer_results.append((idx, bin_count, task_wise_score))
        print()
        self.plot(layer_results)

    def plot(self, layer_results):
        save_dir = self.save_dir
        fig, axs = plt.subplots(nrows=7, ncols=1, sharex='all', sharey='all')
        axs_bar = []
        labels = ['Task ' + str(i + 1) for i in range(self.task_stream.n_tasks)]
        x = np.arange(1, self.task_stream.n_tasks + 1)
        width = 0.4
        for i, bin_count, task_wise_score in layer_results:
            ax = axs[i]
            # f1
            ax.bar(x - width / 2, task_wise_score, 0.4, color='b')
            ax.set_xticks(x, labels)
            # bin count
            ax_bar = ax.twinx()
            bin_count = [bin_count[c] if c in bin_count else 0 for c in range(self.task_stream.n_tasks)]
            ax_bar.bar(x + width / 2, bin_count, 0.4, color='r')
            axs_bar.append(ax_bar)
        axs[3].set_ylabel("Accuracy", color='b')
        axs_bar[3].set_ylabel("Bin count ratio", color='r')
        fig.suptitle(f'Layer-wise Probing for {self.train_config.method.upper()}')
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'probe.png'), dpi=fig.dpi)

    def save_model(self):
        save_dir = self.save_dir
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'probe.pt'))
        with open(os.path.join(save_dir, 'probe_config.yaml'), 'w') as f:
            OmegaConf.save(config=self.train_config, f=f)


if __name__ == "__main__":
    config = load_config('config/probe.yaml')
    c_NLP = ProbeTrainer(config)
    c_NLP.run()
