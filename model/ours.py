from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from model.base import CLModel
from modules.memory import Buffer

'''
    Experience Replay + Dataset Distillation
'''
class ERDD(CLModel):
    def __init__(self, config):
        super().__init__(config)
        self.buffer = Buffer(buffer_size=config.buffer_size, device='cpu')
        self.sample_per_task = config.buffer_size
        self.replay_size = config.batch_size
        self.distill_iter = config.distill_iter
        self.distill_lr = config.distill_lr

    def observe(self, x, y, attn_mask):
        logits, _ = self.forward(x, attn_mask)
        loss = F.cross_entropy(logits, y)
        aux_loss = 0
        if not self.buffer.is_empty():
            _, buffer_y, buffer_x, buffer_attn_mask = self.buffer.get_data(self.replay_size, self.device)
            s_logits, _ = self.forward(inputs_embeds=buffer_x, attn_mask=buffer_attn_mask)
            aux_loss += F.cross_entropy(s_logits, buffer_y)
        loss = loss + aux_loss
        return logits, loss

    def end_task(self, dataloader, label_offset):
        self.backbone.eval()
        # selected = self._random_sampling(dataloader, k=self.sample_per_task)
        selected = self._importance_sampling(dataloader, label_offset, k=self.sample_per_task)
        s_x, s_y, s_attn_mask = self._dataset_distillation(
            *self._sample_by_idx(dataloader, label_offset, selected)
        )
        self.buffer.add_data(x=torch.ones_like(s_attn_mask), y=s_y, logits=s_x, attn_mask=s_attn_mask)
        self.backbone.train()

    def _importance_sampling(self, dataloader, label_offset, k=50):
        samples = []
        total = len(dataloader.dataset)
        params = [p for p in list(self.backbone.parameters()) if p.requires_grad]
        progress = tqdm(range(total), desc='Sampling datapoints', position=0, leave=True)
        for idx in range(total):
            x = dataloader.dataset[idx]['input_ids']
            y = dataloader.dataset[idx]['labels']
            y = (y + label_offset).long()
            m = dataloader.dataset[idx]['attention_mask']
            # compute grad norm
            self.backbone.zero_grad()
            output = self.backbone(input_ids=x.unsqueeze(0), attention_mask=m.unsqueeze(0))
            loss = F.cross_entropy(output.logits, target=y.unsqueeze(0))
            grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            grad_norm = (torch.concat([g.flatten() for g in grads if g is not None]) ** 2).sum().cpu()
            samples.append((idx, grad_norm))
            progress.update(1)
        self.backbone.zero_grad()
        # select k smallest grad norms
        samples = sorted(samples, key=lambda v: v[1])[:k]
        return [idx for idx, _ in samples]

    def _random_sampling(self, dataloader, k=50):
        random_idx = np.random.choice(np.arange(len(dataloader.dataset)), size=k, replace=False)
        return random_idx

    @torch.no_grad()
    def _sample_by_idx(self, dataloader, label_offset, samples):
        sample = dataloader.dataset[samples]
        x = sample['input_ids']
        emb = self.backbone.distilbert.embeddings(x.to(self.device)).cpu()
        y = sample['labels']
        y = y + label_offset
        m = sample['attention_mask']
        return emb, y, m

    def _dataset_distillation(self, gen_x, gen_y, gen_mask):
        gen_x, gen_y, gen_mask = gen_x.to(self.device), gen_y.to(self.device), gen_mask.to(self.device)
        gen_x = torch.nn.Parameter(gen_x, requires_grad=True)
        optimizer = Adam([gen_x], lr=self.distill_lr)
        progress = tqdm(range(self.distill_iter), position=0, leave=True)
        for e in range(self.distill_iter):
            optimizer.zero_grad()
            # shuffle
            perm = torch.randperm(len(gen_x))
            avg_loss = []
            for i in range(len(gen_x)):
                x, y, m = gen_x[perm][i], gen_y[perm][i], gen_mask[perm][i]
                self.backbone.zero_grad()
                output = self.backbone(inputs_embeds=x.unsqueeze(0), attention_mask=m.unsqueeze(0))
                loss = F.cross_entropy(output.logits, target=y.unsqueeze(0))
                avg_loss.append(loss.item())
                loss.backward()
            optimizer.step()
            progress.set_description(
                f"Distilling dataset - [Iter: {e}/{self.distill_iter} ] Training: loss: {np.mean(avg_loss):.4f}"
            )
            progress.update(1)
        self.backbone.zero_grad()
        return gen_x.detach().cpu(), gen_y.cpu(), gen_mask.cpu()
