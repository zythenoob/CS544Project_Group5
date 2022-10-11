import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        hidden_dim = 100
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.classifier(x)
        return logits


def make_backbone(seq_len, n_classes):
    return LinearModel(seq_len, n_classes)
