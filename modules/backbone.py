import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification, GPT2ForSequenceClassification


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


def make_backbone(name, seq_len, n_classes):
    print("Make backbone:", name)
    model_class, key = name_to_backbone[name]
    if model_class == LinearModel:
        return LinearModel(seq_len, n_classes)
    else:
        return model_class.from_pretrained(key, num_labels=n_classes, problem_type='single_label_classification')


name_to_backbone = {
    'linear': (LinearModel, None),
    'distilbert': (DistilBertForSequenceClassification, "distilbert-base-uncased"),
    'gpt2': (GPT2ForSequenceClassification, "gpt2"),
}
