import numpy as np
import torch
import torchvision.transforms as T


def get_generator(seed=None):
    if seed is None:
        seed = get_random_seed()
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def get_random_seed():
    return int(torch.empty((), dtype=torch.int64).random_().item())


def extend_transform_to_tensor(transforms):
    transform = T.Compose([T.ToPILImage()] + transforms)
    return transform