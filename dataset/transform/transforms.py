import torch
from torchvision.transforms import transforms

from dataset.transform.permutation import FixedPermutation
from dataset.transform.rotation import FixedRotation


def make_transform_mnist(seed):
    # permuted
    transform = transforms.Compose(
        [transforms.ToTensor(),
         FixedPermutation(seed=seed),
         transforms.Normalize((0.1307,), (0.3081,)),
         transforms.Lambda(lambda x: torch.flatten(x))]
    )
    return transform
