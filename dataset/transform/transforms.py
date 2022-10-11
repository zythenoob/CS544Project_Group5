from torchvision.transforms import transforms

from dataset.transform.permutation import FixedPermutation
from dataset.transform.rotation import FixedRotation


def make_transform_mnist(task_name, seed):
    normalize_transform = [
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
    # permuted
    if task_name.startswith('perm'):
        transform = transforms.Compose(
            [transforms.Resize(32),
             transforms.ToTensor()] + normalize_transform + [FixedPermutation(seed=seed)]
        )
    # rotated
    elif task_name.startswith('rot'):
        transform = transforms.Compose(
            [transforms.Resize(32),
             transforms.ToTensor()] + normalize_transform + [FixedRotation(seed=seed, deg_min=30, deg_max=180)]
        )
    else:
        transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()] + normalize_transform)
    return transform


def make_transform_svhn(task_name, seed):
    normalize_transform = [
        transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]),
        transforms.Grayscale(num_output_channels=1),
    ]
    # permuted
    if task_name.startswith('perm'):
        transform = transforms.Compose(
            [transforms.Resize(32),
             transforms.ToTensor()] + normalize_transform + [FixedPermutation(seed=seed)]
        )
    # rotated
    elif task_name.startswith('rot'):
        transform = transforms.Compose(
            [transforms.Resize(32),
             transforms.ToTensor()] + normalize_transform + [FixedRotation(seed=seed, deg_min=30, deg_max=180)]
        )
    else:
        transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()] + normalize_transform)
    return transform


def make_transform(task_name, seed, prev_task, prev_trans):
    if task_name.startswith('svhn'):
        base_transform = [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]),
            transforms.Grayscale(num_output_channels=1),
        ]
    else:
        base_transform = [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]

    if 'rot' in task_name:
        deg_min = 30
        task_transform = [FixedRotation(seed=seed, deg_min=deg_min, deg_max=180)]
        if prev_trans is not None:
            if isinstance(prev_trans[-1], FixedRotation):
                if prev_task == task_name == 'rot' or prev_task == task_name == 'svhn_rot':
                    # rot->rot, svhn_rot->svhn_rot
                    deg_min = (prev_trans[-1].degrees + 30) % 180
                    task_transform = prev_trans + [FixedRotation(seed=seed, deg_min=deg_min, deg_max=180)]
                elif 'rot' in prev_task:
                    # rot->svhn_rot, svhn_rot->rot
                    task_transform = prev_trans
            elif isinstance(prev_trans[-1], FixedPermutation):
                if not (prev_task.startswith('svhn') ^ task_name.startswith('svhn')):
                    # perm->rot， svhn_perm->svhn_rot
                    task_transform = prev_trans + [FixedRotation(seed=seed, deg_min=30, deg_max=180)]
        transform = base_transform + task_transform
    elif 'perm' in task_name:
        task_transform = [FixedPermutation(seed=seed)]
        if prev_trans is not None:
            if isinstance(prev_trans[-1], FixedPermutation):
                if prev_task.startswith('svhn') ^ task_name.startswith('svhn'):
                    # perm->svhn_perm, svhn_perm->perm
                    task_transform = prev_trans
        transform = base_transform + task_transform
    else:
        transform = base_transform
        task_transform = None

    return transforms.Compose(transform), task_transform


TRANSFORMS_MAP = {
    "mnist": make_transform_mnist,
    "svhn": make_transform_svhn,
}


def make_transforms(dataset_name):
    if dataset_name in TRANSFORMS_MAP:
        return TRANSFORMS_MAP[dataset_name]()
    else:
        raise NotImplementedError
