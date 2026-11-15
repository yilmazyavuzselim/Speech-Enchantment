"""Deterministic seeding utilities for reproducible experiments."""

import os
import random
import numpy as np
import torch


def set_global_seed(seed: int = 42, deterministic_cudnn: bool = True) -> None:
    """Sets random seeds across Python, NumPy, and PyTorch (CPU & CUDA)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn for reproducible multi-process data loading."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
