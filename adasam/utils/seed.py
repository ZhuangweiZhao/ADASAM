"""
随机种子工具 | Random Seed Utility.

统一所有实验脚本的随机种子设置。
Unified random seed setup for all experiment scripts.

用法 | Usage::
    from adasam.utils.seed import set_seed, get_worker_init_fn
    set_seed(42)

    # 在 DataLoader 中使用 worker_init_fn 保证数据增强可复现
    # Use worker_init_fn in DataLoader for reproducible data augmentation
    loader = DataLoader(ds, ..., worker_init_fn=get_worker_init_fn(42))
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    固定所有随机种子以保证可复现性。
    Fix all random seeds for reproducibility.

    包括 | Includes:
        - Python random
        - NumPy
        - PyTorch (CPU + CUDA)
        - cuDNN (deterministic mode)

    :param seed: 随机种子值 | Random seed value.
    :type seed: int
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 确定性 cuDNN (可能稍慢, 但保证可复现)
    # Deterministic cuDNN (slightly slower, but guarantees reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_worker_init_fn(seed: int = 42):
    """
    返回 DataLoader 的 worker_init_fn，确保每个 worker 的可复现性。
    Returns a DataLoader worker_init_fn that ensures per-worker reproducibility.

    每个 DataLoader worker 是独立进程，需要独立设置种子。
    Seed = global_seed + worker_id，保证不同 worker 产生不同但可复现的随机序列。
    Each DataLoader worker is a separate process and needs its own seed.
    Seed = global_seed + worker_id, ensuring different workers produce
    different but reproducible random sequences.

    用法 | Usage::
        loader = DataLoader(ds, ..., worker_init_fn=get_worker_init_fn(42))

    :param seed: 全局种子值 | Global seed value.
    :type seed: int

    :returns: worker_init_fn callable.
    :rtype: callable
    """
    def _worker_init_fn(worker_id: int) -> None:
        """初始化单个 DataLoader worker 的随机种子 | Init random seed for a single DataLoader worker."""
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return _worker_init_fn
