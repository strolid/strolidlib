"""Shared utilities for Conserver links."""

from .audio import (
    DEFAULT_SAMPLE_RATE,
    LoadedTensor,
    load_tensor_from_vcon,
    move_tensor_to_device,
    normalize_vcon,
)
from .gpu import (
    cuda_synchronize,
    enable_tf32,
    get_current_cuda_device,
    get_cuda_device_count,
    is_cuda_available,
    move_to_gpu_maybe,
    set_cuda_device,
)
from .utils import (
    are_we_parallel,
    opts_have_changed,
    seconds_to_ydhms,
)

__all__ = [
    "are_we_parallel",
    "opts_have_changed",
    "seconds_to_ydhms",
    "DEFAULT_SAMPLE_RATE",
    "LoadedTensor",
    "load_tensor_from_vcon",
    "move_tensor_to_device",
    "normalize_vcon",
    "move_to_gpu_maybe",
    "set_cuda_device",
    "is_cuda_available",
    "get_cuda_device_count",
    "get_current_cuda_device",
    "cuda_synchronize",
    "enable_tf32",
]

