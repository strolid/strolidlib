import base64
from types import SimpleNamespace

import numpy as np
import torch

from strolidlib.audio import load_tensor_from_vcon, normalize_vcon


def test_load_tensor_from_embedded_tensor_round_trip():
    samples = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    payload = base64.b64encode(samples.tobytes()).decode("utf-8")
    vcon = {
        "uuid": "sample",
        "dialog": [
            {
                "type": "tensor",
                "data_b64": payload,
                "dtype": "float32",
                "sample_rate": 8000,
            }
        ],
    }

    loaded = load_tensor_from_vcon(vcon)
    assert loaded.sample_rate == 8000
    assert torch.allclose(loaded.tensor, torch.tensor(samples))


def test_normalize_vcon_assigns_back_when_mutated():
    original = {"uuid": "abc", "dialog": []}
    wrapper = SimpleNamespace(vcon_dict=original, to_dict=lambda: original.copy())

    vcon_dict, assign_back = normalize_vcon(wrapper)
    vcon_dict["dialog"].append({"type": "tensor"})
    assert assign_back is not None
    assign_back(vcon_dict)
    assert wrapper.vcon_dict["dialog"]  # ensure mutation persisted

