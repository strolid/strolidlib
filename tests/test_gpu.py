"""Tests for strolidlib.gpu module."""

import pytest
import torch
import numpy as np

from strolidlib.gpu import (
    is_cuda_available,
    is_tensor,
    is_numpy,
    numpy_to_tensor,
    is_mono,
    convert_to_mono,
    ensure_mono,
    tensor_to_contiguous,
    strolid_model_name_to_nemo_model_name,
)


class TestIsCudaAvailable:
    """Tests for is_cuda_available function."""

    def test_returns_bool(self):
        result = is_cuda_available()
        assert isinstance(result, bool)


class TestIsTensor:
    """Tests for is_tensor function."""

    def test_tensor_returns_true(self):
        t = torch.tensor([1, 2, 3])
        assert is_tensor(t) is True

    def test_numpy_returns_false(self):
        arr = np.array([1, 2, 3])
        assert is_tensor(arr) is False

    def test_list_returns_false(self):
        assert is_tensor([1, 2, 3]) is False

    def test_none_returns_false(self):
        assert is_tensor(None) is False


class TestIsNumpy:
    """Tests for is_numpy function."""

    def test_numpy_returns_true(self):
        arr = np.array([1, 2, 3])
        assert is_numpy(arr) is True

    def test_tensor_returns_false(self):
        t = torch.tensor([1, 2, 3])
        assert is_numpy(t) is False

    def test_list_returns_false(self):
        assert is_numpy([1, 2, 3]) is False


class TestNumpyToTensor:
    """Tests for numpy_to_tensor function."""

    def test_converts_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = numpy_to_tensor(arr)
        assert is_tensor(result)
        assert result.shape == (3,)

    def test_preserves_values(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = numpy_to_tensor(arr)
        assert torch.allclose(result, torch.tensor([1.0, 2.0, 3.0]))

    def test_2d_array(self):
        arr = np.array([[1, 2], [3, 4]])
        result = numpy_to_tensor(arr)
        assert result.shape == (2, 2)


class TestIsMono:
    """Tests for is_mono function."""

    def test_mono_audio(self):
        mono = torch.randn(1, 16000)  # 1 channel, 1 second at 16kHz
        assert is_mono(mono) is True

    def test_stereo_audio(self):
        stereo = torch.randn(2, 16000)  # 2 channels
        assert is_mono(stereo) is False


class TestConvertToMono:
    """Tests for convert_to_mono function."""

    def test_converts_stereo_to_mono(self):
        stereo = torch.randn(2, 16000)
        result = convert_to_mono(stereo)
        assert result.shape[0] == 1

    def test_preserves_length(self):
        stereo = torch.randn(2, 16000)
        result = convert_to_mono(stereo)
        assert result.shape[1] == 16000


class TestEnsureMono:
    """Tests for ensure_mono function."""

    def test_mono_unchanged(self):
        mono = torch.randn(1, 16000)
        result = ensure_mono(mono)
        assert result.shape == mono.shape

    def test_stereo_converted(self):
        stereo = torch.randn(2, 16000)
        result = ensure_mono(stereo)
        assert result.shape[0] == 1


class TestTensorToContiguous:
    """Tests for tensor_to_contiguous function."""

    def test_makes_contiguous(self):
        t = torch.randn(10, 10).t()  # Transposed tensor is not contiguous
        assert not t.is_contiguous()
        result = tensor_to_contiguous(t)
        assert result.is_contiguous()


class TestStrolidModelNameToNemoModelName:
    """Tests for strolid_model_name_to_nemo_model_name function."""

    def test_en_real_quick_0(self):
        result = strolid_model_name_to_nemo_model_name("en_real_quick_0")
        assert result == "nvidia/parakeet-tdt_ctc-110m"

    def test_unknown_model(self):
        result = strolid_model_name_to_nemo_model_name("unknown_model")
        assert result is None

    def test_empty_string(self):
        result = strolid_model_name_to_nemo_model_name("")
        assert result is None

