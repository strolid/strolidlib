import importlib
import types

import pytest

import strolidlib.gpu as gpu


@pytest.fixture(autouse=True)
def reload_gpu_module():
    importlib.reload(gpu)
    yield


@pytest.fixture
def fake_gpu_env(monkeypatch):
    class FakeCuda:
        def __init__(self):
            self.available = False
            self.set_calls = []
            self.sync_calls = []
            self.current_device_id = 0
            self.count = 1

        def is_available(self):
            return self.available

        def set_device(self, device_id):
            self.current_device_id = device_id
            self.set_calls.append(device_id)

        def synchronize(self, device_id=None):
            self.sync_calls.append(device_id)

        def device_count(self):
            return self.count

        def current_device(self):
            return self.current_device_id

    class FakeTensor:
        def __init__(self, data=None):
            self.data = data
            self.detach_called = False
            self.to_device = None

        def to(self, device=None, non_blocking=None):
            self.to_device = device
            return self

        def detach(self):
            self.detach_called = True
            return self

    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_cuda = FakeCuda()

    backend_cuda = types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False))
    backend_cudnn = types.SimpleNamespace(allow_tf32=False)

    precision_calls: list[str] = []

    def set_precision(level: str):
        precision_calls.append(level)

    fake_torch = types.SimpleNamespace(
        cuda=fake_cuda,
        Tensor=FakeTensor,
        device=lambda name: f"device:{name}",
        backends=types.SimpleNamespace(cuda=backend_cuda, cudnn=backend_cudnn),
        no_grad=lambda: FakeNoGrad(),
        set_float32_matmul_precision=set_precision,
    )

    class FakeArray:
        def __init__(self, data, ndim=1):
            self.data = data
            self.ndim = ndim

        def astype(self, dtype, copy=False):
            return self

    def fake_mean(array, axis=0):
        if not isinstance(array, FakeArray):
            raise TypeError("unexpected array type")
        if array.ndim <= 1:
            return array
        # simple column-wise average for 2D structures
        columns = list(zip(*array.data))
        averaged = [sum(col) / len(col) for col in columns]
        return FakeArray(averaged, ndim=1)

    fake_numpy = types.SimpleNamespace(
        ndarray=FakeArray,
        float32="float32",
        mean=fake_mean,
    )

    monkeypatch.setattr(gpu, "torch", fake_torch, raising=False)
    monkeypatch.setattr(gpu, "numpy", fake_numpy, raising=False)
    monkeypatch.setattr(gpu, "pyannote", object(), raising=False)

    return types.SimpleNamespace(
        torch=fake_torch,
        numpy=fake_numpy,
        FakeArray=FakeArray,
        precision_calls=precision_calls,
    )


def test_set_cuda_device_maybe_raises_when_unavailable(fake_gpu_env):
    fake_gpu_env.torch.cuda.available = False
    with pytest.raises(ValueError):
        gpu.set_cuda_device_maybe(0)


def test_set_cuda_device_maybe_sets_device_when_available(fake_gpu_env):
    fake_gpu_env.torch.cuda.available = True
    gpu.set_cuda_device_maybe(2)
    assert fake_gpu_env.torch.cuda.set_calls == [2]


def test_load_transcription_model_invokes_loader(monkeypatch, fake_gpu_env):
    fake_gpu_env.torch.cuda.available = True

    calls = []

    class FakeModel:
        def __init__(self):
            self.eval_called = False
            self.to_device = None

        def eval(self):
            self.eval_called = True
            return self

        def to(self, device=None, non_blocking=None):
            self.to_device = device
            return self

    fake_model = FakeModel()

    def fake_loader(name: str):
        calls.append(name)
        return fake_model

    monkeypatch.setattr(gpu, "load_nemo_model_maybe_refresh", fake_loader)

    result = gpu.load_transcription_model("en_real_quick_0", cuda_device=1)

    assert calls == ["nvidia/parakeet-tdt_ctc-110m"]
    assert result is fake_model
    assert fake_model.eval_called is True
    assert fake_model.to_device in {"cuda", "device:cuda"}
    assert fake_gpu_env.torch.cuda.set_calls == [1]


def test_load_transcription_model_invalid_name(fake_gpu_env):
    with pytest.raises(ValueError):
        gpu.load_transcription_model("unknown")


def test_transcribe_trims_response(monkeypatch, fake_gpu_env):
    fake_gpu_env.torch.cuda.available = False

    class FakeModel:
        def __init__(self):
            self.calls = []

        def transcribe(self, *, audio, batch_size, logprobs):
            self.calls.append((audio, batch_size, logprobs))
            return ["  hello world  "]

    fake_audio = fake_gpu_env.FakeArray([0.1, 0.2, 0.3], ndim=1)

    model = FakeModel()
    transcript = gpu.transcribe(model, fake_audio)

    assert transcript == "hello world"
    assert model.calls
    sent_audio, sent_batch_size, sent_logprobs = model.calls[0]
    assert sent_audio == [fake_audio]
    assert sent_batch_size == 1
    assert sent_logprobs is False


def test_enable_tf32_sets_backend_flags(fake_gpu_env):
    gpu.enable_tf32()
    assert fake_gpu_env.torch.backends.cuda.matmul.allow_tf32 is True
    assert fake_gpu_env.torch.backends.cudnn.allow_tf32 is True
    assert fake_gpu_env.precision_calls == ["high"]

