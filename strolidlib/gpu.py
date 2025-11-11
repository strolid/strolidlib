"""GPU helper utilities shared across Conserver components."""

from __future__ import annotations

from typing import Optional

# Instead of import torch-- we instead require the use of init() for imports.
torch = None
numpy = None
pyannote = None
EncDecSpeakerLabelModel = None
ASRModel = None

def init():
    global torch, numpy, pyannote, EncDecSpeakerLabelModel, ASRModel
    try:
        import torch
    except ImportError:
        pass
    try:
        import numpy
    except ImportError:
        pass
    try:
        import pyannote
    except ImportError:
        pass
    try:
        from nemo.collections.asr.models import EncDecSpeakerLabelModel
    except ImportError:
        pass
    try:
        from nemo.collections.asr.models import ASRModel
    except ImportError:
        pass

def is_cuda_available():
    return torch.cuda.is_available()

def is_tensor(obj) -> bool:
    return isinstance(obj, torch.Tensor)

def is_numpy(obj) -> bool:
    return isinstance(obj, numpy.ndarray)

def is_pyannote_pipeline(obj) -> bool:
    return hasattr(obj, "__class__") and "pyannote" in str(type(obj))

def move_to_gpu_maybe(obj):
    if not is_cuda_available():
        return obj

    if is_tensor(obj):
        on_gpu = obj.to(device="cuda", non_blocking=True)
        on_gpu.detach()
        return on_gpu

    # pyannote Pipeline.to(...) does not accept non_blocking
    if is_pyannote_pipeline(obj):
        return obj.to(torch.device("cuda"))

    try:
        return obj.to(device="cuda", non_blocking=True)
    except TypeError:
        pass
    return obj.to(torch.device("cuda"))

def set_cuda_device(device_id: int):
    torch.cuda.set_device(device_id)
    torch.cuda.synchronize(device_id)

def set_cuda_device_maybe(device_id: Optional[int] = None):
    if device_id is not None:
        if is_cuda_available():
            set_cuda_device(device_id)
        else:
            raise ValueError(f"CUDA device {device_id} is not available")

def get_cuda_device_count() -> int:
    return torch.cuda.device_count()

def get_current_cuda_device():
    return torch.cuda.current_device()

def cuda_synchronize(device_id=None):
    if device_id is not None:
        torch.cuda.synchronize(device_id)
    else:
        torch.cuda.synchronize()

def enable_tf32():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        print("torch.backends.cuda.matmul.allow_tf32 = True failed")
    try:
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        print("torch.backends.cudnn.allow_tf32 = True failed")
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        print("torch.set_float32_matmul_precision(high) failed")

def load_ambernet_model(cuda_device: Optional[int] = None):
    set_cuda_device_maybe(cuda_device)
    model = EncDecSpeakerLabelModel.from_pretrained(
        "langid_ambernet", refresh_cache=False
    )
    model.eval()
    model = move_to_gpu_maybe(model)
    return model

# horseshit unreadable GPU code
def predict_language_ambernet(model, audio_chunk):
    audio_chunk = move_to_gpu_maybe(audio_chunk)

    with torch.no_grad():
        input_signal_length = torch.tensor([audio_chunk.shape[1]], dtype=torch.long)
        input_signal_length = move_to_gpu_maybe(input_signal_length)

        logits = model(
            input_signal=audio_chunk, input_signal_length=input_signal_length
        )
        del input_signal_length

        pred_label_idx = logits[0].argmax(dim=-1).cpu().item()
        del logits

        if pred_label_idx == 22:
            return "es"
        return "en"

def strolid_model_name_to_nemo_model_name(model_name: str) -> str | None:
    if model_name == "en_real_quick_0": 
        return "nvidia/parakeet-tdt_ctc-110m"
    return None

def load_nemo_model_maybe_refresh(model_name: str):
    try:
        model = ASRModel.from_pretrained(model_name=model_name, refresh_cache=False)
    except Exception as e:
        pass
    if model is None:
        model = ASRModel.from_pretrained(model_name=model_name)
    return model

def load_transcription_model(model_name: str, cuda_device: Optional[int] = None):
    set_cuda_device_maybe(cuda_device)
    nemo_model_name = strolid_model_name_to_nemo_model_name(model_name)
    if nemo_model_name is None:
        raise ValueError(f"Invalid model name: {model_name}")
    model = load_nemo_model_maybe_refresh(nemo_model_name)
    model.eval()
    return move_to_gpu_maybe(model)

def transcribe(model, audio):
    if audio.ndim > 1:
        audio = numpy.mean(audio, axis=0)

    audio = audio.astype(numpy.float32, copy=False)
    audio = move_to_gpu_maybe(audio)
    with torch.no_grad():
        result = model.transcribe(
            audio=[audio],
            batch_size=1,
            logprobs=False,
        )

    transcript = result[0] if isinstance(result, (list, tuple)) else str(result)
    return transcript.strip()
