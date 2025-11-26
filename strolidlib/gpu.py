"""GPU helper utilities shared across Conserver components."""

from __future__ import annotations

import base64
import io
from typing import Optional

import vcon
from vcon import Vcon
import requests
from strolidlib.utils import get_first_dialog, is_valid_url, is_int
import torch
import torchaudio
from torch import Tensor
import numpy
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.collections.asr.models import ASRModel

def is_cuda_available():
    return torch.cuda.is_available()

def is_tensor(obj) -> bool:
    return isinstance(obj, torch.Tensor)

def is_numpy(obj) -> bool:
    return isinstance(obj, numpy.ndarray)

def is_pyannote_pipeline(obj) -> bool:
    return hasattr(obj, "__class__") and "pyannote" in str(type(obj))

def numpy_to_tensor(array: "numpy.ndarray") -> "torch.Tensor":
    return torch.from_numpy(array)

def move_to_gpu_maybe(obj):
    if not is_cuda_available():
        return obj

    if is_tensor(obj):
        on_gpu = obj.to(device="cuda", non_blocking=True)
        on_gpu.detach()
        return on_gpu

    if is_numpy(obj):
        obj = numpy_to_tensor(obj)
        return move_to_gpu_maybe(obj)

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

def is_mono(audio_data: Tensor) -> bool:
    return audio_data.shape[0] == 1

def convert_to_mono(audio_data: Tensor):
    audio_data = audio_data.mean(dim=0, keepdim=True)
    return audio_data

def ensure_mono(audio_data: Tensor) -> Tensor:
    if not is_mono(audio_data):
        audio_data = convert_to_mono(audio_data)
    return audio_data

def resample_tensor(audio_data: Tensor, tensor_sample_rate: int, target_sample_rate: int) -> Tensor:
    resampler = torchaudio.transforms.Resample(orig_freq=tensor_sample_rate, new_freq=target_sample_rate)
    return resampler(audio_data)

def resample_tensor_maybe(audio_data: Tensor, tensor_sample_rate: int, target_sample_rate: int) -> Tensor:
    if tensor_sample_rate != target_sample_rate:
        return resample_tensor(audio_data, tensor_sample_rate, target_sample_rate)
    return audio_data

def tensor_to_contiguous(audio_data: Tensor) -> Tensor:
    return audio_data.contiguous()

def load_ambernet_model(cuda_device: Optional[int] = None):
    set_cuda_device_maybe(cuda_device)
    model = EncDecSpeakerLabelModel.from_pretrained(
        "langid_ambernet", refresh_cache=False
    )
    model.eval()
    model = move_to_gpu_maybe(model)
    return model

def predict_language_ambernet(audio_chunk, model):
    """Predict language using AmberNet model.
    
    Args:
        audio_chunk: 1D numpy array (time,) or 2D tensor (batch, time)
        model: NeMo EncDecSpeakerLabelModel
    
    Returns:
        "en" or "es"
    """
    # Convert numpy to tensor if needed
    if is_numpy(audio_chunk):
        audio_chunk = numpy_to_tensor(audio_chunk)
    
    # Ensure 2D shape (batch, time) - NeMo speaker models expect this
    if audio_chunk.dim() == 1:
        audio_chunk = audio_chunk.unsqueeze(0)  # (time,) -> (1, time)
    
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
        return ASRModel.from_pretrained(model_name=model_name, refresh_cache=False)
    except Exception as e:
        pass
    return ASRModel.from_pretrained(model_name=model_name)

def load_transcription_model(model_name: str, cuda_device: Optional[int] = None):
    set_cuda_device_maybe(cuda_device)
    nemo_model_name = strolid_model_name_to_nemo_model_name(model_name)
    if nemo_model_name is None:
        raise ValueError(f"Invalid model name: {model_name}")
    model = load_nemo_model_maybe_refresh(nemo_model_name)
    model.eval()
    return move_to_gpu_maybe(model)

def load_langid_model(model_name: str, cuda_device: Optional[int] = None):
    return load_ambernet_model(cuda_device)

def load_tensor_from_url(url: str):
    response = requests.get(url)
    audio_buffer = io.BytesIO(response.content)
    return torchaudio.load(audio_buffer)

def load_tensor_from_b64(body: str):
    audio_bytes = base64.b64decode(body)
    audio_buffer = io.BytesIO(audio_bytes)
    return torchaudio.load(audio_buffer)

def dialog_to_tensor(dialog):
    if "url" in dialog:
        return load_tensor_from_url(dialog["url"])
    
    if "body" in dialog:
        if is_valid_url(dialog["body"]):
            return load_tensor_from_url(dialog["body"])
        return load_tensor_from_b64(dialog["body"])

def move_to_cpu_maybe(tensor: Tensor) -> Tensor:
    return tensor.cpu()

def move_to_device_maybe(tensor: Tensor, to_device: int | str = 0) -> Tensor:
    if to_device == "cpu":
        move_to_cpu_maybe(tensor)
    if is_int(to_device):
        set_cuda_device(to_device)
        return move_to_gpu_maybe(tensor)
    return tensor

def preprocess_tensor(tensor: Tensor, 
                      input_sample_rate: int = 16000,
                      output_sample_rate: int = 16000,
                      to_device: int | str = 0):
    """Preprocess audio tensor for NeMo ASR models.
    
    NeMo's transcribe() expects numpy arrays with shape (time,) - 1D audio signal.
    Input tensor from torchaudio.load() has shape (channels, time).
    """
    tensor = ensure_mono(tensor)
    tensor = resample_tensor_maybe(tensor, input_sample_rate, output_sample_rate)
    tensor = tensor.detach()
    tensor = tensor.cpu()
    # Squeeze to 1D: (1, time) -> (time,) as NeMo expects
    audio_np = tensor.squeeze(0).numpy()
    return audio_np

def load_tensor_from_vcon(vcon: Vcon):
    """Load audio tensor from a vCon dialog.
    
    Returns a tensor and sample rate 
    """
    dialog = get_first_dialog(vcon)
    tensor, sample_rate = dialog_to_tensor(dialog)
    return tensor, sample_rate

def load_and_preprocess_tensor_from_vcon(vcon: Vcon, output_sample_rate: int = 16000, to_device: int | str = 0):
    tensor, input_sample_rate = load_tensor_from_vcon(vcon)
    tensor = preprocess_tensor(tensor, input_sample_rate, output_sample_rate, to_device)
    return tensor

def transcribe_many(model, audio_list: list[Tensor]) -> list[str]:
    with torch.no_grad():
        hypotheses = model.transcribe(audio=audio_list)
    return [hyp.text for hyp in hypotheses]

def transcribe(model, audio: Tensor) -> str:
    return transcribe_many(model, [audio])[0]

