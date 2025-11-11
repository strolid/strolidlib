"""GPU helper utilities shared across Conserver components."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

try:  # pragma: no cover - optional dependency
    import torch  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover -- type checking only
    from torch import Tensor  # type: ignore
    from nemo.collections.asr.models import EncDecSpeakerLabelModel  # type: ignore[import]


def is_cuda_available() -> bool:
    """Return True when torch with CUDA support is available."""
    return bool(torch and torch.cuda.is_available())


def move_to_gpu_maybe(obj):
    """Move torch-backed objects to GPU when possible."""
    if torch is None:
        return obj

    if isinstance(obj, torch.Tensor):
        on_gpu = obj.to(device="cuda", non_blocking=True)
        on_gpu.detach()
        return on_gpu

    # pyannote Pipeline.to(...) does not accept non_blocking
    if hasattr(obj, "__class__") and "pyannote" in str(type(obj)):
        return obj.to(torch.device("cuda"))

    try:
        return obj.to(device="cuda", non_blocking=True)
    except TypeError:
        return obj.to(torch.device("cuda"))


def set_cuda_device(device_id: int) -> bool:
    if not is_cuda_available():
        return False

    torch.cuda.set_device(device_id)
    torch.cuda.synchronize(device_id)
    return True


def get_cuda_device_count() -> int:
    if is_cuda_available():
        return torch.cuda.device_count()
    return 0


def get_current_cuda_device() -> Optional[int]:
    if is_cuda_available():
        return torch.cuda.current_device()
    return None


def cuda_synchronize(device_id: Optional[int] = None) -> bool:
    if not is_cuda_available():
        return False

    if device_id is not None:
        torch.cuda.synchronize(device_id)
    else:
        torch.cuda.synchronize()
    return True


def enable_tf32() -> None:
    if torch is None:
        return

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


def _import_encdec_speaker_label_model():
    try:
        from nemo.collections.asr.models import EncDecSpeakerLabelModel  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise ImportError(
            "nemo_toolkit is required for load_ambernet_model(); "
            "install strolidlib with the 'langid' extra (pip install strolidlib[langid])."
        ) from exc
    return EncDecSpeakerLabelModel


def load_ambernet_model(cuda_device: Optional[int] = None):
    if cuda_device is not None:
        set_cuda_device(cuda_device)
    EncDecSpeakerLabelModel = _import_encdec_speaker_label_model()
    model = EncDecSpeakerLabelModel.from_pretrained(
        "langid_ambernet", refresh_cache=False
    )
    model.eval()
    model = move_to_gpu_maybe(model)
    return model


# horseshit unreadable GPU code
def predict_language_ambernet(model, audio_chunk):
    audio_chunk = move_to_gpu_maybe(audio_chunk)

    if torch is None:
        raise RuntimeError("torch is required to run predict_language_ambernet")

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
