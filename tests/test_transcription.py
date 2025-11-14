import json
from pathlib import Path

import vcon
from strolidlib.gpu import (
    load_tensor_from_vcon,
    preprocess_tensor,
    load_transcription_model,
    transcribe,
)

EXAMPLE_VCONS_DIR = Path(__file__).parent.parent.parent.parent / "tests" / "example-vcons"


def test_transcription():
    # Load a vcon from the example-vcons directory
    vcon_file = EXAMPLE_VCONS_DIR / "05cec3d4-0561-4010-9fb8-23cc7edef146.vcon.json"
    with open(vcon_file) as f:
        vcon_dict = json.load(f)
    vcon_obj = vcon.Vcon(vcon_dict)
    
    # Load tensor from vcon
    tensor, sample_rate = load_tensor_from_vcon(vcon_obj)
    
    # Preprocess tensor
    preprocessed = preprocess_tensor(tensor, sample_rate, 16000)
    
    # Load transcription model
    model = load_transcription_model("en_real_quick_0")
    
    # Convert to numpy for transcribe
    audio_np = preprocessed.detach().cpu().numpy()
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=0)
    
    # Transcribe
    result = transcribe(model, audio_np)
    
    assert result is not None

