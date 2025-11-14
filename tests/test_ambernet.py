import json
from pathlib import Path

import vcon
from strolidlib.gpu import (
    load_tensor_from_vcon,
    preprocess_tensor,
    load_ambernet_model,
    predict_language_ambernet,
)

EXAMPLE_VCONS_DIR = Path(__file__).parent.parent.parent.parent / "tests" / "example-vcons"


def test_ambernet():
    # Load a vcon from the example-vcons directory
    vcon_file = EXAMPLE_VCONS_DIR / "05cec3d4-0561-4010-9fb8-23cc7edef146.vcon.json"
    with open(vcon_file) as f:
        vcon_dict = json.load(f)
    vcon_obj = vcon.Vcon(vcon_dict)
    
    # Load tensor from vcon
    tensor, sample_rate = load_tensor_from_vcon(vcon_obj)
    
    # Preprocess tensor
    preprocessed = preprocess_tensor(tensor, sample_rate, 16000)
    
    # Load ambernet model
    model = load_ambernet_model()
    
    # Predict language
    language = predict_language_ambernet(preprocessed, model)
    
    assert language in ["en", "es"]

