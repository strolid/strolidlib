# strolidlib

GPU utilities and helpers for Strolid services.

## Installation

```bash
cd local-repos/strolidlib
poetry install
```

## Usage

```python
from strolidlib import is_cuda_available, move_to_gpu_maybe
from strolidlib.gpu import load_transcription_model, transcribe
from strolidlib.utils import seconds_to_ydhms

# Check CUDA availability
if is_cuda_available():
    print("CUDA is available!")

# Load a model and transcribe
model = load_transcription_model("en_real_quick_0")
result = transcribe(model, audio_tensor)
```

## Development

```bash
# Install with dev dependencies
poetry install

# Run tests
poetry run pytest
```

