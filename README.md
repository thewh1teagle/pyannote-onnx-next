# Next revolution of pyannote

## Setup

### Install FFmpeg 7 (required for torchcodec)

```console
brew install ffmpeg@7
```

Note: The default `ffmpeg` package (version 8) is not compatible with torchcodec. You need FFmpeg 7.

## Usage (PyTorch)

Set the library path to use FFmpeg 7:

```console
DYLD_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg@7/lib:$DYLD_LIBRARY_PATH" HUGGINGFACE_TOKEN="your_token" uv run run_pytorch/main.py
```

Or use the wrapper script:

```console
chmod +x run_pytorch/run.sh
HUGGINGFACE_TOKEN="your_token" ./run_pytorch/run.sh
```

## Usage (ONNX)

```console
uv run run_onnx/main.py
```