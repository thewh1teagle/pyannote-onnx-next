"""
wget https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/6_speakers.wav -O audio.wav
uv run run_pytorch/main.py
"""
from pyannote.audio import Pipeline

# https://huggingface.co/settings/tokens
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1", 
    token="{huggingface-token}"
) 

output = pipeline("audio.wav")

# print the predicted speaker diarization 
for turn, speaker in output.speaker_diarization:
    print(f"{speaker} speaks between t={turn.start:.3f}s and t={turn.end:.3f}s")
