"""
wget https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/6_speakers.wav -O audio.wav
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg@7/lib:$DYLD_LIBRARY_PATH" 
export HUGGINGFACE_TOKEN="hf_xxx" 
uv run run_pytorch/main.py
"""
from pyannote.audio import Pipeline
import os

# https://huggingface.co/settings/tokens
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1", 
    token=os.getenv("HUGGINGFACE_TOKEN")
) 

# Specify the number of speakers
output = pipeline("audio.wav", min_speakers=6, max_speakers=6)

# print the predicted speaker diarization 
for turn, speaker in output.speaker_diarization:
    print(f"{speaker} speaks between t={turn.start:.3f}s and t={turn.end:.3f}s")
    
# Print summary
speakers = set()
for turn, speaker in output.speaker_diarization:
    speakers.add(speaker)
print(f"\nTotal unique speakers detected: {len(speakers)}")
