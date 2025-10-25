"""
wget https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/6_speakers.wav -O ./web/audio.wav
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg@7/lib:$DYLD_LIBRARY_PATH" 
export HUGGINGFACE_TOKEN="hf_xxx" 
uv run run_pytorch/main.py ./web/audio.wav ./web/diarization.json
"""
from pyannote.audio import Pipeline
import argparse
import os
import json
import webbrowser
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("audio_path", type=str)
parser.add_argument("diarization_path", type=str)
args = parser.parse_args()

# https://huggingface.co/settings/tokens
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1", 
    token=os.getenv("HUGGINGFACE_TOKEN")
) 

# Specify the number of speakers
output = pipeline(args.audio_path)

# Prepare diarization data
diarization_data = []

# print the predicted speaker diarization 
for turn, speaker in output.speaker_diarization:
    print(f"{speaker} speaks between t={turn.start:.3f}s and t={turn.end:.3f}s")
    diarization_data.append({
        "start": float(turn.start),
        "end": float(turn.end),
        "speaker": str(speaker)
    })
    
# Print summary
speakers = set()
for turn, speaker in output.speaker_diarization:
    speakers.add(speaker)
print(f"\nTotal unique speakers detected: {len(speakers)}")

# Save to JSON file
with open(args.diarization_path, "w") as f:
    json.dump(diarization_data, f, indent=2)
    
print(f"Diarization data saved to {args.diarization_path}")
print('Open the web browser with live server in web folder')