import whisper
import os

import os
import subprocess

import whisper 


wmodel = whisper.load_model("small")
audio_dir="pyxel_examples/mp4_sample"
audio_inputs = os.listdir(audio_dir)
for audio_input in audio_inputs:
  audio = whisper.load_audio(os.path.join(audio_dir,audio_input))
  audio = whisper.pad_or_trim(audio)
  mel = whisper.log_mel_spectrogram(audio).to(wmodel.device)
  result = whisper.decode(wmodel, mel)
  print(result.text)