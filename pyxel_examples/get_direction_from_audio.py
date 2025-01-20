import numpy as np
import os  # 必要に応じて残す
import time
import whisper # whisperをimport
# from transformers import RobertaTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
from difflib import SequenceMatcher

# デバイスをCPUに設定
device = torch.device('cpu')

# whisperモデルのロード
whisper_model = whisper.load_model("tiny") # モデルサイズを指定 (例: "tiny", "base", "small", "medium", "large")
whisper_model.to(device) # whisperモデルもdeviceに配置

# SentenceTransformerモデルのロード
model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)

# ターゲット単語
directions = ["上", "右","左"]

# ベクトル化
wmodel = whisper_model.load_model("small")

def get_text(audio_path:str)->str:
  """audioのパスを受け取り、その音声ファイルのテキストを返す"""
  audio = whisper_model.load_audio(audio_path)
  audio = whisper_model.pad_or_trim(audio)
  mel = whisper_model.log_mel_spectrogram(audio).to(wmodel.device)
  result =whisper_model.decode(wmodel, mel)
  return result.text


def closest_direction(input_text: str) -> str:
    """
    入力テキストに最も近い方向を判定する関数。
    """
    similarities = [SequenceMatcher(None, input_text, direction).ratio() for direction in directions]
    max_index = int(np.argmax(similarities))
    return directions[max_index]

# テスト例
if __name__ == "__main__":
  audio_dir="pyxel_examples/mp4_sample"
  audio_file_names = os.listdir(audio_dir)
  for audio_file_name in audio_file_names:
    time_st=time.time()
    test_text = get_text(os.path.join(audio_dir,audio_file_name))
    print(test_text)
    result = closest_direction(test_text)
    print(f"入力テキスト: '{test_text}' -> 最も近い方向: '{result}'")
    print(f"処理時間: {time.time()-time_st}")