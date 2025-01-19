import whisper
import os
import numpy as np
import os
import subprocess
from transformers import RobertaTokenizer
import whisper 
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import time
# モデルをロード
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# ターゲット単語
directions = ["上", "下", "停止", "左", "右"]

# ベクトル化
direction_embeddings = model.encode(directions)
wmodel = whisper.load_model("small")

def get_text(audio_path:str)->str:
  """audioのパスを受け取り、その音声ファイルのテキストを返す"""
  audio = whisper.load_audio(audio_path)
  audio = whisper.pad_or_trim(audio)
  mel = whisper.log_mel_spectrogram(audio).to(wmodel.device)
  result = whisper.decode(wmodel, mel)
  return result.text

def get_similarity(encoding1,encoding2):
    """numpy配列のエンコーディングを受け取り、その類似度を返す"""
    return np.dot(encoding1, encoding2) / (np.linalg.norm(encoding1) * np.linalg.norm(encoding2))

def closest_direction(input_text:str)->str:
    """
    入力テキストに最も近い方向を判定する関数。
    
    Args:
        input_text (str): 判定したいテキスト。
    
    Returns:
        str: 最も近い方向（"上", "下", "停止", "左", "右" のいずれか）。
    """
    encoding_input_text = model.encode([input_text])
    similarities = cosine_similarity(encoding_input_text, direction_embeddings).flatten()
    most_similar_direction = directions[similarities.argmax()]
    return most_similar_direction

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