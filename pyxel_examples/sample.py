import whisper
import os

import os
import subprocess
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import whisper 


wmodel = whisper.load_model("small")

def get_text(audio_path:str)->str:
  """audioのパスを受け取り、その音声ファイルのテキストを返す"""
  audio = whisper.load_audio(audio_path)
  audio = whisper.pad_or_trim(audio)
  mel = whisper.log_mel_spectrogram(audio).to(wmodel.device)
  result = whisper.decode(wmodel, mel)
  return result.text

def closest_direction(input_text):
    """
    入力テキストに最も近い方向を判定する関数。
    
    Args:
        input_text (str): 判定したいテキスト。
    
    Returns:
        str: 最も近い方向（"上", "下", "停止", "左", "右" のいずれか）。
    """
    # 方向候補とスコアリング用の辞書
    directions = {
        "上": ["上", "上がる", "上昇", "登る"],
        "下": ["下", "下がる", "下降", "落ちる"],
        "停止": ["停止", "止まる", "ストップ", "中断"],
        "左": ["左", "左側", "左に曲がる", "左寄り"],
        "右": ["右", "右側", "右に曲がる", "右寄り"]
    }
    
    # スコアリング用の辞書
    scores = {direction: 0 for direction in directions}
    
    # 入力テキストをチェックしてスコアを計算
    for direction, keywords in directions.items():
        for keyword in keywords:
            if keyword in input_text:
                scores[direction] += 1  # 部分一致でスコアを加算
    
    # 最もスコアが高い方向を返す
    return max(scores, key=scores.get)

# テスト例
if __name__ == "__main__":
  audio_dir="pyxel_examples/mp4_sample"
  audio_file_names = os.listdir(audio_dir)
  for audio_file_name in audio_file_names:
    test_text = get_text(os.path.join(audio_dir,audio_file_name))
    result = closest_direction(test_text)
    print(f"入力テキスト: '{test_text}' -> 最も近い方向: '{result}'")
