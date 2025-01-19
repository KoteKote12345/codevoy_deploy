from flask import Flask, render_template
import pyxel
from sentence_transformers import SentenceTransformer
# 既存のゲームコードをインポート
from pyxel_examples.platformer import *

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# ゲーム実行用のエンドポイント
@app.route('/game')
def game():
    # ゲーム起動コード
    pyxel.run(App)  # Appは既存のゲームクラス
    return ''

if __name__ == '__main__':
    app.run()