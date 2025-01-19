# giku_vol20_2025
# 内容
pythonを使ったゲーム
# 作成者
miry41
socha2510
齊藤陽向
syarasyoujyu
# 作動方法
1.python 3.10.12の仮想環境を作成し、入る
2.pip install -r requirements.txtをコマンドに入力
3.PulseAudio をインストール
4.pactl list short sourcesでデバイスが正しく認識されているか確認
5.pactl set-default-source <デバイス名>でオーディオデバイスを設定
6.pyxel run pyxel_examples/10_platformer.pyによってゲームを作動
# ゲーム動作
space:ジャンプ
r:録音(max 5秒)
右キー:右移動
左キー：左移動
#　デプロイ作業中
