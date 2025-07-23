# ベースイメージ
FROM python:3.12-slim

# 作業ディレクトリ
WORKDIR /app

# ライブラリ追加（← これが必要）
RUN apt-get update && apt-get install -y libgomp1

# 必要ファイルをコピー
COPY . /app

# 必要なPythonパッケージをインストール（pipアップグレード込み）
#RUN pip install --no-cache-dir flask numpy pandas lightgbm scikit-learn
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

# Flaskアプリの起動
CMD ["python", "predict.py"]
