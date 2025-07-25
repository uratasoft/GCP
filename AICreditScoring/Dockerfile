# ベースイメージ
FROM python:3.12-slim

# 作業ディレクトリ
WORKDIR /app

# 依存ライブラリに必要なパッケージをインストール（LightGBMなど用）
RUN apt-get update && apt-get install -y libgomp1

# アプリケーションコードとモデル・データをコピー
COPY . /app

# 依存パッケージのインストール（requirements.txt はプロジェクトルートにある前提）
RUN pip install --no-cache-dir -r requirements.txt

# Flaskで使うポート
EXPOSE 8080

# アプリ起動（main.pyがFlaskを呼び出す構成）
CMD ["python", "main.py"]
