import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import lightgbm as lgb
import shap
import pickle
import re
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="shap")

# CSVファイル名
csv_filename = "credit_data_learn.csv"

### ---------- １．データ取得 -------------------
# CSV読み込み
df = pd.read_csv(csv_filename, header=0, encoding="utf-8-sig")

### ---------- ２．データクレンジング ------------
# カラム名クリーンナップ
df.columns = df.columns.str.replace(r"[\s\t\r\n\uFEFF]", "", regex=True)

# 空文字をNaNに置換
df.replace(r"^\s*$", np.nan, regex=True, inplace=True)

### ---------- ３．エンコーディング ---------------
# カテゴリ変数を数値化・マップ保存
cat_cols = ["Sex", "Marital", "Occupation", "Industry"]
cat_maps = {}
for col in cat_cols:
    df[col] = df[col].astype("category")
    cat_maps[col] = dict(
        zip(df[col].cat.categories, range(len(df[col].cat.categories)))
    )
    df[col] = df[col].cat.codes

# マップを保存
with open("cat_maps.pkl", "wb") as f:
    pickle.dump(cat_maps, f)

# ターゲット変数（目的変数）抽出
y = df["DelinquencyInfo"].fillna(0)

### ---------- ５．次元削除 -----------------------
# 説明変数の抽出
X = df.drop(["DelinquencyInfo", "No"], axis=1)

### ---------- ６．特徴量選択 ----------------------
# 説明変数の抽出
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

# データ分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 日本語変換辞書
feature_jp_map = {
    "Sex": "性別",
    "Marital": "婚姻状況",
    "Age": "年齢",
    "Income": "年収",
    "JobType": "職種区分",
    "Occupation": "職業",
    "BorrowingRatio": "借入比率",
    "CreditAppAmount": "申込額",
    "OtherDebts": "他債務",
    "DebtRestruct": "債務整理",
    "Industry": "業種",
    "Education": "学歴",
    "Dependents": "扶養人数",
    "OwnHouse": "持ち家",
    "Foreigner": "外国人",
    "Phone": "電話有無",
    "EmploymentYears": "勤続年数",
    "Guarantor": "保証人有無",
    "Collateral": "担保有無",
}

# 正例（延滞者）、負例（延滞なし）抽出
df_0 = df[df["DelinquencyInfo"] == 0]
df_1 = df[df["DelinquencyInfo"] == 1]

# 重み補正率計算（正例を増強する重みを比率で計算）
# 正例が少ないとモデルは延滞なしと認識しやすいので正例に重みを付けて補正する
scale_pos_weight = len(df_0) / len(df_1) if len(df_1) > 0 else 1.0
print(f"scale_pos_weight = {scale_pos_weight}")

# モデル作成
model = lgb.LGBMClassifier(
    objective="binary",   # 二値分類問題で使う
    n_estimators=500,     # 木を500本作る（多いほど精度上がるが過学習に注意）
    learning_rate=0.01,   # 学習率を小さくして少しずつ学習
    random_state=42,      # 乱数固定で再現性確保
    scale_pos_weight=scale_pos_weight, # クラス不均衡対策の重み
    max_depth=3,          # 決定木の最大深さ（過学習防止）
    feature_fraction=0.8, # 各決定木で使う特徴量の割合（過学習防止）
    reg_alpha=1.0,        # L1正則化（過学習防止）
    reg_lambda=1.0,       # L2正則化（過学習防止）
    min_child_samples=5,  # 最小サンプル数（過学習防止）
    subsample=0.7,        # データのサブサンプリング率（過学習防止）
    colsample_bytree=0.7, # 各木の構築時に使う特徴量の割合（過学習防止）
)

### ---------- ７．AIモデル学習 ----------------------
# 説明変数と目的変数でモデルを訓練
model.fit(X_train, y_train)

# 与信判定
# テストデータに対する予測ラベル（判定）を取得（0：融資 1：却下）
y_pred = model.predict(X_test)

# 正解率（Accuracy）を計算
accuracy = accuracy_score(y_test, y_pred)

# AUC（曲線下面積）（延滞を当てた確率）を計算
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

# テストデータにおける（1クラスの確率（延滞））の予測
y_pred_proba = model.predict_proba(X_test)[:, 1]

### ---------- ８．与信スコアリング ----------------------
# 個人ごとのスコア
for i in range(len(X_test)):
    score = y_pred_proba[i]
    print(f"\n■■ 個人 No.{i+1} の予測スコア: {score:.4f}")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap_vals_individual = (
        shap_values[1] if isinstance(shap_values, list) else shap_values
    )

    # 個人ごとの SHAP 値をシリーズ化（各特徴量の貢献度を並べる）
    shap_series = pd.Series(shap_vals_individual[i], index=X_test.columns)

    # 絶対値を取って、重要度の大小を比較できるようにする
    shap_series_abs = shap_series.abs()

    # 大きい順に並べ替え（重要度の高い特徴量を上にする
    shap_series_sorted = shap_series_abs.sort_values(ascending=False)

### ---------- ９．通知・レポート ----------------------
    # 上位3つを抽出
    top3 = shap_series_sorted.head(3)

# SHAP値TOP3表示
    print("▼ 個人別 SHAP 上位3項目 ▼")
    for feature, value in top3.items():
        sign_value = shap_series[feature]
        direction = "プラス要因" if sign_value > 0 else "マイナス要因"
        jp_feature = feature_jp_map.get(feature, feature)
        print(f"{jp_feature}: SHAP = {sign_value:.4f} （{direction}）")

# 全体評価
print("\nモデル全体の評価結果")
print(f"Accuracy（正解率）: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")
print("\n分類レポート:")
print(classification_report(y_test, y_pred, digits=4))

# 個人別スコアCSV出力
results = X_test.copy()
results["PredictedProbability"] = y_pred_proba
results["Actual"] = y_test.values
results.to_csv("individual_scores.csv", index=False, encoding="utf-8-sig")

# モデル保存
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✅ モデルとスコア出力が完了しました。individual_scores.csv に保存しました。")
