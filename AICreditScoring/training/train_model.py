import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import lightgbm as lgb
import shap
import pickle

# CSVファイル
csv_filename = "./data/credit_data_learn.csv"
df = pd.read_csv(csv_filename, encoding="utf-8-sig")
df.columns = df.columns.str.replace(r"[\s\t\r\n\uFEFF]", "", regex=True)
df.replace(r"^\s*$", np.nan, regex=True, inplace=True)

# カテゴリ変数エンコード
cat_cols = ["Sex", "Marital", "Occupation", "Industry"]
cat_maps = {}
for col in cat_cols:
    df[col] = df[col].astype("category")
    cat_maps[col] = dict(
        zip(df[col].cat.categories, range(len(df[col].cat.categories)))
    )
    df[col] = df[col].cat.codes

with open("cat_maps.pkl", "wb") as f:
    pickle.dump(cat_maps, f)

y = df["DelinquencyInfo"].fillna(0)
X = df.drop(["DelinquencyInfo", "No"], axis=1)
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 日本語変換マップ
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

# 重み補正
#Recall 重視 ⇒ 上げる（25, 30...）      陽性判定／実際の陽性
#Precision 重視 ⇒ 下げる（15, 10...）   実際陽性／陽性判定
class_weight = {0: 1.0, 1: 6}

"""
パラメータ	       厳しい設定（過学習防止）  緩い設定（表現力強化）	   備考
learning_rate	    0.001～0.005	        0.01～0.05	       小さいほど学習は慎重に、過学習しにくい
n_estimators	    500～800	            100～300	       小さければ単純、大きければ複雑なモデル
max_depth	        3～4	                5～10	           深いと複雑（過学習リスク）、浅いと汎化しやすい
min_child_samples	5～8	                10～30	           小さいほど柔軟、大きいほど保守的（データ不足で分割しにくい）
reg_alpha (L1)	    0.5～2	                0～0.1	           大きいと特徴量の選択が厳しくなる
reg_lambda (L2)	    0.5～2	                0～0.1	           大きいと滑らかなモデル（過学習抑制）
feature_fraction	0.6～0.8	            0.9～1.0	       少ないとランダム性UP、過学習防止に寄与
colsample_bytree	0.6～0.8	            0.9～1.0	       木あたり使用特徴量を減らすと汎化性能UP
subsample	        0.7～0.9	            1.0	               100%使うと過学習リスクあり。0.8が一般的
class_weight（比率）1:5～1:10                1:1～1:3	        重みを強くするとRecallが向上、Precisionが下がる傾向
"""

# モデル構築
model = lgb.LGBMClassifier(
    objective="binary",  # 二値分類（例：延滞あり／なし）
    random_state=42,  # 乱数シード（再現性確保）
    verbosity=-1,  # 学習中の出力を抑制（ログ非表示）

    # モデル表現力と安定性
    n_estimators=700,  # 学習する決定木の数（多いほど複雑になる）
    learning_rate=0.0055,  # 学習率（小さいほど学習は遅いが安定）

    # 延滞の重み付け
    #scale_pos_weight=scale_pos_weight,  # クラス不均衡対策（少数派に重み付け）
    class_weight=class_weight,
    feature_fraction=0.8,  # 各木で使用する特徴量の割合（ランダム性）

    # 正則化の微調整
    reg_alpha=1,  # L1正則化（特徴量選択効果あり）
    reg_lambda=1,  # L2正則化（重みを滑らかにする）

    #過学習とのバランス
    max_depth=4,  # 木の最大深さ（過学習防止）
    min_child_samples=8,  # 葉に必要な最小データ数（過学習防止）
    subsample=1,  # データのサブサンプリング率（過学習防止）
    colsample_bytree=0.7,  # 各木の構築時に使う列の割合（過学習防止）
)

model.fit(X_train, y_train)

# 推論・評価
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# SHAP値取得
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap_vals_individual = shap_values[1] if isinstance(shap_values, list) else shap_values

# 個別 SHAP とスコア出力
for i in range(len(X_test)):
    print(f"\n■■ 個人 No.{i+1} の予測スコア: {y_pred_proba[i]:.4f}")
    shap_series = pd.Series(shap_vals_individual[i], index=X_test.columns)
    shap_top3 = shap_series.abs().sort_values(ascending=False).head(3)

    print("▼ 個人別 SHAP 上位3項目 ▼")
    for feat in shap_top3.index:
        val = shap_series[feat]
        sign = "プラス要因" if val > 0 else "マイナス要因"
        jp = feature_jp_map.get(feat, feat)
        print(f"{jp}: SHAP = {val:.4f}（{sign}）")

# 全体評価
print("\nモデル全体の評価結果")
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ AUC: {auc:.4f}")
print("\n分類レポート:")
print(classification_report(y_test, y_pred, digits=4, zero_division=0))

# 出力保存
results = X_test.copy()
results["PredictedProbability"] = y_pred_proba
results["Actual"] = y_test.values
results.to_csv("individual_scores.csv", index=False, encoding="utf-8-sig")

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✅ モデルとスコア出力が完了しました。individual_scores.csv に保存しました。")
