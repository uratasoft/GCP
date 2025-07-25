import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import lightgbm as lgb
import shap
import pickle
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="shap")

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
    cat_maps[col] = dict(zip(df[col].cat.categories, range(len(df[col].cat.categories))))
    df[col] = df[col].cat.codes

with open("cat_maps.pkl", "wb") as f:
    pickle.dump(cat_maps, f)

y = df["DelinquencyInfo"].fillna(0)
X = df.drop(["DelinquencyInfo", "No"], axis=1)
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 日本語変換マップ
feature_jp_map = {
    "Sex": "性別", "Marital": "婚姻状況", "Age": "年齢", "Income": "年収", "JobType": "職種区分",
    "Occupation": "職業", "BorrowingRatio": "借入比率", "CreditAppAmount": "申込額", "OtherDebts": "他債務",
    "DebtRestruct": "債務整理", "Industry": "業種", "Education": "学歴", "Dependents": "扶養人数",
    "OwnHouse": "持ち家", "Foreigner": "外国人", "Phone": "電話有無", "EmploymentYears": "勤続年数",
    "Guarantor": "保証人有無", "Collateral": "担保有無",
}

# 重み補正
scale_pos_weight = len(df[df["DelinquencyInfo"] == 0]) / max(len(df[df["DelinquencyInfo"] == 1]), 1)
print(f"scale_pos_weight = {scale_pos_weight:.4f}")

# モデル構築
model = lgb.LGBMClassifier(
    objective="binary", n_estimators=500, learning_rate=0.01, random_state=42,
    scale_pos_weight=scale_pos_weight, max_depth=3, feature_fraction=0.8,
    reg_alpha=1.0, reg_lambda=1.0, min_child_samples=5, subsample=0.7,
    colsample_bytree=0.7
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
print(classification_report(y_test, y_pred, digits=4))

# 出力保存
results = X_test.copy()
results["PredictedProbability"] = y_pred_proba
results["Actual"] = y_test.values
results.to_csv("individual_scores.csv", index=False, encoding="utf-8-sig")

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✅ モデルとスコア出力が完了しました。individual_scores.csv に保存しました。")
