from flask import Flask, render_template_string
import pandas as pd
import pickle
import numpy as np

# モデル読み込み
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# カテゴリマップ読み込み
with open("cat_maps.pkl", "rb") as f:
    cat_maps = pickle.load(f)

CSV_FILE = "credit_data_predict.csv"

app = Flask(__name__)

HTML_PAGE = """
<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <title>与信スコア 一覧表示</title>
  <style>
    table, th, td {
      border: 1px solid black;
      border-collapse: collapse;
      padding: 5px;
      text-align: center;
    }
    .risk-high {
      background-color: #ffcccc;
      font-weight: bold;
    }
  </style>
</head>
<body>
  <h1>与信スコア 一覧表示</h1>
  {{ table|safe }}
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    try:
### ---------- １．データ取得 -------------------
        df = pd.read_csv(CSV_FILE, encoding="utf-8-sig")

        # 正方向のカテゴリ変換
        for col, mapping in cat_maps.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(0)

        # DelinquencyInfo を一旦退避
        delinquency_info_col = None
        if "DelinquencyInfo" in df.columns:
            delinquency_info_col = df["DelinquencyInfo"].copy()
            df = df.drop(columns=["DelinquencyInfo"])

        feature_cols = [
            "Sex",
            "Marital",
            "Age",
            "Income",
            "JobType",
            "Occupation",
            "BorrowingRatio",
            "CreditAppAmount",
            "OtherDebts",
            "DebtRestruct",
            "Industry",
            "Education",
            "Dependents",
            "OwnHouse",
            "Foreigner",
            "Phone",
            "EmploymentYears",
            "Guarantor",
            "Collateral",
        ]

        X = df[feature_cols].copy()

### ---------- ８．与信スコアリング ----------------------
        # スコア予測
        proba_list = model.predict_proba(X)[:, 1]
        df["Score"] = np.round(proba_list, 4)

        if delinquency_info_col is not None:
            df["DelinquencyInfo"] = delinquency_info_col

        # リバースマップ作成
        cat_maps_rev = {
            col: {v: k for k, v in mapping.items()} for col, mapping in cat_maps.items()
        }

        for col, mapping_rev in cat_maps_rev.items():
            if col in df.columns:
                df[col] = df[col].map(mapping_rev).fillna("")

        display_cols = [
            "No",
            "Sex",
            "Age",
            "Income",
            "BorrowingRatio",
            "Industry",
            "CreditAppAmount",
            "OtherDebts",
            "DebtRestruct",
            "DelinquencyInfo",
            "Score",
        ]

        # 延滞ハイリスクのスコアを赤で表示
        def highlight_risk(val):
            try:
                return 'class="risk-high"' if float(val) >= 0.7 else ""
            except:
                return ""

        # Web用に項目名を表示
        table_html = df[display_cols].to_html(
            index=False,
            escape=False,
            border=1,
            formatters={"Score": lambda x: f"<span {highlight_risk(x)}>{x}</span>"},
        )

        return render_template_string(HTML_PAGE, table=table_html)

    except Exception as e:
        return render_template_string(
            HTML_PAGE, table=f"<p style='color:red;'>エラー: {e}</p>"
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
