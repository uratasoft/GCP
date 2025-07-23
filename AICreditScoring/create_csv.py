import pandas as pd
import random
import math

# モデル学習用にデータ1000件を作成する様に設定
num_records = 1000

# マスタ定義
sexes = ["男", "女"]
maritals = ["未", "既"]

### ---------- エンコーディング ---------------
# 職種と業種の対応表
jobtype_occupation_map = {
    "1": ["市役所職員", "高校教員"],
    "2": ["製造オペレーター", "販売スタッフ", "接客スタッフ"],
    "3": ["プログラマ", "介護スタッフ", "店舗販売員"],
    "4": ["ホールスタッフ"],
    "5": ["無職"],
}

occupation_industry_map = {
    "市役所職員": "行政",
    "高校教員": "教育",
    "製造オペレーター": "製造",
    "販売スタッフ": "小売り",
    "接客スタッフ": "サービス",
    "プログラマ": "IT",
    "介護スタッフ": "サービス",
    "店舗販売員": "小売り",
    "清掃員": "サービス",
    "ホールスタッフ": "飲食",
    "無職": "なし",
}

### ---------- スケーリング ---------------
# 職業ごとの年収レンジ（万単位）
occupation_income_range = {
    "市役所職員": (400, 700),
    "高校教員": (400, 700),
    "製造オペレーター": (250, 500),
    "販売スタッフ": (250, 500),
    "接客スタッフ": (250, 500),
    "プログラマ": (400, 800),
    "介護スタッフ": (250, 500),
    "店舗販売員": (250, 500),
    "清掃員": (180, 250),
    "ホールスタッフ": (200, 400),
    "無職": (0, 100),
}

educations = ["1", "2", "3", "4"]

data = []

for i in range(1, num_records + 1):
    sex = random.choice(sexes)
    marital = random.choices(maritals, weights=[0.7, 0.3])[0]
    age = random.randint(25, 65)

### ---------- 次元削除 ------------------
    # 無職と清掃員を減らす
    job_type = random.choices(["1", "2", "3", "4", "5"], weights=[20, 30, 30, 18, 1])[0]

    if job_type == "4":
        occupation = random.choices(["清掃員", "ホールスタッフ"], weights=[2, 98])[0]
    else:
        occupation = random.choice(jobtype_occupation_map[job_type])

    industry = occupation_industry_map[occupation]

    # 収入
    min_income, max_income = occupation_income_range[occupation]
    if max_income == 0:
        income = 0
    else:
        income = random.randint(min_income, max_income) * 10000

### ---------- スケーリング ---------------
    # OtherDebtsを50%の確率で 0 にする
    if random.random() < 0.5:
        other_debts = 0
    else:
        if income > 0:
            other_debts = random.randint(int(income * 0.05), int(income * 0.5))
        else:
            other_debts = random.randint(100000, 500000)
        other_debts = math.floor(other_debts / 10000) * 10000

    borrowing_ratio = round(other_debts / income, 2) if income > 0 else 0.0

    # CreditAppAmount は必ず設定する
    credit_amount = random.randint(30, 500) * 10000

    # ------------------------
    # 属性別確率モデルここ！
    # ------------------------
    delinquency_prob = 0.01

    if occupation == "無職":
        delinquency_prob = 0.5
    elif income < 3000000 and income > 0:
        delinquency_prob = 0.1
    elif borrowing_ratio > 0.4:
        delinquency_prob = 0.15
    elif age < 30 and income < 3000000:
        delinquency_prob = 0.2

    # 債務整理があればさらに上げる
    debt_restruct = random.choices([0, 1], weights=[97, 3])[0]
    if debt_restruct == 1:
        delinquency_prob = 0.9

    # 確率で付与
    delinquency_info = 1 if random.random() < delinquency_prob else 0

    """
    # DelinquencyInfo が 1 の場合だけ、属性を高リスク寄りに変更
    if delinquency_info == 1:
        age = random.randint(50, 65)
        income = random.randint(200, 300) * 10000
        borrowing_ratio = other_debts / income
        other_debts = random.randint(int(income * 0.5), int(income * 0.8))
        other_debts = math.floor(other_debts / 10000) * 10000
        debt_restruct = 1
    """

    education = random.choice(educations)
    dependents = random.randint(0, 3)
    own_house = random.choice([0, 1])
    foreigner = random.choices([0, 1], weights=[95, 5])[0]
    phone = random.choice([0, 1])
    employment_years = random.randint(0, max(age - 18, 0))
    guarantor = random.choice([0, 1])
    collateral = random.choice([0, 1])

    record = {
        "No": i,
        "Sex": sex,
        "Marital": marital,
        "Age": age,
        "Income": income,
        "JobType": job_type,
        "Occupation": occupation,
        "BorrowingRatio": borrowing_ratio,
        "CreditAppAmount": credit_amount,
        "OtherDebts": other_debts,
        "DelinquencyInfo": delinquency_info,
        "DebtRestruct": debt_restruct,
        "Industry": industry,
        "Education": education,
        "Dependents": dependents,
        "OwnHouse": own_house,
        "Foreigner": foreigner,
        "Phone": phone,
        "EmploymentYears": employment_years,
        "Guarantor": guarantor,
        "Collateral": collateral,
    }

    data.append(record)

df = pd.DataFrame(data)

# 並び順固定
df = df[
    [
        "No",
        "Sex",
        "Marital",
        "Age",
        "Income",
        "JobType",
        "Occupation",
        "BorrowingRatio",
        "CreditAppAmount",
        "OtherDebts",
        "DelinquencyInfo",
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
]

df.to_csv("credit_data_learn.csv", index=False, encoding="utf-8-sig")

print("✅ credit_data_learn.csv を書き出しました！")
