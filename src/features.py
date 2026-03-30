"""
SBL Lead Engine — Feature Engineering
"""
import pandas as pd
import numpy as np

HOT_TITLE_KEYWORDS = ["founder","ceo","co-founder","head of sales","vp sales","chief revenue","growth","cro"]
WARM_TITLE_KEYWORDS = ["manager","marketing","product","director","partnerships","business development","cmo"]
SALES_BIO_KEYWORDS = ["outbound","pipeline","revenue","lead gen","b2b","sales","cold outreach","gtm","scaling"]
ACTIVITY_HOT_KEYWORDS = ["outreach","sales","cold email","lead","b2b","revenue","pipeline"]
COMPANY_SIZE_ORDER = {"1-10":1,"11-50":2,"51-200":3,"201-500":4,"501-1000":5,"1000+":6}
HOT_INDUSTRIES = ["saas","e-commerce","b2b tech","sales tech","marketing tech","fintech"]
WARM_INDUSTRIES = ["consulting","digital marketing","real estate","insurance","retail tech"]

def title_score(title):
    t = title.lower()
    if any(k in t for k in HOT_TITLE_KEYWORDS): return 2
    if any(k in t for k in WARM_TITLE_KEYWORDS): return 1
    return 0

def bio_sales_score(bio):
    return sum(1 for k in SALES_BIO_KEYWORDS if k in bio.lower())

def activity_score(activity):
    a = activity.lower()
    if any(k in a for k in ACTIVITY_HOT_KEYWORDS): return 2
    if "no recent" in a or "no posts" in a: return 0
    return 1

def industry_score(industry):
    i = industry.lower()
    if i in HOT_INDUSTRIES: return 2
    if i in WARM_INDUSTRIES: return 1
    return 0

def engineer_features(df):
    df = df.copy()
    df["title_score"] = df["job_title"].apply(title_score)
    df["bio_sales_score"] = df["bio"].apply(bio_sales_score)
    df["activity_score"] = df["recent_activity"].apply(activity_score)
    df["industry_score"] = df["industry"].apply(industry_score)
    df["company_size_num"] = df["company_size"].apply(lambda x: COMPANY_SIZE_ORDER.get(x,3))
    df["connections_log"] = np.log1p(df["connections"])
    df["engagement_score"] = df["posts_per_month"] * df["activity_score"]
    df["seniority_score"] = df["title_score"] * df["years_experience"]
    df["profile_completeness"] = (
        df["has_website"] +
        (df["connections"]>200).astype(int) +
        (df["posts_per_month"]>2).astype(int) +
        (df["bio"].str.len()>50).astype(int)
    )
    return df

FEATURE_COLS = [
    "title_score","bio_sales_score","activity_score","industry_score",
    "company_size_num","connections_log","engagement_score","seniority_score",
    "profile_completeness","has_website","posts_per_month","years_experience"
]

def get_feature_matrix(df):
    df_feat = engineer_features(df)
    return df_feat[FEATURE_COLS].values, FEATURE_COLS

if __name__ == "__main__":
    import sys; sys.path.insert(0,".")
    from data.generate_data import generate_dataset
    df = generate_dataset(50)
    X, cols = get_feature_matrix(df)
    print("Feature matrix shape:", X.shape)
    print("Features:", cols)
