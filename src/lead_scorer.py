"""
SBL Lead Engine — Component 1: Lead Scorer
Classifies leads as Hot / Warm / Cold
"""
import pandas as pd
import numpy as np
import joblib
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import get_feature_matrix, engineer_features, FEATURE_COLS
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

LABEL_MAP = {0:"Cold", 1:"Warm", 2:"Hot"}

def train_model(df, save_path="models/lead_scorer.pkl"):
    X, _ = get_feature_matrix(df)
    y = df["label_encoded"].values
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
    models = {
        "Random Forest": (RandomForestClassifier(n_estimators=200,max_depth=8,class_weight="balanced",random_state=42), False),
        "Gradient Boosting": (GradientBoostingClassifier(n_estimators=150,learning_rate=0.08,max_depth=4,random_state=42), False),
        "Logistic Regression": (LogisticRegression(C=1.0,max_iter=1000,class_weight="balanced",random_state=42), True)
    }
    best_acc, best_name, best_model, best_scaled = 0, "", None, False
    for name,(model,scaled) in models.items():
        Xtr = X_tr_s if scaled else X_train
        Xte = X_te_s if scaled else X_test
        model.fit(Xtr, y_train)
        acc = accuracy_score(y_test, model.predict(Xte))
        print(f"{name}: {acc:.3f}")
        if acc > best_acc:
            best_acc,best_name,best_model,best_scaled = acc,name,model,scaled
    print(f"\nBest: {best_name} ({best_acc:.3f})")
    Xte_final = X_te_s if best_scaled else X_test
    print(classification_report(y_test, best_model.predict(Xte_final), target_names=["Cold","Warm","Hot"]))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump({"model":best_model,"scaler":scaler,"scaled":best_scaled,"feature_cols":FEATURE_COLS}, save_path)
    print(f"Saved → {save_path}")

def score_lead(profile, model_path="models/lead_scorer.pkl"):
    art = joblib.load(model_path)
    model,scaler,scaled = art["model"],art["scaler"],art["scaled"]
    df = pd.DataFrame([profile])
    df_feat = engineer_features(df)
    for c in FEATURE_COLS:
        if c not in df_feat.columns: df_feat[c] = 0
    X = df_feat[FEATURE_COLS].values
    if scaled: X = scaler.transform(X)
    proba = model.predict_proba(X)[0]
    pred = int(np.argmax(proba))
    label = LABEL_MAP[pred]
    score = int(proba[2]*100*0.6 + proba[1]*100*0.3 + proba[0]*100*0.1)
    reasons = []
    row = df_feat.iloc[0]
    if row.get("title_score",0)==2: reasons.append("decision-maker title")
    if row.get("industry_score",0)==2: reasons.append("high-value industry")
    if row.get("bio_sales_score",0)>=2: reasons.append("sales keywords in bio")
    if row.get("activity_score",0)==2: reasons.append("active in sales content")
    if row.get("connections",0)>500: reasons.append("500+ connections")
    reasoning = ("Strong signals: " + ", ".join(reasons[:3])) if reasons else "Low outreach receptivity"
    return {
        "label": label,
        "score": score,
        "confidence": round(float(proba[pred])*100, 1),
        "probabilities": {"Hot":round(float(proba[2])*100,1),"Warm":round(float(proba[1])*100,1),"Cold":round(float(proba[0])*100,1)},
        "reasoning": reasoning
    }

if __name__ == "__main__":
    from data.generate_data import generate_dataset
    print("Training lead scorer...")
    df = generate_dataset(200)
    train_model(df)
    print("\nTest scoring a Founder/SaaS lead:")
    result = score_lead({
        "job_title":"Founder","industry":"SaaS","company_size":"11-50",
        "connections":1200,"bio":"Scaling B2B outbound. Cold outreach GTM.",
        "recent_activity":"Posted about cold outreach tips",
        "has_website":1,"posts_per_month":8,"years_experience":7
    })
    print(f"Label: {result['label']} | Score: {result['score']}/100 | Confidence: {result['confidence']}%")
    print(f"Reasoning: {result['reasoning']}")
