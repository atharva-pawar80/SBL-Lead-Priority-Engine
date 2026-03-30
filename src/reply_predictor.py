"""
SBL Lead Engine — Component 2: Reply Intent Predictor
Predicts reply probability for a message + profile combination
"""
import pandas as pd
import numpy as np
import joblib
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import get_feature_matrix, FEATURE_COLS, engineer_features
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

GOOD_OPENERS = ["noticed","saw your post","came across","your work","congrats","your recent"]
BAD_OPENERS = ["hi there","dear sir","to whom","hope this finds","i am writing","my name is"]
PERSONALISATION = ["your company","your role","your industry","your post","you recently"]

def message_features(msg):
    m = msg.lower()
    wc = len(msg.split())
    return {
        "word_count": wc,
        "has_question": int("?" in msg),
        "personalisation": sum(1 for s in PERSONALISATION if s in m),
        "good_opener": int(any(o in m for o in GOOD_OPENERS)),
        "bad_opener": int(any(o in m for o in BAD_OPENERS)),
        "has_cta": int(any(w in m for w in ["call","chat","15 min","quick call","connect","meeting"])),
        "has_value": int(any(w in m for w in ["help","improve","increase","save","reduce","grow"])),
        "optimal_length": int(50 <= wc <= 120)
    }

def generate_msg_dataset(df):
    good_templates = [
        "Hey, noticed your work on {topic} at {company} — impressive. We help {industry} teams automate outreach while keeping it human. 15 mins this week?",
        "Saw your recent post on {topic} — really resonated. Your {industry} background is exactly the space we work in. Open to a quick call?",
        "Your {industry} focus caught my eye. We've helped 200+ founders like you generate pipeline through AI-led outreach. Worth a 10-min chat?"
    ]
    bad_templates = [
        "Hi there, I am writing to introduce myself. We offer many services that might interest you. Please let me know.",
        "Dear Sir/Madam, hope this message finds you well. I wanted to reach out about a potential opportunity.",
        "My name is John and I work at XYZ. We have been in business 10 years. Hope to connect soon."
    ]
    medium_templates = [
        "Hi, I help companies with outbound automation. Would you be open to learning more?",
        "Quick question — are you currently doing LinkedIn outreach manually?",
        "Hey, saw your profile and thought you might find what we're building interesting."
    ]
    topics = {"SaaS":"SaaS growth","B2B Tech":"B2B sales","E-commerce":"e-commerce","Sales Tech":"sales automation","default":"business growth"}
    records = []
    for _, row in df.iterrows():
        base = row["reply_probability"]
        topic = topics.get(row["industry"], topics["default"])
        company = "your company"
        for t in good_templates:
            msg = t.format(topic=topic, company=company, industry=row["industry"])
            adj = min(1.0, base + np.random.uniform(0.05, 0.20))
            records.append({"message":msg, **row.to_dict(), "reply_prob":adj})
        for t in bad_templates:
            adj = max(0.0, base - np.random.uniform(0.10, 0.25))
            records.append({"message":t, **row.to_dict(), "reply_prob":adj})
        for t in medium_templates:
            msg = t.format(topic=topic, company=company, industry=row["industry"]) if "{" in t else t
            adj = max(0.0, min(1.0, base + np.random.uniform(-0.05,0.08)))
            records.append({"message":msg, **row.to_dict(), "reply_prob":adj})
    return pd.DataFrame(records)

def train_reply_predictor(df, save_path="models/reply_predictor.pkl"):
    print("Building message dataset...")
    mdf = generate_msg_dataset(df)
    tfidf = TfidfVectorizer(max_features=200, ngram_range=(1,2), stop_words="english")
    X_text = tfidf.fit_transform(mdf["message"]).toarray()
    X_msg = pd.DataFrame(list(mdf["message"].apply(message_features))).values
    X_profile, _ = get_feature_matrix(mdf)
    X = np.hstack([X_text, X_msg, X_profile])
    y = mdf["reply_prob"].values
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = GradientBoostingRegressor(n_estimators=150,learning_rate=0.08,max_depth=4,random_state=42)
    model.fit(X_train_s, y_train)
    preds = np.clip(model.predict(X_test_s), 0, 1)
    print(f"MAE: {mean_absolute_error(y_test,preds):.3f} | R²: {r2_score(y_test,preds):.3f}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump({"model":model,"tfidf":tfidf,"scaler":scaler}, save_path)
    print(f"Saved → {save_path}")

def predict_reply(profile, message, model_path="models/reply_predictor.pkl"):
    art = joblib.load(model_path)
    model,tfidf,scaler = art["model"],art["tfidf"],art["scaler"]
    X_text = tfidf.transform([message]).toarray()
    mf = message_features(message)
    X_msg = np.array([list(mf.values())])
    X_profile, _ = get_feature_matrix(pd.DataFrame([profile]))
    X = np.hstack([X_text, X_msg, X_profile])
    prob = float(np.clip(model.predict(scaler.transform(X))[0], 0, 1))
    tips = []
    if mf["bad_opener"]: tips.append("Avoid generic openers like 'Hope this finds you well'")
    if mf["personalisation"]==0: tips.append("Reference something specific from their profile")
    if not mf["has_question"]: tips.append("End with a clear yes/no question")
    if not mf["has_cta"]: tips.append("Add a soft CTA like '15 mins this week?'")
    quality = "Strong" if (mf["good_opener"]*2+mf["personalisation"]*2+mf["has_question"]+mf["has_cta"]-mf["bad_opener"]*3)>=5 else "Average" if (mf["good_opener"]+mf["has_question"])>=1 else "Weak"
    return {
        "reply_probability": round(prob*100, 1),
        "reply_label": "High" if prob>=0.45 else "Medium" if prob>=0.20 else "Low",
        "message_quality": quality,
        "improvement_tips": tips[:3]
    }

if __name__ == "__main__":
    from data.generate_data import generate_dataset
    print("Training reply predictor...")
    df = generate_dataset(200)
    train_reply_predictor(df)
    profile = {
        "job_title":"Founder","industry":"SaaS","company_size":"11-50",
        "connections":1200,"bio":"Scaling B2B outbound. Cold outreach GTM.",
        "recent_activity":"Posted about cold outreach tips",
        "has_website":1,"posts_per_month":8,"years_experience":7
    }
    good = "Hey, noticed your work on SaaS growth — impressive. We help founders automate outreach. 15 mins this week?"
    bad = "Hi there, I am writing to introduce my services. Hope this message finds you well."
    r1 = predict_reply(profile, good)
    r2 = predict_reply(profile, bad)
    print(f"\nGood message: {r1['reply_probability']}% ({r1['reply_label']}) | Quality: {r1['message_quality']}")
    print(f"Bad message:  {r2['reply_probability']}% ({r2['reply_label']}) | Quality: {r2['message_quality']}")
    print(f"Tips: {r2['improvement_tips']}")
