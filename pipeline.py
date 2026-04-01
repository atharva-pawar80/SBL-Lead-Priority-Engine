"""
SBL Lead Engine — Main Pipeline
Runs all 3 components together
Usage:
  python pipeline.py          # demo on 1 lead
  python pipeline.py --train  # train all models
  python pipeline.py --batch  # score 4 leads
"""
import os, sys, argparse
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.lead_scorer import train_model, score_lead
from src.reply_predictor import train_reply_predictor, predict_reply
from src.personalizer import personalise_with_llm

MODEL_SCORER  = "models/lead_scorer.pkl"
MODEL_REPLY   = "models/reply_predictor.pkl"
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", None)

DEMO_PROFILES = [
    {
        "name": "Priya Sharma",
        "job_title": "Founder", "industry": "SaaS",
        "company": "GrowthStack", "company_size": "11-50",
        "connections": 1800, "bio": "Scaling B2B outbound revenue. Cold outreach obsessed. GTM from 0 to 1.",
        "recent_activity": "Posted about cold outreach tips",
        "has_website": 1, "posts_per_month": 10, "years_experience": 8
    },
    {
        "name": "Rahul Mehta",
        "job_title": "Head of Sales", "industry": "B2B Tech",
        "company": "TechPipe", "company_size": "51-200",
        "connections": 950, "bio": "Building pipeline through outbound. Revenue growth at scale.",
        "recent_activity": "Shared a B2B sales article",
        "has_website": 1, "posts_per_month": 5, "years_experience": 6
    },
    {
        "name": "Sneha Patel",
        "job_title": "Marketing Manager", "industry": "Digital Marketing",
        "company": "BrandUp", "company_size": "51-200",
        "connections": 420, "bio": "Digital marketing strategy and brand campaigns.",
        "recent_activity": "Liked a marketing post",
        "has_website": 0, "posts_per_month": 3, "years_experience": 4
    },
    {
        "name": "Amit Singh",
        "job_title": "Software Engineer", "industry": "Healthcare",
        "company": "MedSoft", "company_size": "201-500",
        "connections": 180, "bio": "Building healthcare software. Python and APIs.",
        "recent_activity": "No recent activity",
        "has_website": 0, "posts_per_month": 1, "years_experience": 3
    }
]


def train_all():
    print("="*55)
    print("  SBL LEAD ENGINE — TRAINING ALL MODELS")
    print("="*55)
    from data.generate_data import generate_dataset
    print("\n[1/3] Generating dataset (600 leads)...")
    df = generate_dataset(n_per_class=200)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/leads_dataset.csv", index=False)
    print("      Saved → data/leads_dataset.csv")
    print("\n[2/3] Training Lead Scorer...")
    train_model(df, save_path=MODEL_SCORER)
    print("\n[3/3] Training Reply Predictor...")
    train_reply_predictor(df, save_path=MODEL_REPLY)
    print("\nAll models trained. Run `python pipeline.py` to demo.\n")


def run_pipeline(profile, message=None, verbose=True):
    if not os.path.exists(MODEL_SCORER):
        print("Models not found. Run: python pipeline.py --train")
        sys.exit(1)
    scoring     = score_lead(profile, MODEL_SCORER)
    label       = scoring["label"].lower()
    if message is None:
        message = f"Hey, noticed your background in {profile.get('industry','your industry')} and your role as {profile.get('job_title','professional')}. We help teams like yours automate outbound and book more qualified calls. 15-min call this week?"
    reply       = predict_reply(profile, message, MODEL_REPLY)
    personalised = personalise_with_llm(profile, label, GROQ_API_KEY)
    result = {
        "profile":  {"name": profile.get("name","Lead"), "title": profile.get("job_title"), "industry": profile.get("industry")},
        "scoring":  scoring,
        "reply":    reply,
        "message":  personalised
    }
    if verbose:
        _print_result(result)
    return result


def _print_result(r):
    s  = r["scoring"]
    rp = r["reply"]
    pm = r["message"]
    p  = r["profile"]
    icons = {"Hot":"🔴","Warm":"🟡","Cold":"🔵"}
    print("\n" + "="*55)
    print(f"  SBL LEAD ENGINE REPORT")
    print("="*55)
    print(f"  {p['name']} | {p['title']} | {p['industry']}")
    print("-"*55)
    print(f"\n  {icons.get(s['label'],'⚪')} LEAD: {s['label']} ({s['score']}/100) | Confidence: {s['confidence']}%")
    print(f"     Hot/Warm/Cold: {s['probabilities']['Hot']}% / {s['probabilities']['Warm']}% / {s['probabilities']['Cold']}%")
    print(f"     Reason: {s['reasoning']}")
    print(f"\n  REPLY PREDICTION: {rp['reply_probability']}% ({rp['reply_label']}) | Quality: {rp['message_quality']}")
    if rp["improvement_tips"]:
        for tip in rp["improvement_tips"]:
            print(f"     - {tip}")
    print(f"\n  PERSONALISED MESSAGE ({pm['word_count']} words):")
    print(f"  {'-'*50}")
    for line in pm["message"].split("\n"):
        print(f"  {line}")
    print(f"  {'-'*50}")
    print("="*55)


def batch_score(profiles):
    rows = []
    for p in profiles:
        r = run_pipeline(p, verbose=False)
        rows.append({
            "name":             r["profile"]["name"],
            "title":            r["profile"]["title"],
            "industry":         r["profile"]["industry"],
            "label":            r["scoring"]["label"],
            "score":            r["scoring"]["score"],
            "reply_%":          r["reply"]["reply_probability"],
            "reasoning":        r["scoring"]["reasoning"]
        })
    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    df.index += 1
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SBL Lead Engine")
    parser.add_argument("--train", action="store_true", help="Train all models")
    parser.add_argument("--batch", action="store_true", help="Batch score 4 demo leads")
    parser.add_argument("--lead",  type=int, default=0, help="Demo lead index 0-3")
    args = parser.parse_args()

    if args.train:
        train_all()
    elif args.batch:
        print("\nBatch scoring 4 leads...\n")
        df = batch_score(DEMO_PROFILES)
        print(df.to_string())
        print(f"\nTop lead: {df.iloc[0]['name']} ({df.iloc[0]['label']}, {df.iloc[0]['score']}/100)")
    else:
        idx = min(args.lead, len(DEMO_PROFILES)-1)
        run_pipeline(DEMO_PROFILES[idx], groq_key=GROQ_API_KEY if 'groq_key' in run_pipeline.__code__.co_varnames else None)
