"""
SBL Lead Engine — Main Pipeline v2.0
Now with Agentic AI + LLM layer
Usage:
  python pipeline.py --train     # train all models
  python pipeline.py --demo      # run agent on 1 lead
  python pipeline.py --batch     # run agent on 4 leads
  python pipeline.py --full      # full agentic batch with report
"""
import os, sys, argparse, json
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.lead_scorer import train_model, score_lead
from src.reply_predictor import train_reply_predictor, predict_reply
from src.agent import run_agent, batch_agent

MODEL_SCORER = "models/lead_scorer.pkl"
MODEL_REPLY  = "models/reply_predictor.pkl"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)

DEMO_PROFILES = [
    {
        "name": "Priya Sharma",
        "job_title": "Founder", "industry": "SaaS",
        "company": "GrowthStack", "company_size": "11-50",
        "connections": 1800,
        "bio": "Scaling B2B outbound revenue. Cold outreach obsessed. GTM from 0 to 1.",
        "recent_activity": "Posted about cold outreach tips",
        "has_website": 1, "posts_per_month": 10, "years_experience": 8
    },
    {
        "name": "Rahul Mehta",
        "job_title": "Head of Sales", "industry": "B2B Tech",
        "company": "TechPipe", "company_size": "51-200",
        "connections": 950,
        "bio": "Building pipeline through outbound. Revenue growth at scale.",
        "recent_activity": "Shared a B2B sales article",
        "has_website": 1, "posts_per_month": 5, "years_experience": 6
    },
    {
        "name": "Sneha Patel",
        "job_title": "Marketing Manager", "industry": "Digital Marketing",
        "company": "BrandUp", "company_size": "51-200",
        "connections": 420,
        "bio": "Digital marketing strategy and brand campaigns.",
        "recent_activity": "Liked a marketing post",
        "has_website": 0, "posts_per_month": 3, "years_experience": 4
    },
    {
        "name": "Amit Singh",
        "job_title": "Software Engineer", "industry": "Healthcare",
        "company": "MedSoft", "company_size": "201-500",
        "connections": 180,
        "bio": "Building healthcare software. Python and APIs.",
        "recent_activity": "No recent activity",
        "has_website": 0, "posts_per_month": 1, "years_experience": 3
    }
]


def train_all():
    print("="*55)
    print("  SBL LEAD ENGINE v2.0 — TRAINING")
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
    print("\nAll models trained.")
    print("Run: python pipeline.py --demo\n")


def run_full_report(profiles):
    """
    Full agentic batch report
    Shows every agent decision in a clean summary
    """
    print("\n" + "="*60)
    print("  SBL LEAD ENGINE — AGENTIC CAMPAIGN REPORT")
    print("="*60)

    results = batch_agent(profiles, verbose=False)

    send_list    = [r for r in results if r["action"]["action"] == "SEND"]
    review_list  = [r for r in results if r["action"]["action"] == "REVIEW"]
    nurture_list = [r for r in results if r["action"]["action"] == "NURTURE"]
    skip_list    = [r for r in results if r["action"]["action"] == "SKIP"]

    print(f"\n  Total leads processed : {len(results)}")
    print(f"  🟢 SEND   : {len(send_list)}")
    print(f"  🟡 REVIEW : {len(review_list)}")
    print(f"  🔵 NURTURE: {len(nurture_list)}")
    print(f"  🔴 SKIP   : {len(skip_list)}")
    print(f"\n  Campaign efficiency   : "
          f"{round((len(send_list)+len(review_list))/len(results)*100)}% "
          f"leads worth pursuing")

    if send_list:
        print(f"\n{'─'*60}")
        print("  🟢 IMMEDIATE SEND — Top priority leads")
        print(f"{'─'*60}")
        for r in send_list:
            s = r["summary"]
            print(f"\n  {s['profile']['name'] if 'profile' in s else 'Lead'} "
                  f"| {s['lead_label']} | "
                  f"Score: {s['lead_score']}/100 | "
                  f"Reply: {s['reply_prob']}%")
            if r["action"].get("message"):
                preview = r["action"]["message"][:100].replace("\n"," ")
                print(f"  Message: {preview}...")

    if review_list:
        print(f"\n{'─'*60}")
        print("  🟡 REVIEW QUEUE — Needs human approval")
        print(f"{'─'*60}")
        for r in review_list:
            s = r["summary"]
            print(f"\n  {s.get('lead_label','Lead')} | "
                  f"Score: {s['lead_score']}/100 | "
                  f"Reply: {s['reply_prob']}%")

    if skip_list:
        print(f"\n{'─'*60}")
        print("  🔴 SKIPPED — Not worth campaign budget")
        print(f"{'─'*60}")
        for r in skip_list:
            s = r["summary"]
            print(f"  Score: {s['lead_score']}/100 | "
                  f"Reason: {r['decision']['reasoning'][:60]}...")

    print(f"\n{'='*60}")
    print("  Agent decisions saved → data/agent_decisions.json")
    print(f"{'='*60}\n")

    save_report(results)
    return results


def save_report(results):
    """Save full report as JSON"""
    os.makedirs("data", exist_ok=True)
    report = []
    for r in results:
        report.append({
            "name":         r["summary"].get("lead_label", ""),
            "lead_score":   r["summary"]["lead_score"],
            "reply_prob":   r["summary"]["reply_prob"],
            "action":       r["summary"]["final_action"],
            "message":      r["action"].get("message", ""),
            "reasoning":    r["decision"]["reasoning"]
        })
    with open("data/campaign_report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SBL Lead Engine v2.0")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--demo",  action="store_true")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--full",  action="store_true")
    args = parser.parse_args()

    if args.train:
        train_all()

    elif args.demo:
        print("\nRunning agent on single lead...\n")
        run_agent(DEMO_PROFILES[0], verbose=True)

    elif args.batch:
        print("\nBatch agent run on 4 leads...\n")
        results = batch_agent(DEMO_PROFILES, verbose=False)
        rows = []
        for r in results:
            rows.append({
                "lead_score":  r["summary"]["lead_score"],
                "reply_%":     r["summary"]["reply_prob"],
                "action":      r["summary"]["final_action"],
                "reasoning":   r["decision"]["reasoning"][:50]
            })
        df = pd.DataFrame(rows)
        df.index += 1
        print(df.to_string())

    elif args.full:
        run_full_report(DEMO_PROFILES)

    else:
        print("SBL Lead Engine v2.0")
        print("Commands:")
        print("  python pipeline.py --train")
        print("  python pipeline.py --demo")
        print("  python pipeline.py --batch")
        print("  python pipeline.py --full")
