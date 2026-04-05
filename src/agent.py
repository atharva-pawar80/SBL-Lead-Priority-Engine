"""
SBL Lead Engine — Agentic Layer
The agent thinks, decides, and acts autonomously
This is what Ayush asked for — Core ML + Agentic AI
"""
import os
import sys
import json
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lead_scorer import score_lead
from src.reply_predictor import predict_reply
from src.llm_engine import generate_message_llm

MODEL_SCORER = "models/lead_scorer.pkl"
MODEL_REPLY  = "models/reply_predictor.pkl"
LOG_FILE     = "data/agent_decisions.json"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)


AGENT_RULES = {
    "hot":  {"min_reply": 40, "action": "SEND",   "priority": 1},
    "warm": {"min_reply": 25, "action": "REVIEW",  "priority": 2},
    "cold": {"min_reply":  0, "action": "SKIP",    "priority": 3},
}


def agent_think(profile: dict) -> dict:
    """
    Agent reasoning step — analyses the profile
    and thinks before deciding
    """
    scoring = score_lead(profile, MODEL_SCORER)
    label   = scoring["label"].lower()

    default_msg = (
        f"Hey, noticed your background in "
        f"{profile.get('industry','your industry')} "
        f"as a {profile.get('job_title','professional')}. "
        f"We help teams like yours automate outbound. "
        f"15 mins this week?"
    )

    reply = predict_reply(profile, default_msg, MODEL_REPLY)

    thoughts = []
    thoughts.append(
        f"Lead is {scoring['label']} with "
        f"{scoring['confidence']}% confidence"
    )
    thoughts.append(
        f"Reply probability is "
        f"{reply['reply_probability']}% — "
        f"classified as {reply['reply_label']}"
    )
    if scoring["score"] >= 50:
        thoughts.append(
            "High priority signals detected — "
            "worth personalised outreach"
        )
    elif scoring["score"] >= 25:
        thoughts.append(
            "Moderate signals — "
            "standard outreach may work"
        )
    else:
        thoughts.append(
            "Low priority signals — "
            "outreach unlikely to convert"
        )

    return {
        "scoring":  scoring,
        "reply":    reply,
        "label":    label,
        "thoughts": thoughts
    }


def agent_decide(thinking: dict) -> dict:
    """
    Agent decision step — makes the final call
    SEND / REVIEW / SKIP with full reasoning
    """
    label      = thinking["label"]
    score      = thinking["scoring"]["score"]
    reply_prob = thinking["reply"]["reply_probability"]
    rules      = AGENT_RULES[label]

    action = rules["action"]

    if label == "hot" and reply_prob >= 60:
        action   = "SEND"
        reasoning = (
            f"Hot lead ({score}/100) with strong reply "
            f"probability ({reply_prob}%). "
            f"Agent decided to send immediately."
        )
    elif label == "hot" and reply_prob < 60:
        action   = "REVIEW"
        reasoning = (
            f"Hot lead ({score}/100) but reply probability "
            f"is moderate ({reply_prob}%). "
            f"Recommend human review before sending."
        )
    elif label == "warm" and reply_prob >= 35:
        action   = "REVIEW"
        reasoning = (
            f"Warm lead ({score}/100) with decent reply "
            f"probability ({reply_prob}%). "
            f"Queue for human-approved send."
        )
    elif label == "warm" and reply_prob < 35:
        action   = "NURTURE"
        reasoning = (
            f"Warm lead but low reply probability "
            f"({reply_prob}%). "
            f"Add to nurture sequence instead."
        )
    else:
        action   = "SKIP"
        reasoning = (
            f"Cold lead ({score}/100) with low reply "
            f"probability ({reply_prob}%). "
            f"Agent decided to skip — "
            f"not worth campaign budget."
        )

    return {
        "action":    action,
        "reasoning": reasoning,
        "priority":  rules["priority"],
        "score":     score,
        "reply_prob": reply_prob
    }


def agent_act(profile: dict,
              thinking: dict,
              decision: dict) -> dict:
    """
    Agent action step — executes the decision
    Generates message if action is SEND or REVIEW
    """
    action = decision["action"]
    label  = thinking["label"]

    if action in ["SEND", "REVIEW"]:
        message_result = generate_message_llm(
            profile, label, GROQ_API_KEY
        )
        return {
            "action":          action,
            "message":         message_result["message"],
            "message_source":  message_result["source"],
            "word_count":      message_result["word_count"],
            "executed":        True
        }

    elif action == "NURTURE":
        return {
            "action":   action,
            "message":  "Added to 30-day nurture sequence.",
            "executed": True
        }

    else:
        return {
            "action":   "SKIP",
            "message":  None,
            "executed": False
        }


def run_agent(profile: dict,
              verbose: bool = True) -> dict:
    """
    Full agentic pipeline — Think → Decide → Act
    This is the complete autonomous loop
    """
    if not os.path.exists(MODEL_SCORER):
        print("Models not found. Run: python pipeline.py --train")
        sys.exit(1)

    if verbose:
        print(f"\n{'='*55}")
        print(f"  AGENT PROCESSING: {profile.get('name','Lead')}")
        print(f"{'='*55}")

    if verbose:
        print("\n[1/3] THINKING...")
    thinking = agent_think(profile)

    if verbose:
        for t in thinking["thoughts"]:
            print(f"  → {t}")

    if verbose:
        print("\n[2/3] DECIDING...")
    decision = agent_decide(thinking)

    if verbose:
        print(f"  → Action: {decision['action']}")
        print(f"  → Reason: {decision['reasoning']}")

    if verbose:
        print("\n[3/3] ACTING...")
    result = agent_act(profile, thinking, decision)

    if verbose:
        icons = {
            "SEND":    "🟢",
            "REVIEW":  "🟡",
            "NURTURE": "🔵",
            "SKIP":    "🔴"
        }
        icon = icons.get(result["action"], "⚪")
        print(f"\n  {icon} FINAL ACTION: {result['action']}")
        if result["message"]:
            print(f"\n  MESSAGE ({result.get('word_count',0)} words):")
            print(f"  {'-'*50}")
            for line in result["message"].split("\n"):
                print(f"  {line}")
            print(f"  {'-'*50}")
        print(f"{'='*55}")

    final = {
        "timestamp":   datetime.now().isoformat(),
        "profile":     {
            "name":     profile.get("name"),
            "title":    profile.get("job_title"),
            "industry": profile.get("industry")
        },
        "thinking":    thinking,
        "decision":    decision,
        "action":      result,
        "summary": {
            "lead_label":      thinking["scoring"]["label"],
            "lead_score":      thinking["scoring"]["score"],
            "reply_prob":      thinking["reply"]["reply_probability"],
            "final_action":    result["action"],
            "message_preview": (
                result["message"][:80] + "..."
                if result.get("message") else None
            )
        }
    }

    _log_decision(final)
    return final


def _log_decision(result: dict):
    """Save every agent decision to log file"""
    os.makedirs("data", exist_ok=True)
    logs = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                logs = json.load(f)
        except Exception:
            logs = []
    logs.append(result["summary"])
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)


def batch_agent(profiles: list,
                verbose: bool = False) -> list:
    """Run agent on multiple profiles — returns sorted results"""
    results = []
    for p in profiles:
        r = run_agent(p, verbose=verbose)
        results.append(r)

    results.sort(
        key=lambda x: x["summary"]["lead_score"],
        reverse=True
    )
    return results


if __name__ == "__main__":
    sys.path.insert(
        0,
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    test_profiles = [
        {
            "name": "Priya Sharma",
            "job_title": "Founder",
            "industry": "SaaS",
            "company_size": "11-50",
            "connections": 1800,
            "bio": "Scaling B2B outbound. Cold outreach GTM.",
            "recent_activity": "Posted about cold outreach tips",
            "has_website": 1,
            "posts_per_month": 10,
            "years_experience": 8
        },
        {
            "name": "Rahul Mehta",
            "job_title": "Marketing Manager",
            "industry": "Digital Marketing",
            "company_size": "51-200",
            "connections": 420,
            "bio": "Brand strategy and digital campaigns.",
            "recent_activity": "Liked a post",
            "has_website": 0,
            "posts_per_month": 3,
            "years_experience": 4
        },
        {
            "name": "Amit Singh",
            "job_title": "Software Engineer",
            "industry": "Healthcare",
            "company_size": "201-500",
            "connections": 180,
            "bio": "Building healthcare software.",
            "recent_activity": "No recent activity",
            "has_website": 0,
            "posts_per_month": 1,
            "years_experience": 3
        }
    ]

    print("Running agent on 3 leads...\n")
    for profile in test_profiles:
        run_agent(profile, verbose=True)

    print("\nAgent decision log saved → data/agent_decisions.json")
