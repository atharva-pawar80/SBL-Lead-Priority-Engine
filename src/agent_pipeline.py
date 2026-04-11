"""
SBL Lead Engine — Complete Agentic Pipeline
Full flow: Profile → Score → Decide → LLM Message → Act
"""
import os, sys, json
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lead_scorer import score_lead
from src.reply_predictor import predict_reply
from src.llm_engine import generate_message_llm

MODEL_SCORER = "models/lead_scorer.pkl"
MODEL_REPLY  = "models/reply_predictor.pkl"
GROQ_KEY     = os.getenv("GROQ_API_KEY", None)


def full_pipeline(profile: dict, verbose: bool = True) -> dict:
    scoring = score_lead(profile, MODEL_SCORER)
    label   = scoring["label"].lower()
    score   = scoring["score"]

    test_msg = (
        f"Hey, noticed your background in "
        f"{profile.get('industry','your industry')} "
        f"as a {profile.get('job_title','professional')}. "
        f"We help teams like yours automate outbound. "
        f"15 mins this week?"
    )
    reply = predict_reply(profile, test_msg, MODEL_REPLY)
    reply_prob = reply["reply_probability"]

    if label == "hot" and reply_prob >= 60:
        action    = "SEND"
        reasoning = f"Hot lead ({score}/100), strong reply probability ({reply_prob}%). Send immediately."
    elif label == "hot" and reply_prob < 60:
        action    = "REVIEW"
        reasoning = f"Hot lead ({score}/100) but moderate reply probability ({reply_prob}%). Review before sending."
    elif label == "warm" and reply_prob >= 35:
        action    = "REVIEW"
        reasoning = f"Warm lead ({score}/100). Queue for human-approved send."
    elif label == "warm" and reply_prob < 35:
        action    = "NURTURE"
        reasoning = f"Warm lead but low reply probability ({reply_prob}%). Add to nurture sequence."
    else:
        action    = "SKIP"
        reasoning = f"Cold lead ({score}/100). Not worth campaign budget."

    message = None
    if action in ["SEND", "REVIEW"]:
        msg_result = generate_message_llm(profile, label, GROQ_KEY)
        message    = msg_result["message"]

    result = {
        "timestamp": datetime.now().isoformat(),
        "profile": {
            "name":     profile.get("name", "Lead"),
            "title":    profile.get("job_title"),
            "industry": profile.get("industry")
        },
        "score":      score,
        "label":      scoring["label"],
        "confidence": scoring["confidence"],
        "reply_prob": reply_prob,
        "action":     action,
        "reasoning":  reasoning,
        "message":    message
    }

    if verbose:
        icons = {"SEND":"🟢","REVIEW":"🟡","NURTURE":"🔵","SKIP":"🔴"}
        print(f"\n{'='*55}")
        print(f"  {profile.get('name','Lead')} | {profile.get('job_title')} | {profile.get('industry')}")
        print(f"  Label: {scoring['label']} | Score: {score}/100 | Reply: {reply_prob}%")
        print(f"  {icons.get(action,'⚪')} ACTION: {action}")
        print(f"  Reason: {reasoning}")
        if message:
            print(f"\n  MESSAGE:")
            print(f"  {'─'*50}")
            for line in message.split("\n"):
                print(f"  {line}")
            print(f"  {'─'*50}")
        print(f"{'='*55}")

    return result


def run_campaign(profiles: list, save: bool = True) -> list:
    results = []
    for p in profiles:
        r = full_pipeline(p, verbose=False)
        results.append(r)

    results.sort(key=lambda x: x["score"], reverse=True)

    send    = [r for r in results if r["action"] == "SEND"]
    review  = [r for r in results if r["action"] == "REVIEW"]
    nurture = [r for r in results if r["action"] == "NURTURE"]
    skip    = [r for r in results if r["action"] == "SKIP"]

    print(f"\n{'='*55}")
    print(f"  CAMPAIGN REPORT")
    print(f"{'='*55}")
    print(f"  Total    : {len(results)}")
    print(f"  🟢 SEND   : {len(send)}")
    print(f"  🟡 REVIEW : {len(review)}")
    print(f"  🔵 NURTURE: {len(nurture)}")
    print(f"  🔴 SKIP   : {len(skip)}")
    eff = round((len(send)+len(review))/len(results)*100)
    print(f"  Efficiency: {eff}% leads worth pursuing")

    if send:
        print(f"\n  🟢 SEND NOW:")
        for r in send:
            print(f"  → {r['profile']['name']} | {r['label']} | Score:{r['score']} | Reply:{r['reply_prob']}%")
            if r["message"]:
                print(f"    {r['message'][:80].replace(chr(10),' ')}...")

    if skip:
        print(f"\n  🔴 SKIPPED:")
        for r in skip:
            print(f"  → Score:{r['score']} | {r['reasoning'][:55]}...")

    print(f"{'='*55}\n")

    if save:
        os.makedirs("data", exist_ok=True)
        with open("data/campaign_report.json", "w") as f:
            json.dump(results, f, indent=2)
        print("  Saved → data/campaign_report.json")

    return results


if __name__ == "__main__":
    from demo_10_leads import LEADS
    run_campaign(LEADS)
