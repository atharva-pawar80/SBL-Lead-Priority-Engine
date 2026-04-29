"""
SBL Lead Engine — MLOps Monitor
Tracks model performance over time
Shows drift, accuracy, agent decisions
This is what separates deployed from maintained
"""
import json
import os
from datetime import datetime, timedelta
import random

LOG_FILE    = "data/agent_decisions.json"
MONITOR_LOG = "data/monitor_log.json"


def log_prediction(profile: dict, result: dict):
    """Log every prediction for monitoring"""
    os.makedirs("data", exist_ok=True)
    logs = []
    if os.path.exists(MONITOR_LOG):
        try:
            with open(MONITOR_LOG) as f:
                logs = json.load(f)
        except:
            logs = []
    logs.append({
        "timestamp":  datetime.now().isoformat(),
        "profile":    profile.get("job_title",""),
        "industry":   profile.get("industry",""),
        "label":      result.get("label",""),
        "score":      result.get("score", 0),
        "reply_prob": result.get("reply_prob", 0),
        "action":     result.get("action",""),
    })
    with open(MONITOR_LOG, "w") as f:
        json.dump(logs, f, indent=2)


def get_model_health() -> dict:
    """Check if model is performing within expected ranges"""
    if not os.path.exists(MONITOR_LOG):
        return {"status": "no data yet", "predictions": 0}

    with open(MONITOR_LOG) as f:
        logs = json.load(f)

    if not logs:
        return {"status": "no data", "predictions": 0}

    total     = len(logs)
    hot_rate  = len([l for l in logs if l["label"]=="Hot"]) / total
    cold_rate = len([l for l in logs if l["label"]=="Cold"]) / total
    send_rate = len([l for l in logs if l["action"]=="SEND"]) / total
    skip_rate = len([l for l in logs if l["action"]=="SKIP"]) / total
    avg_score = sum(l["score"] for l in logs) / total
    avg_reply = sum(l["reply_prob"] for l in logs) / total

    # Health checks
    issues = []
    if hot_rate > 0.7:
        issues.append("WARNING: Too many Hot leads — possible data drift")
    if cold_rate > 0.8:
        issues.append("WARNING: Too many Cold leads — check input quality")
    if avg_reply < 20:
        issues.append("WARNING: Low reply predictions — check message quality")
    if skip_rate > 0.7:
        issues.append("WARNING: Agent skipping too many — leads may be low quality")

    return {
        "status":        "healthy" if not issues else "needs attention",
        "predictions":   total,
        "hot_rate":      round(hot_rate * 100, 1),
        "cold_rate":     round(cold_rate * 100, 1),
        "send_rate":     round(send_rate * 100, 1),
        "skip_rate":     round(skip_rate * 100, 1),
        "avg_score":     round(avg_score, 1),
        "avg_reply_prob":round(avg_reply, 1),
        "issues":        issues,
        "last_updated":  datetime.now().isoformat()
    }


def print_health_report():
    health = get_model_health()
    print("\n" + "="*55)
    print("  SBL LEAD ENGINE — MODEL HEALTH REPORT")
    print("="*55)
    status_icon = "✅" if health["status"] == "healthy" else "⚠️"
    print(f"\n  {status_icon} Status: {health['status'].upper()}")
    print(f"  Total predictions logged: {health['predictions']}")
    if health['predictions'] > 0:
        print(f"\n  Lead distribution:")
        print(f"    Hot leads  : {health['hot_rate']}%")
        print(f"    Cold leads : {health['cold_rate']}%")
        print(f"\n  Agent decisions:")
        print(f"    Send rate  : {health['send_rate']}%")
        print(f"    Skip rate  : {health['skip_rate']}%")
        print(f"\n  Model metrics:")
        print(f"    Avg score  : {health['avg_score']}/100")
        print(f"    Avg reply  : {health['avg_reply_prob']}%")
    if health.get('issues'):
        print(f"\n  Issues detected:")
        for issue in health['issues']:
            print(f"    ⚠️  {issue}")
    else:
        print(f"\n  No issues detected — model performing normally")
    print("="*55 + "\n")
    return health


if __name__ == "__main__":
    print_health_report()
