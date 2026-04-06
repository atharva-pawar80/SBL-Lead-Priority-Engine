"""
SBL Lead Engine API v2.0
Now with Agentic AI endpoint
Run: uvicorn api:app --reload
Docs: http://localhost:8000/docs
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

from src.lead_scorer import score_lead
from src.reply_predictor import predict_reply
from src.personalizer import personalise_with_llm
from src.agent import run_agent, batch_agent

app = FastAPI(
    title="SBL Lead Engine API",
    description="""
ML-powered lead scoring, reply prediction,
message personalisation and agentic AI
for Second Brain Labs — sbl.so

Built by Atharva Pawar
GitHub: github.com/atharva-pawar80/SBL-Lead-Priority-Engine
    """,
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

MODEL_SCORER = "models/lead_scorer.pkl"
MODEL_REPLY  = "models/reply_predictor.pkl"
GROQ_KEY     = os.getenv("GROQ_API_KEY", None)


class LeadProfile(BaseModel):
    name:             Optional[str] = "Lead"
    job_title:        str
    industry:         str
    company_size:     str
    connections:      Optional[int] = 500
    bio:              Optional[str] = ""
    recent_activity:  Optional[str] = "No recent activity"
    has_website:      Optional[int] = 0
    posts_per_month:  Optional[int] = 3
    years_experience: Optional[int] = 3
    company:          Optional[str] = "their company"

class ScoreRequest(BaseModel):
    profile: LeadProfile

class ReplyRequest(BaseModel):
    profile: LeadProfile
    message: str

class AgentRequest(BaseModel):
    profile: LeadProfile

class BatchAgentRequest(BaseModel):
    profiles: List[LeadProfile]


@app.get("/")
def root():
    return {
        "product":    "SBL Lead Engine",
        "version":    "2.0.0",
        "built_for":  "Second Brain Labs — sbl.so",
        "what_it_does": [
            "Scores LinkedIn leads as Hot / Warm / Cold",
            "Predicts reply probability for any message",
            "Generates personalised outreach messages using LLM",
            "Autonomous agent: SEND / REVIEW / SKIP decisions"
        ],
        "endpoints": [
            "GET  /health",
            "POST /score",
            "POST /reply",
            "POST /personalise",
            "POST /agent",
            "POST /agent/batch"
        ]
    }


@app.get("/health")
def health():
    return {
        "status":          "ready" if os.path.exists(MODEL_SCORER) else "models not trained",
        "lead_scorer":     "loaded" if os.path.exists(MODEL_SCORER) else "run --train",
        "reply_predictor": "loaded" if os.path.exists(MODEL_REPLY) else "run --train",
        "llm":             "groq connected" if GROQ_KEY else "template fallback",
        "version":         "2.0.0"
    }


@app.post("/score")
def score(req: ScoreRequest):
    """
    Score a LinkedIn lead as Hot / Warm / Cold.
    Returns priority score 0-100 + confidence + reasoning.
    """
    if not os.path.exists(MODEL_SCORER):
        raise HTTPException(503, "Models not trained. Run: python pipeline.py --train")
    try:
        result = score_lead(req.profile.dict(), MODEL_SCORER)
        return {
            "name":          req.profile.name,
            "label":         result["label"],
            "score":         result["score"],
            "confidence":    result["confidence"],
            "probabilities": result["probabilities"],
            "reasoning":     result["reasoning"]
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/reply")
def reply(req: ReplyRequest):
    """
    Predict reply probability for a message + profile.
    Returns reply % + message quality + improvement tips.
    """
    if not os.path.exists(MODEL_REPLY):
        raise HTTPException(503, "Models not trained. Run: python pipeline.py --train")
    try:
        result = predict_reply(req.profile.dict(), req.message, MODEL_REPLY)
        return {
            "name":              req.profile.name,
            "reply_probability": result["reply_probability"],
            "reply_label":       result["reply_label"],
            "message_quality":   result["message_quality"],
            "improvement_tips":  result["improvement_tips"]
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/personalise")
def personalise(req: ScoreRequest):
    """
    Generate a personalised outreach message using LLM.
    Auto-detects lead label and adapts tone.
    """
    try:
        scoring = score_lead(req.profile.dict(), MODEL_SCORER)
        label   = scoring["label"].lower()
        result  = personalise_with_llm(req.profile.dict(), label, GROQ_KEY)
        return {
            "name":                  req.profile.name,
            "lead_label":            scoring["label"],
            "message":               result["message"],
            "word_count":            result["word_count"],
            "hooks_used":            result["hooks_used"],
            "personalisation_level": result["personalisation_level"],
            "source":                result.get("source", "template")
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/agent")
def agent(req: AgentRequest):
    """
    Full agentic pipeline for one lead.
    Think → Decide → Act autonomously.

    Returns:
    - Lead score + label
    - Agent decision: SEND / REVIEW / SKIP / NURTURE
    - LLM-generated personalised message (if SEND or REVIEW)
    - Full reasoning chain

    This is the main endpoint SBL would call
    before launching a campaign.
    """
    if not os.path.exists(MODEL_SCORER):
        raise HTTPException(503, "Models not trained. Run: python pipeline.py --train")
    try:
        result = run_agent(req.profile.dict(), verbose=False)
        return {
            "name":    req.profile.name,
            "lead": {
                "label":      result["thinking"]["scoring"]["label"],
                "score":      result["thinking"]["scoring"]["score"],
                "confidence": result["thinking"]["scoring"]["confidence"],
                "reasoning":  result["thinking"]["scoring"]["reasoning"]
            },
            "reply_probability": result["thinking"]["reply"]["reply_probability"],
            "agent": {
                "action":    result["action"]["action"],
                "reasoning": result["decision"]["reasoning"],
                "message":   result["action"].get("message"),
                "executed":  result["action"]["executed"]
            },
            "thoughts": result["thinking"]["thoughts"],
            "summary":  result["summary"]
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/agent/batch")
def agent_batch(req: BatchAgentRequest):
    """
    Run the agent on multiple leads at once.
    Returns sorted results — highest priority first.

    Perfect for: pre-campaign lead filtering.
    Feed in 500 leads → get back prioritised SEND list.
    """
    if not os.path.exists(MODEL_SCORER):
        raise HTTPException(503, "Models not trained.")
    if len(req.profiles) > 50:
        raise HTTPException(400, "Max 50 leads per batch request.")
    try:
        profiles = [p.dict() for p in req.profiles]
        results  = batch_agent(profiles, verbose=False)

        summary = {
            "total":    len(results),
            "send":     len([r for r in results if r["action"]["action"]=="SEND"]),
            "review":   len([r for r in results if r["action"]["action"]=="REVIEW"]),
            "nurture":  len([r for r in results if r["action"]["action"]=="NURTURE"]),
            "skip":     len([r for r in results if r["action"]["action"]=="SKIP"]),
            "efficiency": f"{round((len([r for r in results if r['action']['action'] in ['SEND','REVIEW']])/len(results))*100)}%"
        }

        leads = []
        for r in results:
            leads.append({
                "action":       r["action"]["action"],
                "lead_score":   r["summary"]["lead_score"],
                "reply_prob":   r["summary"]["reply_prob"],
                "reasoning":    r["decision"]["reasoning"],
                "message":      r["action"].get("message")
            })

        return {
            "campaign_summary": summary,
            "leads": leads
        }
    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
