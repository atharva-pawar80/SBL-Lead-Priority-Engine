"""
SBL Lead Engine — REST API
Run: uvicorn api:app --reload
Docs: http://localhost:8000/docs
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

from src.lead_scorer import score_lead
from src.reply_predictor import predict_reply
from src.personalizer import personalise_with_llm

app = FastAPI(
    title="SBL Lead Engine API",
    description="ML-powered lead scoring, reply prediction and message personalisation for Second Brain Labs",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

MODEL_SCORER = "models/lead_scorer.pkl"
MODEL_REPLY  = "models/reply_predictor.pkl"


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

class FullPipelineRequest(BaseModel):
    profile:  LeadProfile
    message:  Optional[str] = None


@app.get("/")
def root():
    return {
        "product": "SBL Lead Engine",
        "version": "1.0.0",
        "built_for": "Second Brain Labs — sbl.so",
        "endpoints": ["/score", "/reply", "/personalise", "/analyse", "/health"]
    }


@app.get("/health")
def health():
    scorer_ready = os.path.exists(MODEL_SCORER)
    reply_ready  = os.path.exists(MODEL_REPLY)
    return {
        "status": "ready" if scorer_ready and reply_ready else "models not trained",
        "lead_scorer":     "loaded" if scorer_ready else "missing — run python pipeline.py --train",
        "reply_predictor": "loaded" if reply_ready  else "missing — run python pipeline.py --train"
    }


@app.post("/score")
def score(req: ScoreRequest):
    """
    Score a LinkedIn lead as Hot / Warm / Cold.
    Returns priority score 0-100, confidence %, and reasoning.
    """
    if not os.path.exists(MODEL_SCORER):
        raise HTTPException(status_code=503, detail="Models not trained. Run: python pipeline.py --train")
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
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reply")
def reply(req: ReplyRequest):
    """
    Predict reply probability for a message + profile combination.
    Returns reply %, message quality, and improvement tips.
    """
    if not os.path.exists(MODEL_REPLY):
        raise HTTPException(status_code=503, detail="Models not trained. Run: python pipeline.py --train")
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
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/personalise")
def personalise(req: ScoreRequest):
    """
    Generate a personalised outreach message for a lead.
    Automatically detects lead label and adapts tone.
    """
    try:
        scoring = score_lead(req.profile.dict(), MODEL_SCORER)
        label   = scoring["label"].lower()
        result  = personalise_with_llm(req.profile.dict(), label)
        return {
            "name":                req.profile.name,
            "lead_label":          scoring["label"],
            "message":             result["message"],
            "word_count":          result["word_count"],
            "hooks_used":          result["hooks_used"],
            "personalisation_level": result["personalisation_level"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyse")
def analyse(req: FullPipelineRequest):
    """
    Full pipeline — score + reply prediction + personalised message.
    This is the main endpoint SBL would call before launching a campaign.
    """
    if not os.path.exists(MODEL_SCORER):
        raise HTTPException(status_code=503, detail="Models not trained. Run: python pipeline.py --train")
    try:
        profile = req.profile.dict()

        scoring      = score_lead(profile, MODEL_SCORER)
        label        = scoring["label"].lower()

        message = req.message or f"Hey, noticed your background in {profile.get('industry')} as a {profile.get('job_title')}. We help teams like yours automate outbound. 15 mins this week?"
        reply        = predict_reply(profile, message, MODEL_REPLY)
        personalised = personalise_with_llm(profile, label)

        return {
            "name":     req.profile.name,
            "lead_score": {
                "label":         scoring["label"],
                "score":         scoring["score"],
                "confidence":    scoring["confidence"],
                "probabilities": scoring["probabilities"],
                "reasoning":     scoring["reasoning"]
            },
            "reply_prediction": {
                "reply_probability": reply["reply_probability"],
                "reply_label":       reply["reply_label"],
                "message_quality":   reply["message_quality"],
                "improvement_tips":  reply["improvement_tips"]
            },
            "personalised_message": {
                "message":               personalised["message"],
                "word_count":            personalised["word_count"],
                "hooks_used":            personalised["hooks_used"],
                "personalisation_level": personalised["personalisation_level"]
            },
            "recommendation": _recommendation(scoring["label"], reply["reply_probability"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _recommendation(label, reply_prob):
    if label == "Hot" and reply_prob >= 50:
        return "Priority — launch personalised campaign immediately"
    elif label == "Hot" and reply_prob < 50:
        return "High potential — improve message before sending"
    elif label == "Warm":
        return "Queue for standard campaign — monitor reply rate"
    else:
        return "Low priority — skip or add to nurture sequence"


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
