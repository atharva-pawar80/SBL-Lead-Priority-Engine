"""
SBL Lead Engine — LLM Message Generator
Uses Groq free API (LLaMA 3) to generate
hyper-personalised outreach messages
No cost — Groq free tier is generous
"""
import os
import re

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)


def build_prompt(profile: dict, lead_label: str) -> str:
    title = profile.get("job_title", "professional")
    industry = profile.get("industry", "tech")
    company_size = profile.get("company_size", "startup")
    bio = profile.get("bio", "")[:120]
    name = profile.get("name", "there")
    label = lead_label.lower()

    urgency = {
        "hot": "This is a HOT lead — they are highly receptive. Be direct, specific, confident.",
        "warm": "This is a WARM lead — some receptivity. Be friendly but professional.",
        "cold": "This is a COLD lead — low receptivity. Be very brief and low pressure."
    }.get(label, "Be professional.")

    return f"""You are an expert B2B outreach specialist writing a LinkedIn cold message for Second Brain Labs (SBL).

SBL automates LinkedIn and WhatsApp outreach using AI — helping B2B companies generate pipeline at scale.

Lead Profile:
- Name: {name}
- Title: {title}
- Industry: {industry}
- Company size: {company_size}
- Bio: {bio}
- Lead warmth: {label.upper()}

{urgency}

Write ONE short LinkedIn outreach message following these rules:
1. Maximum 80 words
2. First line must reference something SPECIFIC from their profile — title, industry, or bio
3. One clear value proposition about SBL
4. End with ONE soft yes/no question
5. Sound human — never robotic or salesy
6. Never start with "Hi" or "Hello" alone
7. Never use phrases like "Hope this finds you well"

Output only the message. Nothing else. No subject line. No explanation."""


def generate_message_llm(profile: dict, lead_label: str,
                          api_key: str = None) -> dict:
    """
    Generate personalised message using LLaMA 3 via Groq.
    Falls back to template engine if no API key.
    """
    key = api_key or GROQ_API_KEY

    if not key or not GROQ_AVAILABLE:
        return _fallback_message(profile, lead_label)

    try:
        client = Groq(api_key=key)
        prompt = build_prompt(profile, lead_label)

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.75
        )

        message = response.choices[0].message.content.strip()
        message = re.sub(r'\n{3,}', '\n\n', message)

        return {
            "message": message,
            "word_count": len(message.split()),
            "source": "groq_llama3",
            "model": "llama3-8b-8192",
            "hooks_used": ["LLM-personalised from profile"],
            "personalisation_level": 3
        }

    except Exception as e:
        print(f"Groq API error: {e} — falling back to template")
        return _fallback_message(profile, lead_label)


def _fallback_message(profile: dict, lead_label: str) -> dict:
    """Template fallback when no API key available"""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    from src.personalizer import personalise_message
    result = personalise_message(profile, lead_label)
    result["source"] = "template_fallback"
    return result


if __name__ == "__main__":
    print("Testing LLM message generator...\n")

    test_profiles = [
        {
            "name": "Priya",
            "job_title": "Founder",
            "industry": "SaaS",
            "company_size": "11-50",
            "bio": "Scaling B2B outbound revenue. Cold outreach obsessed."
        },
        {
            "name": "Rahul",
            "job_title": "Marketing Manager",
            "industry": "Digital Marketing",
            "company_size": "51-200",
            "bio": "Brand strategy and digital campaigns."
        }
    ]

    labels = ["hot", "warm"]

    for profile, label in zip(test_profiles, labels):
        print(f"[{label.upper()}] {profile['name']} — {profile['job_title']}")
        key = os.getenv("GROQ_API_KEY")
        result = generate_message_llm(profile, label, key)
        print(f"Source: {result['source']}")
        print(f"Message:\n{result['message']}")
        print(f"Words: {result['word_count']}")
        print("-" * 55)
