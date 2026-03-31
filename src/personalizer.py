"""
SBL Lead Engine — Component 3: Message Personalizer
No API key needed — template NLP engine
Optional: set GROQ_API_KEY env variable for LLM upgrade
"""
import random, re, os

random.seed(None)

OPENERS = {
    "hot": [
        "Noticed you're scaling outbound at {company} — impressive.",
        "Your {industry} background immediately stood out to me.",
        "As a {title} in {industry}, you're probably getting a lot of outreach — so I'll keep this sharp.",
        "Came across your profile while researching top {title}s in {industry}.",
    ],
    "warm": [
        "Your work in {industry} caught my eye.",
        "Hey {name}, noticed your focus on {topic} — solid approach.",
        "As someone building in {industry}, thought this might be worth 30 seconds.",
    ],
    "cold": [
        "Quick question — are you currently handling outreach manually?",
        "Hey {name}, reaching out because your work in {industry} aligns with what I'm building.",
    ]
}
VALUE_PROPS = [
    "we help {industry} teams increase reply rates by 3x using AI-personalised outreach",
    "we automate the full outbound funnel — LinkedIn, WhatsApp, iMessage — while keeping it human",
    "we've helped 200+ founders generate $5M+ in pipeline through intelligent outreach automation",
    "our AI chats, handles objections, and books calls — so you focus on closing, not prospecting",
]
CTAS = [
    "Would a 15-min call make sense this week?",
    "Open to a quick chat to see if there's a fit?",
    "Worth a 10-minute conversation?",
    "Can I send over a quick 2-min demo?",
]
TOPIC_MAP = {
    "SaaS":"SaaS growth","E-commerce":"e-commerce scaling","B2B Tech":"B2B sales",
    "Sales Tech":"sales automation","Marketing Tech":"marketing automation",
    "Fintech":"fintech growth","Consulting":"consulting","Digital Marketing":"digital marketing",
    "default":"business growth"
}
BIO_HOOKS = {
    "scaling": "The scaling angle in your bio tells me you're in growth mode.",
    "outbound": "Looks like you're already thinking about outbound — so you know the pain.",
    "b2b": "Your B2B focus is exactly the space we work in.",
    "revenue": "Revenue and pipeline focus — you're speaking our language.",
    "founder": "As a builder, your time is your most valuable asset.",
}

def _bio_hook(bio):
    b = bio.lower()
    for k,v in BIO_HOOKS.items():
        if k in b: return v
    return ""

def personalise_message(profile, lead_label="warm"):
    label = lead_label.lower()
    title = profile.get("job_title","professional")
    industry = profile.get("industry","your industry")
    company = profile.get("company","your company")
    name = profile.get("name","there")
    bio = profile.get("bio","")
    topic = TOPIC_MAP.get(industry, TOPIC_MAP["default"])
    opener = random.choice(OPENERS.get(label, OPENERS["warm"])).format(
        title=title, industry=industry, company=company, name=name, topic=topic
    )
    hook = _bio_hook(bio)
    value = random.choice(VALUE_PROPS).format(industry=industry)
    cta = random.choice(CTAS)
    msg = f"{opener} {hook}\n\nAt SBL, {value}.\n\n{cta}" if hook else f"{opener}\n\nAt SBL, {value}.\n\n{cta}"
    msg = re.sub(r'\n{3,}','\n\n', msg).strip()
    hooks_used = []
    if title not in ["professional"]: hooks_used.append(f"Title: {title}")
    if industry not in ["your industry"]: hooks_used.append(f"Industry: {industry}")
    if hook: hooks_used.append("Bio signal detected")
    return {
        "message": msg,
        "word_count": len(msg.split()),
        "hooks_used": hooks_used,
        "personalisation_level": len(hooks_used)
    }

def personalise_with_llm(profile, lead_label="warm", groq_api_key=None):
    if groq_api_key:
        try:
            import requests
            prompt = f"""Write a short personalised LinkedIn cold outreach message.
Profile: {profile.get('job_title')} at a {profile.get('company_size')} {profile.get('industry')} company.
Bio: {profile.get('bio','')[:100]}
Lead warmth: {lead_label}
Rules: max 100 words, specific opener, mention SBL automates LinkedIn/WhatsApp outreach, end with soft yes/no question, sound human."""
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization":f"Bearer {groq_api_key}","Content-Type":"application/json"},
                json={"model":"llama3-8b-8192","messages":[{"role":"user","content":prompt}],"max_tokens":200,"temperature":0.7},
                timeout=10
            )
            if r.status_code==200:
                msg = r.json()["choices"][0]["message"]["content"].strip()
                return {"message":msg,"word_count":len(msg.split()),"hooks_used":["LLM-generated"],"personalisation_level":3,"source":"groq"}
        except Exception:
            pass
    result = personalise_message(profile, lead_label)
    result["source"] = "template"
    return result

if __name__ == "__main__":
    leads = [
        ({"name":"Priya","job_title":"Founder","industry":"SaaS","company":"GrowthStack","bio":"Scaling B2B outbound revenue. Cold outreach obsessed."},"hot"),
        ({"name":"Rahul","job_title":"Marketing Manager","industry":"Digital Marketing","company":"BrandCo","bio":"Digital marketing and brand strategy."},"warm"),
        ({"name":"Sneha","job_title":"Software Engineer","industry":"Healthcare","bio":"Building healthcare software."},"cold"),
    ]
    for profile,label in leads:
        r = personalise_message(profile, label)
        print(f"\n[{label.upper()}] {profile['name']}")
        print(r["message"])
        print(f"Words: {r['word_count']} | Hooks: {r['hooks_used']}")
        print("-"*50)
