"""
SBL Lead Engine — 10 Lead Demo
Run this to generate your Twitter screenshot
python demo_10_leads.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.agent import run_agent, batch_agent

LEADS = [
    {
        "name": "Rahul Sharma",
        "job_title": "Founder", "industry": "SaaS",
        "company_size": "1-10", "connections": 2100,
        "bio": "Building B2B outbound engine. Cold outreach at scale. GTM obsessed.",
        "recent_activity": "Posted about sales automation",
        "has_website": 1, "posts_per_month": 12, "years_experience": 6
    },
    {
        "name": "Priya Menon",
        "job_title": "CEO", "industry": "Sales Tech",
        "company_size": "11-50", "connections": 1800,
        "bio": "Scaling revenue through outbound. Pipeline generation expert.",
        "recent_activity": "Shared a cold email template",
        "has_website": 1, "posts_per_month": 8, "years_experience": 9
    },
    {
        "name": "Arjun Kapoor",
        "job_title": "Head of Sales", "industry": "B2B Tech",
        "company_size": "51-200", "connections": 950,
        "bio": "Revenue growth through outbound. Building sales teams.",
        "recent_activity": "Posted about B2B sales",
        "has_website": 1, "posts_per_month": 5, "years_experience": 7
    },
    {
        "name": "Sneha Iyer",
        "job_title": "VP Sales", "industry": "Fintech",
        "company_size": "51-200", "connections": 1200,
        "bio": "Driving pipeline and revenue at scale.",
        "recent_activity": "Shared revenue milestone post",
        "has_website": 1, "posts_per_month": 6, "years_experience": 8
    },
    {
        "name": "Karan Mehta",
        "job_title": "Director of Growth", "industry": "Marketing Tech",
        "company_size": "11-50", "connections": 800,
        "bio": "Growth hacking and lead generation. GTM strategy.",
        "recent_activity": "Commented on growth post",
        "has_website": 0, "posts_per_month": 4, "years_experience": 5
    },
    {
        "name": "Divya Nair",
        "job_title": "Marketing Manager", "industry": "Digital Marketing",
        "company_size": "51-200", "connections": 450,
        "bio": "Brand strategy and digital campaigns.",
        "recent_activity": "Liked a marketing post",
        "has_website": 0, "posts_per_month": 3, "years_experience": 4
    },
    {
        "name": "Rohit Verma",
        "job_title": "Product Manager", "industry": "Consulting",
        "company_size": "201-500", "connections": 600,
        "bio": "Product strategy and roadmap planning.",
        "recent_activity": "Shared a product article",
        "has_website": 0, "posts_per_month": 2, "years_experience": 5
    },
    {
        "name": "Ananya Singh",
        "job_title": "Business Development Manager",
        "industry": "Real Estate",
        "company_size": "51-200", "connections": 380,
        "bio": "Business development and client relations.",
        "recent_activity": "No recent activity",
        "has_website": 0, "posts_per_month": 1, "years_experience": 3
    },
    {
        "name": "Vikram Joshi",
        "job_title": "Software Engineer", "industry": "Healthcare",
        "company_size": "201-500", "connections": 220,
        "bio": "Building healthcare software. Python and APIs.",
        "recent_activity": "No recent activity",
        "has_website": 0, "posts_per_month": 1, "years_experience": 3
    },
    {
        "name": "Pooja Reddy",
        "job_title": "HR Manager", "industry": "Manufacturing",
        "company_size": "1000+", "connections": 180,
        "bio": "HR operations and talent acquisition.",
        "recent_activity": "Liked a job posting",
        "has_website": 0, "posts_per_month": 0, "years_experience": 4
    }
]

if __name__ == "__main__":
    print("\n" + "="*62)
    print("  SBL LEAD ENGINE — 10 LEAD CAMPAIGN DEMO")
    print("="*62)

    results = batch_agent(LEADS, verbose=False)

    send    = [r for r in results if r["action"]["action"]=="SEND"]
    review  = [r for r in results if r["action"]["action"]=="REVIEW"]
    nurture = [r for r in results if r["action"]["action"]=="NURTURE"]
    skip    = [r for r in results if r["action"]["action"]=="SKIP"]

    print(f"\n  Total leads   : {len(results)}")
    print(f"  SEND          : {len(send)}")
    print(f"  REVIEW        : {len(review)}")
    print(f"  NURTURE       : {len(nurture)}")
    print(f"  SKIP          : {len(skip)}")
    eff = round((len(send)+len(review))/len(results)*100)
    print(f"  Efficiency    : {eff}% leads worth pursuing")

    print(f"\n{'─'*62}")
    print("  SEND — immediate priority")
    print(f"{'─'*62}")
    for r in send:
        s = r["summary"]
        name = r.get("profile",{}).get("name","Lead")
        msg  = r["action"].get("message","")
        preview = msg[:80].replace("\n"," ") if msg else "—"
        print(f"\n  {name}")
        print(f"  Score: {s['lead_score']}/100 | Reply: {s['reply_prob']}%")
        print(f"  Message: {preview}...")

    print(f"\n{'─'*62}")
    print("  REVIEW — needs approval")
    print(f"{'─'*62}")
    for r in review:
        s = r["summary"]
        print(f"  Score: {s['lead_score']}/100 | Reply: {s['reply_prob']}%")

    print(f"\n{'─'*62}")
    print("  SKIP — not worth budget")
    print(f"{'─'*62}")
    for r in skip:
        s = r["summary"]
        print(f"  Score: {s['lead_score']}/100 | Reason: {r['decision']['reasoning'][:55]}...")

    print(f"\n{'='*62}")

    os.makedirs("data", exist_ok=True)
    report = []
    for r in results:
        s = r["summary"]
        report.append({
            "name":       r.get("profile",{}).get("name","Lead"),
            "label":      s["lead_label"],
            "score":      s["lead_score"],
            "reply_prob": s["reply_prob"],
            "action":     s["final_action"],
            "message":    r["action"].get("message","")[:120]
        })
    with open("data/demo_10_leads.json","w") as f:
        json.dump(report, f, indent=2)
    print("  Results saved → data/demo_10_leads.json")
    print(f"{'='*62}\n")
