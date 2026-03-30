"""
SBL Lead Engine — Synthetic Dataset Generator
"""
import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

JOB_TITLES = {
    "hot": ["Founder","Co-Founder","CEO","Head of Sales","VP Sales","Director of Growth","CRO"],
    "warm": ["Product Manager","Marketing Manager","Head of Marketing","Business Development Manager","CMO"],
    "cold": ["Software Engineer","Data Analyst","HR Manager","Finance Manager","Student","Recruiter"]
}
INDUSTRIES = {
    "hot": ["SaaS","E-commerce","B2B Tech","Sales Tech","Marketing Tech","Fintech"],
    "warm": ["Consulting","Digital Marketing","Real Estate","Insurance","Retail Tech"],
    "cold": ["Education","Healthcare","Government","Non-profit","Manufacturing"]
}
COMPANY_SIZES = {
    "hot": ["1-10","11-50","51-200"],
    "warm": ["11-50","51-200","201-500"],
    "cold": ["1-10","201-500","501-1000","1000+"]
}
BIO_KEYWORDS = {
    "hot": ["scaling","outbound","pipeline","revenue","growth hacking","lead gen","B2B sales","cold outreach","GTM"],
    "warm": ["marketing","brand","digital","strategy","campaigns","partnerships","business development"],
    "cold": ["coding","building products","data","engineering","recruiting","operations","learning","student"]
}
RECENT_ACTIVITY = {
    "hot": ["Posted about LinkedIn outreach tips","Shared a cold email template","Published article on B2B lead gen","Shared revenue milestone"],
    "warm": ["Liked a marketing strategy post","Shared industry news","Commented on growth post"],
    "cold": ["No recent activity","Liked a job posting","Updated profile picture","No posts in 90 days"]
}
REPLY_RATES = {"hot":(0.55,0.90),"warm":(0.20,0.54),"cold":(0.01,0.19)}

def generate_profile(label):
    keywords = random.sample(BIO_KEYWORDS[label], k=random.randint(2,4))
    bio = f"Focused on {keywords[0]} and {keywords[1]}. Passionate about {keywords[-1]}."
    return {
        "job_title": random.choice(JOB_TITLES[label]),
        "industry": random.choice(INDUSTRIES[label]),
        "company_size": random.choice(COMPANY_SIZES[label]),
        "connections": random.randint(*{"hot":(300,5000),"warm":(150,2000),"cold":(50,800)}[label]),
        "bio": bio,
        "recent_activity": random.choice(RECENT_ACTIVITY[label]),
        "has_website": 1 if label in ["hot","warm"] and random.random()>0.4 else 0,
        "posts_per_month": {"hot":random.randint(4,20),"warm":random.randint(1,8),"cold":random.randint(0,3)}[label],
        "years_experience": {"hot":random.randint(3,15),"warm":random.randint(2,12),"cold":random.randint(0,8)}[label],
        "reply_probability": round(random.uniform(*REPLY_RATES[label]),2),
        "label": label,
        "label_encoded": {"hot":2,"warm":1,"cold":0}[label]
    }

def generate_dataset(n_per_class=200):
    records = []
    for label in ["hot","warm","cold"]:
        for _ in range(n_per_class):
            records.append(generate_profile(label))
    df = pd.DataFrame(records).sample(frac=1,random_state=42).reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data/leads_dataset.csv", index=False)
    print(f"Generated {len(df)} leads")
    print(df["label"].value_counts())
