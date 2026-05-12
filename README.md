# SBL-Lead-Priority-Engine
An end-to-end ML pipeline built specifically for Second Brain Labs — predicting lead intent, reply probability, and generating personalised outreach messages from LinkedIn profiles
add  the license 


"SBL is an AI sales automation company. They send LinkedIn outreach messages for B2B companies. The problem was they had no way to know which leads were worth messaging before the campaign started.
I built a 3-component ML pipeline that solves this. First, a lead scorer that reads 12 signals from a LinkedIn profile and classifies it as Hot, Warm, or Cold with a priority score. Second, a reply predictor that takes your message and predicts the probability someone will actually reply. Third, an agentic AI layer that uses LLaMA 3 to write a personalised message and then autonomously decides whether to send it, queue it for review, or skip it entirely.
The result — from 1000 leads, the agent picks the right 60 worth having a conversation with. The co-founder of SBL saw it, called it great work publicly, and used my exact architecture as an example in his product video
