# StrategyPulse AI

**Real-time Competitive Intelligence & Strategic Signal Detection Platform**

Built to demonstrate applied AI + full-stack engineering at McKinsey's standard of competitive intelligence work.

---

## What it does

StrategyPulse AI monitors six Fortune 500 companies in real time — ingesting signals from news APIs, SEC EDGAR filings, and research papers — and uses AI to detect strategic shifts before they become public knowledge.

- **Signal ingestion** — NewsAPI, SEC EDGAR (free), arXiv
- **NLP classification** — Zero-shot strategy categorisation via `facebook/bart-large-mnli`
- **Financial sentiment** — FinBERT (`ProsusAI/finbert`) for positive/negative/neutral
- **Porter's Five Forces mapping** — Every signal mapped to a competitive force
- **Anomaly detection** — LSTM autoencoder on 60-day signal volume time-series
- **Strategy briefs** — RAG-powered McKinsey-style intelligence reports via ChromaDB
- **Live dashboard** — FastAPI + WebSocket, signals stream every 9 seconds

---

## Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + WebSocket |
| AI/NLP | HuggingFace Transformers, FinBERT, BART |
| Vector store | ChromaDB + sentence-transformers |
| Anomaly detection | PyTorch LSTM Autoencoder |
| Frontend | Vanilla JS + HTML (embedded, no build step) |
| Deployment | Railway (backend) |
| Prototype | Google Colab |

---

## Local setup

```bash
git clone https://github.com/YOUR_USERNAME/strategypulse-ai
cd strategypulse-ai
pip install -r requirements.txt
python app.py
# Open http://localhost:8000
```

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `NEWSAPI_KEY` | Optional | Live news data. Get free key at newsapi.org |
| `OPENAI_API_KEY` | Optional | GPT-4o-mini brief generation |
| `PORT` | Auto-set | Set by Railway automatically |

Without keys the app runs in demo mode with rich synthetic data.

---

## Deploy to Railway

1. Fork this repo
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Select this repo
4. Add environment variables in Railway dashboard
5. Railway auto-deploys — permanent URL in ~3 minutes

---

*Built in Google Colab. Deployed on Railway.*
