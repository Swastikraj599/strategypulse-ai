# StrategyPulse AI

> Real-time Competitive Intelligence & Strategic Signal Detection Platform

[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace%20Spaces-FFD166?style=for-the-badge&logo=huggingface&logoColor=black)](https://swastikraj-strategypulse-ai.hf.space)
[![Built With](https://img.shields.io/badge/Built%20With-PyTorch%20%7C%20HuggingFace%20%7C%20FastAPI-4f8ef7?style=for-the-badge)](https://swastikraj-strategypulse-ai.hf.space)
[![Prototype](https://img.shields.io/badge/Prototype-Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=black)](https://colab.research.google.com)

---

## What It Does

StrategyPulse AI monitors 6 Fortune 500 companies — Apple, Microsoft, Google, Amazon, Meta, and Tesla in real time — ingesting signals across news, SEC filings, and research papers — and uses AI to detect strategic shifts before they become public knowledge.

This is the work McKinsey's Strategy Practice gets paid $500K+ per engagement to do manually. StrategyPulse automates the signal detection, classification, and brief generation layer entirely.

---

## Live Demo

**[https://swastikraj-strategypulse-ai.hf.space](https://swastikraj-strategypulse-ai.hf.space)**

![StrategyPulse AI Dashboard](https://img.shields.io/badge/Status-Live-00e5a0?style=flat-square)

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DATA INGESTION                        │
│         NewsAPI · SEC EDGAR · arXiv · USPTO             │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                    AI PIPELINE                           │
│                                                          │
│   Zero-Shot NLP            FinBERT Sentiment             │
│   BART-large-mnli          ProsusAI/finbert              │
│   Strategy categories      Pos / Neg / Neutral           │
│                                                          │
│          Porter's Five Forces Mapper                     │
│          Urgency Scorer · Signal Tagger                  │
└───────────────────┬─────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────┐
│              INTELLIGENCE LAYER                          │
│                                                          │
│   ChromaDB Vector Store · Semantic Retrieval             │
│   LSTM Autoencoder · Anomaly Detection                   │
│   RAG-powered Strategy Brief Generator                   │
└───────────────────┬─────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────┐
│              DELIVERY LAYER                              │
│                                                          │
│   FastAPI REST API · WebSocket Live Feed                 │
│   Luxury Intelligence Dashboard                          │
│   Deployed on Hugging Face Spaces                        │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **NLP Classification** | `facebook/bart-large-mnli` — zero-shot strategy categorisation |
| **Financial Sentiment** | `ProsusAI/finbert` — positive / negative / neutral |
| **Embeddings** | `all-MiniLM-L6-v2` — sentence-transformers |
| **Vector Store** | ChromaDB — semantic retrieval for brief generation |
| **Anomaly Detection** | PyTorch LSTM Autoencoder — 60-day rolling signal volume |
| **Backend** | FastAPI + WebSocket — REST API + real-time streaming |
| **Frontend** | Vanilla JS + HTML — embedded, no build step required |
| **Deployment** | Hugging Face Spaces (Docker) |
| **Prototype** | Google Colab — full pipeline notebook |

---

## Features

**Intelligence Brief Tab**
- Executive summary generated per company from retrieved signals
- KPI cards — risk level, confidence score, signal count, action count
- Key signals with urgency classification and strategic implication
- Recommended actions with timeline and priority tags
- Strategic risks and opportunities panel

**Signal Feed Tab**
- Full filterable table of all ingested signals
- Filter by source (News / SEC / Research) and sentiment
- Sorted by urgency score descending

**Porter's Analysis Tab**
- Interactive radar chart — Porter's Five Forces pentagon
- Per-force assessment and score (1–5) derived from signal patterns
- Colour-coded risk scoring

**Anomaly Map Tab**
- LSTM autoencoder anomaly detection across 60-day signal volume
- Per-company risk flags with visual indicators
- Statistical explanation of anomalies as leading indicators

**Live Signal Feed (right panel)**
- WebSocket-powered stream — new signals every 9 seconds
- Toast notifications for incoming signals
- Anomaly risk index bar chart across all 6 companies

---

## Signal Sources

| Source | Data | Mode |
|--------|------|------|
| NewsAPI | Strategic news articles | Live (with key) |
| SEC EDGAR | 8-K, 10-Q filings | Live (free, no key) |
| arXiv | Research papers | Live (free, no key) |
| Synthetic | Rich demo data | Always available |

---

## Run Locally

```bash
# Clone
git clone https://github.com/Swastikraj599/strategypulse-ai
cd strategypulse-ai

# Install
pip install -r requirements.txt

# Run — demo mode, no keys needed
python app.py

# Open
http://localhost:7860
```

**Optional environment variables:**
```
NEWSAPI_KEY=your_key       # Live news — newsapi.org (free tier)
OPENAI_API_KEY=your_key    # GPT-4o-mini brief enhancement (optional)
```

---

## Colab Prototype

Built and tested cell by cell in Google Colab before deployment:

```
Cell 1 — Install dependencies
Cell 2 — API keys & configuration
Cell 3 — Data ingestion (NewsAPI · SEC EDGAR · arXiv)
Cell 4 — AI pipeline (FinBERT + zero-shot classification)
Cell 5 — Vector store & semantic retrieval (ChromaDB)
Cell 6 — Strategy brief generator (RAG)
Cell 7 — Anomaly detection (LSTM autoencoder)
Cell 8 — FastAPI backend + embedded dashboard + deployment
```

---

## Business Context

This project automates a specific slice of McKinsey-style competitive intelligence:

- **Signal detection** — surface strategic moves before analyst reports cover them
- **Classification** — map every signal to a strategy category and Porter's force automatically
- **Anomaly flagging** — signal clusters within a 5-day window historically precede earnings surprises and acquisition announcements by 3–14 days
- **Brief generation** — structured intelligence reports in seconds vs. 3–4 analyst days manually

---

## Project Structure

```
strategypulse-ai/
├── app.py            — FastAPI application, full backend + API
├── dashboard.html    — Intelligence terminal UI, embedded frontend
├── requirements.txt  — Python dependencies
├── Dockerfile        — Container config for HuggingFace Spaces
├── README.md         — This file
└── .gitignore
```

---

## Author

**Swastikraj** 

Built entirely in Google Colab. Deployed on Hugging Face Spaces.

[![Live App](https://img.shields.io/badge/Live%20App-swastikraj--strategypulse--ai.hf.space-FFD166?style=flat-square&logo=huggingface&logoColor=black)](https://swastikraj-strategypulse-ai.hf.space)
[![GitHub](https://img.shields.io/badge/GitHub-Swastikraj599-4f8ef7?style=flat-square&logo=github)](https://github.com/Swastikraj599/strategypulse-ai)
