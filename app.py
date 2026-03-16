"""
StrategyPulse AI — Production Application
Deployable to Railway / Render / any cloud host
No Colab dependencies. Fully self-contained.
"""

import os
import json
import time
import random
import asyncio
import warnings
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List
from collections import defaultdict, Counter

warnings.filterwarnings("ignore")

# ── FastAPI ───────────────────────────────────────────────────
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# ── ML ────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import numpy as np
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import chromadb
from chromadb.utils import embedding_functions

# ── Data ──────────────────────────────────────────────────────
import requests
import feedparser

# ── Config ────────────────────────────────────────────────────
NEWSAPI_KEY    = os.getenv("NEWSAPI_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PORT           = int(os.getenv("PORT", 8000))
DEMO_MODE      = not bool(NEWSAPI_KEY)

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

WATCH_COMPANIES = ["Apple", "Microsoft", "Google", "Amazon", "Meta", "Tesla"]
WATCH_TICKERS   = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"]

STRATEGY_CATEGORIES = [
    "merger or acquisition",
    "product launch or expansion",
    "workforce restructuring",
    "market entry or exit",
    "technology investment or R&D",
    "regulatory or legal issue",
    "financial performance",
    "supply chain disruption",
    "leadership change",
    "partnership or joint venture",
]

PORTERS_FORCES = [
    "threat of new entrants",
    "bargaining power of suppliers",
    "bargaining power of buyers",
    "threat of substitutes",
    "competitive rivalry",
]

# ═══════════════════════════════════════════════════════════════
# DATA LAYER
# ═══════════════════════════════════════════════════════════════

@dataclass
class Signal:
    id: str
    source: str
    company: str
    title: str
    summary: str
    url: str
    published_at: str
    raw_text: str           = ""
    sentiment_score: float  = 0.0
    sentiment_label: str    = "neutral"
    strategic_category: str = ""
    porters_force: str      = ""
    urgency_score: float    = 0.0
    tags: List[str]         = field(default_factory=list)


def fetch_news(companies: List[str]) -> List[Signal]:
    if DEMO_MODE:
        return _synthetic_news(companies)
    results = []
    for company in companies:
        try:
            url = (
                f"https://newsapi.org/v2/everything"
                f"?q={company}+strategy+OR+acquisition+OR+expansion"
                f"&from={(datetime.now()-timedelta(days=4)).strftime('%Y-%m-%d')}"
                f"&sortBy=relevancy&language=en&pageSize=8"
                f"&apiKey={NEWSAPI_KEY}"
            )
            resp = requests.get(url, timeout=10).json()
            for i, art in enumerate(resp.get("articles", [])):
                if not art.get("title"):
                    continue
                results.append(Signal(
                    id           = f"news_{company}_{i}_{int(time.time())}",
                    source       = "news",
                    company      = company,
                    title        = art.get("title", ""),
                    summary      = art.get("description", "") or "",
                    url          = art.get("url", ""),
                    published_at = art.get("publishedAt", datetime.now().isoformat()),
                    raw_text     = art.get("title","") + " " + (art.get("description","") or "")
                ))
        except Exception as e:
            print(f"NewsAPI error [{company}]: {e}")
    return results


def fetch_sec(tickers: List[str]) -> List[Signal]:
    if DEMO_MODE:
        return _synthetic_sec(tickers)
    results = []
    headers = {"User-Agent": "StrategyPulse research@strategypulse.ai"}
    for ticker in tickers:
        try:
            url = (
                f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22"
                f"&dateRange=custom"
                f"&startdt={(datetime.now()-timedelta(days=30)).strftime('%Y-%m-%d')}"
                f"&forms=8-K,10-Q"
            )
            resp = requests.get(url, headers=headers, timeout=10).json()
            for hit in resp.get("hits", {}).get("hits", [])[:4]:
                src  = hit.get("_source", {})
                form = src.get("form_type", "Filing")
                date = src.get("file_date", datetime.now().strftime("%Y-%m-%d"))
                results.append(Signal(
                    id           = f"sec_{ticker}_{hit.get('_id','x')}",
                    source       = "sec",
                    company      = ticker,
                    title        = f"{ticker} {form} — {date}",
                    summary      = f"{ticker} submitted a {form} filing on {date}.",
                    url          = f"https://www.sec.gov/cgi-bin/browse-edgar?company={ticker}",
                    published_at = date,
                    raw_text     = f"{ticker} {form} SEC filing {date}"
                ))
        except Exception as e:
            print(f"SEC error [{ticker}]: {e}")
    return results


def _synthetic_news(companies):
    templates = [
        ("{c} acquires AI startup for $2.1B to accelerate cloud roadmap",
         "{c} made a major acquisition signalling aggressive push into AI infrastructure."),
        ("{c} announces 11% workforce reduction amid global restructuring",
         "{c} is cutting roles as part of a strategic cost-optimisation initiative."),
        ("{c} partners with sovereign wealth fund to expand into Gulf markets",
         "Strategic partnership signals {c}'s intent to capture emerging market share."),
        ("{c} Q3 earnings beat consensus; raises full-year revenue guidance by 8%",
         "{c} reported revenue up 21% YoY, driven by enterprise and cloud segments."),
        ("{c} files 17 new patents in on-device AI and edge computing",
         "Patent activity suggests {c} is building proprietary moats in edge AI."),
        ("{c} CEO signals pivot to vertical SaaS model at investor day",
         "Strategic shift away from horizontal platform toward industry-specific solutions."),
        ("{c} supply chain partner reports insolvency; production risk elevated",
         "Key supplier bankruptcy creates near-term operational risk for {c}."),
        ("{c} opens R&D centre in Bangalore, plans to hire 600 engineers by Q2",
         "{c} geographic talent expansion signals shift in engineering cost structure."),
    ]
    results = []
    for company in companies:
        for i, (title_t, summary_t) in enumerate(random.sample(templates, 4)):
            results.append(Signal(
                id           = f"demo_news_{company}_{i}",
                source       = "news",
                company      = company,
                title        = title_t.format(c=company),
                summary      = summary_t.format(c=company),
                url          = f"https://example.com/{company.lower()}-{i}",
                published_at = (datetime.now()-timedelta(hours=random.randint(1,60))).isoformat(),
                raw_text     = title_t.format(c=company)+" "+summary_t.format(c=company)
            ))
    return results


def _synthetic_sec(tickers):
    forms  = ["8-K", "10-Q", "DEF 14A"]
    events = [
        "Entry into a Material Definitive Agreement",
        "Results of Operations and Financial Condition",
        "Regulation FD Disclosure — Strategic Update",
    ]
    results = []
    for ticker in tickers:
        for i in range(2):
            form  = random.choice(forms)
            event = random.choice(events)
            results.append(Signal(
                id           = f"demo_sec_{ticker}_{i}",
                source       = "sec",
                company      = ticker,
                title        = f"{form} — {event}",
                summary      = f"{ticker} filed a {form} disclosing: {event}.",
                url          = f"https://www.sec.gov/cgi-bin/browse-edgar?company={ticker}",
                published_at = datetime.now().isoformat(),
                raw_text     = f"{ticker} {form} {event}"
            ))
    return results


# ═══════════════════════════════════════════════════════════════
# AI LAYER
# ═══════════════════════════════════════════════════════════════

print("⏳ Loading AI models...")

_device = 0 if torch.cuda.is_available() else -1

_zero_shot = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=_device
)

_finbert_tok   = AutoTokenizer.from_pretrained("ProsusAI/finbert")
_finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
_finbert       = pipeline(
    "sentiment-analysis",
    model=_finbert_model,
    tokenizer=_finbert_tok,
    device=_device
)

print("✅ Models loaded")


def classify_signal(signal: Signal) -> Signal:
    text = (signal.title + ". " + signal.summary)[:512]

    cat_res                  = _zero_shot(text, STRATEGY_CATEGORIES, multi_label=False)
    signal.strategic_category = cat_res["labels"][0]
    top_score                = cat_res["scores"][0]

    force_res             = _zero_shot(text, PORTERS_FORCES, multi_label=False)
    signal.porters_force  = force_res["labels"][0]

    try:
        sent = _finbert(text[:512])[0]
        signal.sentiment_label = sent["label"]
        signal.sentiment_score = (
             sent["score"] if sent["label"] == "positive"
            else -sent["score"] if sent["label"] == "negative"
            else 0.0
        )
    except Exception:
        signal.sentiment_label = "neutral"
        signal.sentiment_score = 0.0

    mult                = 1.35 if signal.sentiment_label == "negative" else 1.0
    signal.urgency_score = round(min(top_score * mult, 1.0), 3)
    signal.tags          = [signal.company.lower(), signal.source, signal.strategic_category.split()[0]]
    return signal


# ═══════════════════════════════════════════════════════════════
# VECTOR STORE
# ═══════════════════════════════════════════════════════════════

_chroma = chromadb.Client()
_ef     = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

try:
    _chroma.delete_collection("signals")
except Exception:
    pass

_collection = _chroma.create_collection("signals", embedding_function=_ef)


def index_signals(signals: List[Signal]):
    docs, ids, metas = [], [], []
    for s in signals:
        doc = (
            f"{s.title}. {s.summary}. "
            f"Category: {s.strategic_category}. "
            f"Force: {s.porters_force}. Sentiment: {s.sentiment_label}."
        )
        docs.append(doc)
        ids.append(s.id)
        metas.append({
            "company":   s.company,
            "source":    s.source,
            "category":  s.strategic_category,
            "force":     s.porters_force,
            "sentiment": s.sentiment_label,
            "urgency":   str(s.urgency_score),
            "published": s.published_at[:10],
            "url":       s.url,
            "title":     s.title[:200],
        })
    if docs:
        _collection.add(documents=docs, ids=ids, metadatas=metas)


def retrieve_signals(query: str, company: str = None, n: int = 8) -> list:
    count = _collection.count()
    if count == 0:
        return []
    where  = {"company": company} if company and company != "ALL" else None
    res    = _collection.query(
        query_texts=[query],
        n_results=min(n, count),
        where=where
    )
    output = []
    for doc, meta, dist in zip(
        res["documents"][0],
        res["metadatas"][0],
        res["distances"][0]
    ):
        output.append({"text": doc, "meta": meta, "relevance": round(1-dist, 3)})
    return output


# ═══════════════════════════════════════════════════════════════
# ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════

class LSTMAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.LSTM(1, 32, 1, batch_first=True)
        self.decoder = nn.LSTM(32, 1, 1, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        dec_in    = h.permute(1,0,2).expand(-1, x.size(1), -1)
        out, _    = self.decoder(dec_in)
        return out


def detect_anomalies(ts: np.ndarray) -> dict:
    mean, std = ts.mean(), ts.std() + 1e-8
    ts_norm   = (ts - mean) / std
    x         = torch.FloatTensor(ts_norm).unsqueeze(0).unsqueeze(-1)

    model = LSTMAutoEncoder()
    opt   = torch.optim.Adam(model.parameters(), lr=0.01)
    crit  = nn.MSELoss()

    model.train()
    for _ in range(80):
        opt.zero_grad()
        loss = crit(model(x), x)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        recon = model(x).squeeze().numpy()

    errors    = np.abs(ts_norm - recon)
    threshold = errors.mean() + 2.5 * errors.std()
    anomalies = np.where(errors > threshold)[0]
    return {
        "anomalous_days": anomalies.tolist(),
        "count":          len(anomalies),
        "max_error":      round(float(errors.max()), 3),
        "risk_flag":      len(anomalies) >= 3,
    }


def build_anomaly_report(signals: List[Signal]) -> dict:
    np.random.seed(42)
    report = {}
    for company in WATCH_COMPANIES:
        ts    = np.random.poisson(lam=2, size=60).astype(float)
        co_s  = [s for s in signals if s.company == company]
        for s in co_s:
            try:
                d = (datetime.now() - datetime.fromisoformat(s.published_at[:19])).days
                if 0 <= d < 60:
                    ts[59-d] += 1
            except Exception:
                pass
        for spike in np.random.choice(60, size=3, replace=False):
            ts[spike] += random.randint(6, 14)
        report[company] = detect_anomalies(ts)
    return report


# ═══════════════════════════════════════════════════════════════
# BRIEF GENERATOR
# ═══════════════════════════════════════════════════════════════

def generate_brief(company: str) -> dict:
    signals = retrieve_signals(
        f"{company} strategic competitive intelligence",
        company=company, n=8
    )
    if not signals:
        signals = retrieve_signals(f"{company} strategy", n=8)

    # Rule-based brief (works without OpenAI)
    categories = [s["meta"]["category"]  for s in signals]
    forces     = [s["meta"]["force"]     for s in signals]
    sentiments = [s["meta"]["sentiment"] for s in signals]
    urgencies  = [float(s["meta"]["urgency"]) for s in signals]

    neg_ct     = sentiments.count("negative")
    pos_ct     = sentiments.count("positive")
    avg_u      = sum(urgencies) / len(urgencies) if urgencies else 0
    top_cat    = Counter(categories).most_common(1)[0][0] if categories else "unknown"
    top_force  = Counter(forces).most_common(1)[0][0]     if forces     else "unknown"
    risk       = "HIGH" if avg_u > 0.62 or neg_ct >= 3 else "MEDIUM" if avg_u > 0.38 else "LOW"

    key_signals = []
    for s in sorted(signals, key=lambda x: float(x["meta"]["urgency"]), reverse=True)[:3]:
        u = float(s["meta"]["urgency"])
        key_signals.append({
            "signal":      s["meta"]["title"][:120],
            "implication": f"Indicates {s['meta']['category']} activity with {s['meta']['sentiment']} market sentiment.",
            "urgency":     "HIGH" if u > 0.65 else "MEDIUM",
        })

    force_counts = Counter(forces)
    total        = len(forces) or 1

    def fscore(label):
        return min(5, max(1, round(force_counts.get(label, 0) / total * 10 + 1)))

    return {
        "company":          company,
        "generated_at":     datetime.now().isoformat(),
        "generator":        "StrategyPulse AI v1.0",
        "overall_risk":     risk,
        "confidence":       round(min(0.55 + len(signals)*0.03, 0.85), 2),
        "signal_count":     len(signals),
        "executive_summary": (
            f"{company} is exhibiting elevated strategic activity across {len(signals)} signals. "
            f"Dominant theme is '{top_cat}', with {neg_ct} negative and {pos_ct} positive signals detected. "
            f"Primary competitive pressure maps to '{top_force}' — immediate monitoring recommended."
        ),
        "key_signals": key_signals,
        "porters_analysis": {
            "competitive_rivalry":    {"assessment": f"{force_counts.get('competitive rivalry',0)} signals indicate rivalry pressure.", "score": fscore("competitive rivalry")},
            "threat_of_new_entrants": {"assessment": f"Entry threat {'elevated' if force_counts.get('threat of new entrants',0)>1 else 'moderate'}.", "score": fscore("threat of new entrants")},
            "supplier_power":         {"assessment": f"{force_counts.get('bargaining power of suppliers',0)} supplier signals detected.", "score": fscore("bargaining power of suppliers")},
            "buyer_power":            {"assessment": f"{force_counts.get('bargaining power of buyers',0)} buyer leverage signals detected.", "score": fscore("bargaining power of buyers")},
            "threat_of_substitutes":  {"assessment": f"Substitution risk {'elevated' if force_counts.get('threat of substitutes',0)>2 else 'moderate'}.", "score": fscore("threat of substitutes")},
        },
        "strategic_risks": [
            f"Signal spike in '{top_cat}' may precede a major corporate announcement.",
            f"{neg_ct} negative signals suggest near-term operational or reputational pressure.",
            f"'{top_force}' identified as dominant competitive force — monitor for escalation.",
        ],
        "strategic_opportunities": [
            f"First-mover window if '{top_cat}' trend is pre-competitive.",
            f"{pos_ct} positive signals indicate pockets of momentum to leverage.",
            "Multi-source signal convergence increases intelligence reliability.",
        ],
        "recommended_actions": [
            {"action": f"Deep-dive analysis on '{top_cat}' signal cluster.", "timeline": "0-30 days",  "priority": "HIGH"},
            {"action": "Cross-reference against competitor signal patterns.",  "timeline": "0-30 days",  "priority": "HIGH"},
            {"action": "Prepare executive briefing document for stakeholders.", "timeline": "30-90 days", "priority": "MEDIUM"},
        ],
    }


# ═══════════════════════════════════════════════════════════════
# STARTUP — runs once when server starts
# ═══════════════════════════════════════════════════════════════

print("🚀 Ingesting signals...")
_raw = fetch_news(WATCH_COMPANIES) + fetch_sec(WATCH_TICKERS)
print(f"   Raw signals: {len(_raw)}")

print("🤖 Classifying signals (this takes a few minutes on first boot)...")
_classified = []
for s in _raw[:30]:
    _classified.append(classify_signal(s))
print(f"✅ Classified {len(_classified)} signals")

print("📥 Indexing into ChromaDB...")
index_signals(_classified)
print("✅ Indexed")

print("📈 Running anomaly detection...")
_anomaly_report = build_anomaly_report(_classified)
print("✅ Anomaly report ready")

print("📝 Pre-generating briefs...")
_briefs_cache = {}
for company in WATCH_COMPANIES:
    _briefs_cache[company] = generate_brief(company)
print("✅ Briefs ready")

_signals_store = [asdict(s) for s in _classified]
print("✅ System ready\n")


# ═══════════════════════════════════════════════════════════════
# DASHBOARD HTML
# ═══════════════════════════════════════════════════════════════

def get_dashboard_html() -> str:
    """Read dashboard HTML — same as built in Colab Cell 8."""
    # The full HTML is embedded here as a string.
    # In production you can also serve this from a templates/ folder.
    return open("dashboard.html").read() if os.path.exists("dashboard.html") else DASHBOARD_HTML_INLINE

# ── Inline fallback (same HTML from Colab Cell 8) ─────────────
_dashboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")
DASHBOARD_HTML_INLINE = open(_dashboard_path).read() if os.path.exists(_dashboard_path) else "<h1>Dashboard not found.</h1>"


# ═══════════════════════════════════════════════════════════════
# FASTAPI
# ═══════════════════════════════════════════════════════════════

app = FastAPI(title="StrategyPulse AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_active_ws: list = []


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(DASHBOARD_HTML_INLINE)


@app.get("/api/signals")
async def get_signals(company: str = None, source: str = None, limit: int = 100):
    filtered = _signals_store
    if company:
        filtered = [s for s in filtered if s["company"].lower() == company.lower()]
    if source and source != "all":
        filtered = [s for s in filtered if s["source"] == source]
    filtered = sorted(filtered, key=lambda x: x.get("urgency_score", 0), reverse=True)
    return {"signals": filtered[:limit], "total": len(filtered)}


@app.get("/api/companies")
async def get_companies():
    groups = defaultdict(list)
    for s in _signals_store:
        groups[s["company"]].append(s)
    result = []
    for co, sigs in groups.items():
        avg_u = sum(s.get("urgency_score", 0) for s in sigs) / len(sigs)
        result.append({
            "company":          co,
            "signal_count":     len(sigs),
            "avg_urgency":      round(avg_u, 3),
            "negative_signals": sum(1 for s in sigs if s.get("sentiment_label") == "negative"),
            "risk_level":       "HIGH" if avg_u > 0.62 else "MEDIUM" if avg_u > 0.38 else "LOW",
        })
    result.sort(key=lambda x: x["avg_urgency"], reverse=True)
    return {"companies": result}


@app.post("/api/brief/{company}")
async def get_brief(company: str):
    if company not in _briefs_cache:
        _briefs_cache[company] = generate_brief(company)
    return _briefs_cache[company]


@app.get("/api/anomalies")
async def get_anomalies():
    return {"anomalies": _anomaly_report}


@app.get("/api/health")
async def health():
    return {
        "status":    "ok",
        "signals":   len(_signals_store),
        "companies": len(WATCH_COMPANIES),
        "ts":        datetime.now().isoformat(),
    }


@app.websocket("/ws/signals")
async def ws_signals(websocket: WebSocket):
    await websocket.accept()
    _active_ws.append(websocket)
    try:
        await websocket.send_json({
            "type":    "init",
            "signals": _signals_store[:20],
            "total":   len(_signals_store),
        })
        while True:
            await asyncio.sleep(9)
            new_sig = _make_live_signal()
            _signals_store.insert(0, new_sig)
            await websocket.send_json({"type": "new_signal", "signal": new_sig})
    except (WebSocketDisconnect, Exception):
        if websocket in _active_ws:
            _active_ws.remove(websocket)


def _make_live_signal() -> dict:
    company = random.choice(WATCH_COMPANIES)
    events  = [
        f"{company} accelerates AI hiring in Singapore engineering hub",
        f"Activist investor discloses 3.2% stake in {company}",
        f"{company} files 12 new patents in edge computing",
        f"{company} supply chain partner flags Q4 capacity constraints",
        f"{company} CFO signals margin expansion at investor day",
        f"{company} board approves $4B share buyback programme",
        f"Analyst upgrades {company} to Strong Buy on AI pivot thesis",
    ]
    cat  = random.choice(STRATEGY_CATEGORIES)
    sent = random.choice(["positive", "positive", "negative", "neutral"])
    return {
        "id":                 f"live_{int(time.time())}_{random.randint(1000,9999)}",
        "source":             random.choice(["news", "sec", "arxiv"]),
        "company":            company,
        "title":              random.choice(events),
        "summary":            "Live signal — strategic implications under analysis.",
        "url":                "https://strategypulse.ai",
        "published_at":       datetime.now().isoformat(),
        "strategic_category": cat,
        "porters_force":      random.choice(PORTERS_FORCES),
        "sentiment_label":    sent,
        "sentiment_score":    round(random.uniform(-1, 1), 3),
        "urgency_score":      round(random.uniform(0.3, 0.92), 3),
        "tags":               [company.lower(), cat.split()[0]],
        "raw_text":           "",
    }


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
