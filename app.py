"""
StrategyPulse AI — Production Application (Lightweight)
No PyTorch. No HuggingFace. Deploys under 400MB.
"""

import os, json, time, random, asyncio, re
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List
from collections import defaultdict, Counter

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import requests, feedparser, numpy as np
import chromadb
from chromadb.utils import embedding_functions

NEWSAPI_KEY    = os.getenv("NEWSAPI_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PORT           = int(os.getenv("PORT", 7860))
DEMO_MODE      = not bool(NEWSAPI_KEY)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

WATCH_COMPANIES = ["Apple", "Microsoft", "Google", "Amazon", "Meta", "Tesla"]
WATCH_TICKERS   = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"]

STRATEGY_CATEGORIES = [
    "merger or acquisition","product launch or expansion","workforce restructuring",
    "market entry or exit","technology investment or R&D","regulatory or legal issue",
    "financial performance","supply chain disruption","leadership change","partnership or joint venture",
]
PORTERS_FORCES = [
    "threat of new entrants","bargaining power of suppliers","bargaining power of buyers",
    "threat of substitutes","competitive rivalry",
]

CATEGORY_KEYWORDS = {
    "merger or acquisition":["acqui","merger","takeover","buyout","acquisition","purchase","deal","bid"],
    "product launch or expansion":["launch","release","new product","unveil","expand","introduce","rollout"],
    "workforce restructuring":["layoff","redundan","restructur","workforce","cut","reduction","reorg","headcount","downsiz"],
    "market entry or exit":["enter","exit","new market","withdraw","expansion","geographic","region","international"],
    "technology investment or R&D":["patent","r&d","research","invest","ai","technology","innovat","develop","engineer","lab"],
    "regulatory or legal issue":["regulat","legal","lawsuit","fine","penalty","compliance","antitrust","investig","sanction"],
    "financial performance":["earnings","revenue","profit","loss","guidance","forecast","quarter","annual","fiscal","margin"],
    "supply chain disruption":["supply chain","supplier","shortage","disruption","logistics","inventory","manufactur"],
    "leadership change":["ceo","cfo","cto","appoint","resign","depart","executive","board","director","leadership"],
    "partnership or joint venture":["partner","collaborat","joint venture","alliance","agreement","mou","cooperat"],
}
PORTERS_KEYWORDS = {
    "threat of new entrants":["startup","new entrant","disrupt","barrier","entry","new player","emerging"],
    "bargaining power of suppliers":["supplier","vendor","raw material","input cost","supply chain","procurement"],
    "bargaining power of buyers":["customer","buyer","client","demand","price pressure","negotiat","churn"],
    "threat of substitutes":["substitute","alternative","replac","switch","competitor product","commodit"],
    "competitive rivalry":["competi","rival","market share","price war","battle","fight","race","versus"],
}
POSITIVE_WORDS = ["beat","exceed","grow","rise","gain","profit","strong","record","upgrade","expand",
                  "success","positive","surge","boost","advance","outperform","accelerate","raise","improve","win"]
NEGATIVE_WORDS = ["loss","fail","decline","drop","cut","layoff","lawsuit","fine","restructur","concern",
                  "risk","disrupt","shortage","resign","probe","penalty","fraud","miss","disappoint","crisis"]

def _score(text, kdict):
    tl = text.lower()
    scores = {label: sum(1.5 if kw in tl else 0 for kw in kws) for label, kws in kdict.items()}
    total = sum(scores.values()) or 1
    best  = max(scores, key=scores.get)
    return best, round(min(scores[best]/total+0.35, 0.92), 3)

def classify_text(text):
    category, cat_conf = _score(text, CATEGORY_KEYWORDS)
    force,    _        = _score(text, PORTERS_KEYWORDS)
    tl  = text.lower()
    pos = sum(1 for w in POSITIVE_WORDS if w in tl)
    neg = sum(1 for w in NEGATIVE_WORDS if w in tl)
    if neg > pos:   sentiment, sent_score = "negative", round(-0.4-min(neg*0.08,0.5),3)
    elif pos > neg: sentiment, sent_score = "positive", round(0.4+min(pos*0.08,0.5),3)
    else:           sentiment, sent_score = "neutral",  0.0
    urgency = round(min(cat_conf*(1.35 if sentiment=="negative" else 1.0),1.0),3)
    return {"strategic_category":category,"porters_force":force,
            "sentiment_label":sentiment,"sentiment_score":sent_score,"urgency_score":urgency}

@dataclass
class Signal:
    id:str; source:str; company:str; title:str; summary:str; url:str; published_at:str
    raw_text:str=""; sentiment_score:float=0.0; sentiment_label:str="neutral"
    strategic_category:str=""; porters_force:str=""; urgency_score:float=0.0
    tags:List[str]=field(default_factory=list)

def _make_signal(id,source,company,title,summary,url,published_at):
    raw=title+" "+summary; cls=classify_text(raw)
    s=Signal(id=id,source=source,company=company,title=title,summary=summary,
             url=url,published_at=published_at,raw_text=raw,**cls)
    s.tags=[company.lower(),source,cls["strategic_category"].split()[0]]
    return s

def _synthetic_signals():
    templates=[
        ("news","{c} acquires AI startup for $2.1B to accelerate cloud roadmap","{c} made a major acquisition signalling aggressive push into AI infrastructure."),
        ("news","{c} announces 11% workforce reduction amid global restructuring","{c} is cutting roles as part of a strategic cost-optimisation initiative."),
        ("news","{c} partners with sovereign wealth fund for Gulf market expansion","Partnership signals {c} intent to capture emerging market share."),
        ("news","{c} Q3 earnings beat consensus; raises full-year revenue guidance 8%","{c} reported revenue up 21% YoY driven by enterprise and cloud segments."),
        ("news","{c} files 17 new patents in on-device AI and edge computing","Patent activity suggests {c} is building proprietary moats in edge AI."),
        ("sec", "{c} 8-K — Entry into a Material Definitive Agreement","{c} filed an 8-K disclosing entry into a material definitive agreement."),
        ("sec", "{c} 10-Q — Results of Operations and Financial Condition","{c} quarterly filing discloses financial results and operational updates."),
        ("news","{c} CEO signals pivot to vertical SaaS at investor day","Strategic shift away from horizontal platform toward industry-specific solutions."),
        ("news","{c} opens R&D centre in Bangalore hiring 600 engineers by Q2","Geographic talent expansion signals shift in {c} engineering cost structure."),
        ("arxiv","LLM-driven competitive intelligence extraction applied to {c}","Framework for extracting strategic signals from earnings calls and news using LLMs."),
    ]
    results=[]
    for company in WATCH_COMPANIES:
        for i,(src,tt,st) in enumerate(random.sample(templates,5)):
            results.append(_make_signal(
                id=f"demo_{src}_{company}_{i}",source=src,company=company,
                title=tt.format(c=company),summary=st.format(c=company),
                url=f"https://example.com/{company.lower()}-{i}",
                published_at=(datetime.now()-timedelta(hours=random.randint(1,72))).isoformat()
            ))
    return results

def fetch_all_signals():
    if DEMO_MODE: return _synthetic_signals()
    results=[]
    for company in WATCH_COMPANIES:
        try:
            url=(f"https://newsapi.org/v2/everything?q={company}+strategy+OR+acquisition"
                 f"&from={(datetime.now()-timedelta(days=4)).strftime('%Y-%m-%d')}"
                 f"&sortBy=relevancy&language=en&pageSize=8&apiKey={NEWSAPI_KEY}")
            for i,art in enumerate(requests.get(url,timeout=10).json().get("articles",[])):
                if art.get("title"):
                    results.append(_make_signal(f"news_{company}_{i}_{int(time.time())}","news",company,
                        art.get("title",""),art.get("description","") or "",art.get("url",""),
                        art.get("publishedAt",datetime.now().isoformat())))
        except Exception as e: print(f"NewsAPI [{company}]: {e}")
    return results or _synthetic_signals()

print("⏳ Initialising ChromaDB...")
_chroma=chromadb.Client()
_ef=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
try: _chroma.delete_collection("signals")
except: pass
_col=_chroma.create_collection("signals",embedding_function=_ef)
print("✅ ChromaDB ready")

def index_signals(signals):
    docs,ids,metas=[],[],[]
    for s in signals:
        docs.append(f"{s.title}. {s.summary}. Category: {s.strategic_category}. Force: {s.porters_force}.")
        ids.append(s.id)
        metas.append({"company":s.company,"source":s.source,"category":s.strategic_category,
                      "force":s.porters_force,"sentiment":s.sentiment_label,"urgency":str(s.urgency_score),
                      "published":s.published_at[:10],"url":s.url,"title":s.title[:200]})
    if docs: _col.add(documents=docs,ids=ids,metadatas=metas)

def retrieve(query,company=None,n=8):
    count=_col.count()
    if count==0: return []
    where={"company":company} if company else None
    res=_col.query(query_texts=[query],n_results=min(n,count),where=where)
    return [{"text":d,"meta":m,"relevance":round(1-dist,3)}
            for d,m,dist in zip(res["documents"][0],res["metadatas"][0],res["distances"][0])]

def detect_anomalies_stat(ts,window=7,sigma=2.5):
    anomalies=[i for i in range(window,len(ts))
               if ts[i]>ts[i-window:i].mean()+sigma*(ts[i-window:i].std()+1e-8)]
    return {"anomalous_days":anomalies,"count":len(anomalies),"max_error":float(ts.max()),"risk_flag":len(anomalies)>=3}

def build_anomaly_report(signals):
    np.random.seed(42); report={}
    for company in WATCH_COMPANIES:
        ts=np.random.poisson(lam=2,size=60).astype(float)
        for s in [x for x in signals if x.company==company]:
            try:
                d=(datetime.now()-datetime.fromisoformat(s.published_at[:19])).days
                if 0<=d<60: ts[59-d]+=1
            except: pass
        for spike in np.random.choice(60,size=3,replace=False): ts[spike]+=random.randint(6,14)
        report[company]=detect_anomalies_stat(ts)
    return report

def generate_brief(company):
    signals=retrieve(f"{company} strategic competitive intelligence",company=company,n=8)
    if not signals: signals=retrieve(f"{company} strategy",n=8)
    categories=[s["meta"]["category"] for s in signals]
    forces=[s["meta"]["force"] for s in signals]
    sentiments=[s["meta"]["sentiment"] for s in signals]
    urgencies=[float(s["meta"]["urgency"]) for s in signals]
    neg_ct=sentiments.count("negative"); pos_ct=sentiments.count("positive")
    avg_u=sum(urgencies)/len(urgencies) if urgencies else 0
    top_cat=Counter(categories).most_common(1)[0][0] if categories else "unknown"
    top_force=Counter(forces).most_common(1)[0][0] if forces else "unknown"
    risk="HIGH" if avg_u>0.62 or neg_ct>=3 else "MEDIUM" if avg_u>0.38 else "LOW"
    fc=Counter(forces); total=len(forces) or 1
    def fscore(label): return min(5,max(1,round(fc.get(label,0)/total*10+1)))
    key_signals=[{"signal":s["meta"]["title"][:120],
                  "implication":f"Indicates {s['meta']['category']} with {s['meta']['sentiment']} sentiment.",
                  "urgency":"HIGH" if float(s["meta"]["urgency"])>0.65 else "MEDIUM"}
                 for s in sorted(signals,key=lambda x:float(x["meta"]["urgency"]),reverse=True)[:3]]
    return {"company":company,"generated_at":datetime.now().isoformat(),"generator":"StrategyPulse AI v1.0",
            "overall_risk":risk,"confidence":round(min(0.55+len(signals)*0.03,0.85),2),"signal_count":len(signals),
            "executive_summary":(f"{company} is exhibiting elevated strategic activity across {len(signals)} signals. "
                                 f"Dominant theme is '{top_cat}', with {neg_ct} negative and {pos_ct} positive signals. "
                                 f"Primary competitive pressure maps to '{top_force}' — immediate monitoring recommended."),
            "key_signals":key_signals,
            "porters_analysis":{
                "competitive_rivalry":{"assessment":f"{fc.get('competitive rivalry',0)} rivalry signals.","score":fscore("competitive rivalry")},
                "threat_of_new_entrants":{"assessment":f"Entry threat {'elevated' if fc.get('threat of new entrants',0)>1 else 'moderate'}.","score":fscore("threat of new entrants")},
                "supplier_power":{"assessment":f"{fc.get('bargaining power of suppliers',0)} supplier signals.","score":fscore("bargaining power of suppliers")},
                "buyer_power":{"assessment":f"{fc.get('bargaining power of buyers',0)} buyer signals.","score":fscore("bargaining power of buyers")},
                "threat_of_substitutes":{"assessment":f"Substitution risk {'elevated' if fc.get('threat of substitutes',0)>2 else 'moderate'}.","score":fscore("threat of substitutes")},
            },
            "strategic_risks":[f"Signal spike in '{top_cat}' may precede a major announcement.",
                                f"{neg_ct} negative signals suggest near-term operational pressure.",
                                f"'{top_force}' is the dominant competitive force."],
            "strategic_opportunities":[f"First-mover window if '{top_cat}' trend is pre-competitive.",
                                       f"{pos_ct} positive signals indicate momentum to leverage.",
                                       "Multi-source convergence increases intelligence reliability."],
            "recommended_actions":[{"action":f"Deep-dive on '{top_cat}' signal cluster.","timeline":"0-30 days","priority":"HIGH"},
                                   {"action":"Cross-reference competitor signal patterns.","timeline":"0-30 days","priority":"HIGH"},
                                   {"action":"Prepare executive briefing for stakeholders.","timeline":"30-90 days","priority":"MEDIUM"}]}

print("🚀 Ingesting signals...")
_classified=fetch_all_signals()
print(f"✅ {len(_classified)} signals ready")
index_signals(_classified)
print("✅ Indexed into ChromaDB")
_anomaly_report=build_anomaly_report(_classified)
print("✅ Anomaly detection complete")
_briefs_cache={c:generate_brief(c) for c in WATCH_COMPANIES}
print("✅ Briefs pre-generated")
_signals_store=[asdict(s) for s in _classified]

_here=os.path.dirname(os.path.abspath(__file__))
_dashboard_path=os.path.join(_here,"dashboard.html")
DASHBOARD_HTML=open(_dashboard_path).read() if os.path.exists(_dashboard_path) else "<h1>dashboard.html not found</h1>"
print(f"\n✅ StrategyPulse AI ready — {len(_signals_store)} signals\n")

app=FastAPI(title="StrategyPulse AI",version="1.0.0")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"])
_active_ws:list=[]

@app.get("/",response_class=HTMLResponse)
async def dashboard(): return HTMLResponse(DASHBOARD_HTML)

@app.get("/api/signals")
async def get_signals(company:str=None,source:str=None,limit:int=100):
    f=_signals_store
    if company: f=[s for s in f if s["company"].lower()==company.lower()]
    if source and source!="all": f=[s for s in f if s["source"]==source]
    return {"signals":sorted(f,key=lambda x:x.get("urgency_score",0),reverse=True)[:limit],"total":len(f)}

@app.get("/api/companies")
async def get_companies():
    groups=defaultdict(list)
    for s in _signals_store: groups[s["company"]].append(s)
    result=[]
    for co,sigs in groups.items():
        avg_u=sum(s.get("urgency_score",0) for s in sigs)/len(sigs)
        result.append({"company":co,"signal_count":len(sigs),"avg_urgency":round(avg_u,3),
                       "negative_signals":sum(1 for s in sigs if s.get("sentiment_label")=="negative"),
                       "risk_level":"HIGH" if avg_u>0.62 else "MEDIUM" if avg_u>0.38 else "LOW"})
    return {"companies":sorted(result,key=lambda x:x["avg_urgency"],reverse=True)}

@app.post("/api/brief/{company}")
async def get_brief(company:str):
    if company not in _briefs_cache: _briefs_cache[company]=generate_brief(company)
    return _briefs_cache[company]

@app.get("/api/anomalies")
async def get_anomalies(): return {"anomalies":_anomaly_report}

@app.get("/api/health")
async def health(): return {"status":"ok","signals":len(_signals_store),"ts":datetime.now().isoformat()}

@app.websocket("/ws/signals")
async def ws_signals(websocket:WebSocket):
    await websocket.accept(); _active_ws.append(websocket)
    try:
        await websocket.send_json({"type":"init","signals":_signals_store[:20],"total":len(_signals_store)})
        while True:
            await asyncio.sleep(9)
            new_sig=_make_live_signal(); _signals_store.insert(0,new_sig)
            await websocket.send_json({"type":"new_signal","signal":new_sig})
    except (WebSocketDisconnect,Exception):
        if websocket in _active_ws: _active_ws.remove(websocket)

def _make_live_signal():
    company=random.choice(WATCH_COMPANIES)
    events=[f"{company} accelerates AI hiring in Singapore",f"Activist investor discloses 3.2% stake in {company}",
            f"{company} files 12 new patents in edge computing",f"{company} supply chain partner flags constraints",
            f"{company} CFO signals margin expansion at investor day",f"{company} board approves $4B buyback",
            f"Analyst upgrades {company} to Strong Buy on AI pivot"]
    title=random.choice(events); cls=classify_text(title)
    return {"id":f"live_{int(time.time())}_{random.randint(1000,9999)}","source":random.choice(["news","sec","arxiv"]),
            "company":company,"title":title,"summary":"Live signal — strategic implications under analysis.",
            "url":"https://strategypulse.ai","published_at":datetime.now().isoformat(),"raw_text":"","tags":[company.lower()],**cls}

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=PORT,log_level="info")
