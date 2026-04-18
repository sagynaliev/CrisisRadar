import os
import json
import uuid
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')

def data_path(f): return os.path.join(DATA_DIR, f)

# =====================================================
# GROQ
# =====================================================
try:
    from groq import Groq
    _key = os.getenv("GROQ_API_KEY")
    if not _key:
        raise ValueError("No GROQ_API_KEY")
    client = Groq(api_key=_key)
    GROQ_AVAILABLE = True
    print("✅ Groq client ready")
except Exception as e:
    GROQ_AVAILABLE = False
    client = None
    print(f"⚠️ Groq: {e}")

# =====================================================
# ChromaDB
# =====================================================
try:
    import chromadb
    _chroma = chromadb.Client()
    collection = _chroma.get_or_create_collection("crisis_radar")
    CHROMA_OK = True
except:
    collection = None
    CHROMA_OK = False

# =====================================================
# БАРЛЫҚ ДЕРЕКТЕРДІ БІР РЕТ ЖҮКТЕП КЭШ САҚТАУ
# =====================================================
_full_context = None

def load_all_data() -> dict:
    """Барлық деректерді бір рет жүктейді"""
    global _full_context
    if _full_context is not None:
        return _full_context

    ctx = {}

    # 1. CONFLICT деректері (негізгі)
    p = data_path('research_summary.json')
    if os.path.exists(p):
        with open(p, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        # AI-ға тек маңызды бағандарды жіберу (token үнемдеу)
        ctx["conflict_data"] = {
            country: {
                "events": vals.get("EVENTS", 0),
                "fatalities": vals.get("FATALITIES", 0)
            }
            for country, vals in raw.items()
        }
        ctx["conflict_source"] = "ACLED Regional Data 2026"
        ctx["conflict_countries_count"] = len(raw)
        print(f"✅ Conflict data loaded: {len(raw)} countries")

    # 2. EARTHQUAKE деректері
    eq_path = data_path('earthquake_data.csv')
    if os.path.exists(eq_path):
        try:
            df = pd.read_csv(eq_path).dropna(subset=['mag'])
            ctx["earthquake_data"] = {
                "total_events": len(df),
                "max_magnitude": round(float(df['mag'].max()), 2),
                "avg_magnitude": round(float(df['mag'].mean()), 2),
                "high_risk_events": len(df[df['mag'] >= 6.0]),
                "top10_strongest": df.nlargest(10, 'mag')[['place', 'mag']].to_dict(orient='records')
            }
            ctx["earthquake_source"] = "USGS Real-time API"
            print(f"✅ Earthquake data loaded: {len(df)} events")
        except Exception as e:
            print(f"⚠️ Earthquake: {e}")

    # 3. NUCLEAR деректері
    nuc_path = data_path('nuclear_inventory.csv')
    if os.path.exists(nuc_path):
        try:
            df = pd.read_csv(nuc_path)
            df.columns = [c.lower().strip() for c in df.columns]
            ctx["nuclear_data"] = df.head(15).to_dict(orient='records')
            ctx["nuclear_source"] = "FAS/NTI 2025"
        except:
            pass

    # Nuclear fallback
    if "nuclear_data" not in ctx:
        ctx["nuclear_data"] = [
            {"country": "Russia",      "warheads": 5889, "risk_score": 92},
            {"country": "USA",         "warheads": 5244, "risk_score": 88},
            {"country": "North Korea", "warheads": 50,   "risk_score": 85},
            {"country": "Pakistan",    "warheads": 170,  "risk_score": 78},
            {"country": "China",       "warheads": 410,  "risk_score": 72},
            {"country": "India",       "warheads": 164,  "risk_score": 65},
            {"country": "Israel",      "warheads": 90,   "risk_score": 70},
        ]
        ctx["nuclear_source"] = "FAS/NTI 2025 (estimated)"

    _full_context = ctx
    return ctx


def chat_with_ai(user_message: str, history: list = None) -> str:
    if history is None:
        history = []

    if not GROQ_AVAILABLE:
        return "⚠️ AI қолжетімсіз. GROQ_API_KEY .env файлында орнатылған ба?"

    # Барлық деректерді жүктеу (кэштен)
    all_data = load_all_data()

    # Сұрауға байланысты тек керекті деректерді таңдау
    q = query = user_message.lower()
    ctx = {}

    # Conflict — кілт сөз болмаса да негізгі деректер
    if any(w in q for w in ["war","соғыс","conflict","қақтығыс","fight","battle",
                              "kill","өлім","ukraine","russia","gaza","syria",
                              "afghanistan","myanmar","sudan","yemen","fatalities",
                              "қауіп","risk","danger","world","әлем","халық"]):
        ctx["conflict_data"] = all_data.get("conflict_data", {})
        ctx["conflict_source"] = all_data.get("conflict_source", "")
        ctx["total_countries_monitored"] = all_data.get("conflict_countries_count", 0)

    # Earthquake
    if any(w in q for w in ["earthquake","seismic","жер сілкінісі","magnitude",
                              "tremor","richter","quake"]):
        ctx["earthquake_data"] = all_data.get("earthquake_data", {})
        ctx["earthquake_source"] = all_data.get("earthquake_source", "")

    # Nuclear
    if any(w in q for w in ["nuclear","ядролық","warhead","nuke","атом",
                              "missile","бомба","bomb"]):
        ctx["nuclear_data"] = all_data.get("nuclear_data", [])
        ctx["nuclear_source"] = all_data.get("nuclear_source", "")

    # Егер ешбір кілт сөз болмаса — БАРЛЫҚ деректерді бер
    if not ctx:
        ctx = all_data

    # =====================================================
    # SYSTEM PROMPT
    # =====================================================
    system = f"""Сен CrisisRadar платформасының стратегиялық аналитигісің.
Сенің жалғыз ақпарат көзің — төменде берілген нақты деректер (ACLED, USGS, FAS).

ЕРЕЖЕЛЕР:
1. Тек берілген деректерге сүйен — өз ойыңнан сан шығарма
2. Сандарды нақты келтір (мысалы: "Ukraine: 308,984 оқиға, 251,731 қаза")
3. Жауаптың соңында дереккөзін көрсет
4. Пайдаланушы тілінде жауап бер (қазақша/орысша/ағылшынша)
5. Қысқа және нақты бол

ДЕРЕКТЕР:
{json.dumps(ctx, indent=2, ensure_ascii=False, default=str)}
"""

    messages = [{"role": "system", "content": system}]
    messages += (history or [])[-4:]
    messages.append({"role": "user", "content": user_message})

    try:
        r = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.0,
            max_tokens=800
        )
        return r.choices[0].message.content
    except Exception as e:
        return f"Қате: {e}"


def save_to_memory(query: str, response: str):
    if not CHROMA_OK or collection is None:
        return
    try:
        collection.add(
            documents=[f"Q:{query}\nA:{response}"],
            ids=[f"c_{uuid.uuid4().hex[:8]}"]
        )
    except:
        pass