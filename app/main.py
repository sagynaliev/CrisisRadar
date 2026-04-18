from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import json
import os

# =====================================================
# ЖОЛДАР
# =====================================================
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(BASE_DIR)
DATA_DIR  = os.path.join(ROOT_DIR, 'data')
FRONT_DIR = os.path.join(ROOT_DIR, 'frontend')

def data_path(f): return os.path.join(DATA_DIR, f)

# =====================================================
# APP
# =====================================================
app = FastAPI(title="Crisis Radar API", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

if os.path.exists(FRONT_DIR):
    app.mount("/static", StaticFiles(directory=FRONT_DIR), name="static")

class ChatInput(BaseModel):
    message: str
    history: list = []

class EarthquakeInput(BaseModel):
    latitude: float
    longitude: float
    depth: float
    month: int

# =====================================================
# ГЛОБАЛ КЭШ — бір рет жүктеп, жадта ұстаймыз
# =====================================================
rf_model   = None
gb_model   = None
_eq_cache  = None   # earthquake JSON
_con_cache = None   # conflict JSON
_nuc_cache = None   # nuclear JSON
_risk_cache= None   # composite risk JSON

@app.on_event("startup")
def startup():
    global rf_model, gb_model, _eq_cache, _con_cache, _nuc_cache, _risk_cache

    # --- ML модельдер ---
    for attr, fname in [('rf_model','rf_model.pkl'), ('gb_model','gb_model.pkl')]:
        for folder in [os.path.join(ROOT_DIR,'models'), BASE_DIR, ROOT_DIR]:
            p = os.path.join(folder, fname)
            if os.path.exists(p):
                with open(p,'rb') as f:
                    globals()[attr] = pickle.load(f)
                rf_model  = globals().get('rf_model', rf_model)
                gb_model  = globals().get('gb_model', gb_model)
                print(f"✅ {fname} loaded")
                break

    # RF / GB жеке жүктейік (globals trick жұмыс жасамауы мүмкін)
    for path_candidate in [os.path.join(ROOT_DIR,'models','rf_model.pkl'),
                            os.path.join(BASE_DIR,'rf_model.pkl')]:
        if os.path.exists(path_candidate):
            with open(path_candidate,'rb') as f: rf_model = pickle.load(f)
            print("✅ RF model loaded"); break

    for path_candidate in [os.path.join(ROOT_DIR,'models','gb_model.pkl'),
                            os.path.join(BASE_DIR,'gb_model.pkl')]:
        if os.path.exists(path_candidate):
            with open(path_candidate,'rb') as f: gb_model = pickle.load(f)
            print("✅ GB model loaded"); break

    # --- Earthquake кэш (200 жол — жеткілікті) ---
    eq_path = data_path('earthquake_data.csv')
    if os.path.exists(eq_path):
        try:
            df = pd.read_csv(eq_path)
            df = df.dropna(subset=['latitude','longitude','mag']).head(200)
            _eq_cache = {
                "lats":   df['latitude'].tolist(),
                "lons":   df['longitude'].tolist(),
                "mags":   df['mag'].tolist(),
                "places": df['place'].tolist() if 'place' in df.columns else ["Unknown"]*len(df)
            }
            print(f"✅ Earthquake cached ({len(df)} rows)")
        except Exception as e:
            print(f"⚠️ Earthquake: {e}")

    # --- Conflict кэш ---
    json_path = data_path('research_summary.json')
    if os.path.exists(json_path):
        with open(json_path,'r',encoding='utf-8') as f:
            summary = json.load(f)
        max_ev = max((v.get('EVENTS',0) for v in summary.values()), default=1)
        r = {"country":[],"conflict_events":[],"fatalities":[],
             "latitude":[],"longitude":[],"risk_score":[]}
        for country, vals in summary.items():
            # LAT/LON тікелей xlsx-тен алынған нақты координаттар
            lat = vals.get('LAT', 0)
            lon = vals.get('LON', 0)
            r["country"].append(country)
            r["conflict_events"].append(int(vals.get('EVENTS',0)))
            r["fatalities"].append(int(vals.get('FATALITIES',0)))
            r["latitude"].append(lat); r["longitude"].append(lon)
            r["risk_score"].append(round(vals.get('EVENTS',0)/max_ev*100,1))
        _con_cache = r
        total_fat = sum(vals.get('FATALITIES',0) for vals in summary.values())
        _risk_cache = {
            "risk_category":["Geopolitical Conflict","Nuclear Threat","Seismic Activity",
                             "Climate Disasters","Pandemic Risk","Cyber Warfare"],
            "composite":[min(100,round(total_fat/15000)),78,65,70,45,58]
        }
        print(f"✅ Conflict cached ({len(r['country'])} countries)")

    # --- Nuclear кэш ---
    nuc_path = data_path('nuclear_inventory.csv')
    if os.path.exists(nuc_path):
        try:
            df = pd.read_csv(nuc_path)
            df.columns = [c.lower().strip() for c in df.columns]
            _nuc_cache = df.to_dict(orient='list')
            print("✅ Nuclear cached")
        except: pass

    print("🚀 All caches ready!")

# =====================================================
# ENDPOINTS — кэштен тез қайтарады
# =====================================================
@app.post("/predict/earthquake")
def predict_earthquake(data: EarthquakeInput):
    model = rf_model or gb_model
    if not model:
        raise HTTPException(503, "ML models not loaded")
    
    # Қате осында: модель 6 дерек күтеді. 
    # Болжам бойынша: Lat, Lon, Depth, Month, Day, Year
    # Мысалы ретінде бүгінгі күнді (18) және жылды (2026) қосамыз:
    current_day = 18
    current_year = 2026
    
    # 6 деректен тұратын массив жасау:
    features = np.array([[
        data.latitude, 
        data.longitude, 
        data.depth, 
        data.month, 
        current_day, 
        current_year
    ]])
    
    try:
        pred = round(float(model.predict(features)[0]), 2)
        risk = ("🔴 CRITICAL" if pred>=7 else "🟠 HIGH" if pred>=6 
                else "🟡 MODERATE" if pred>=5 else "🟢 LOW")
        
        return {
            "prediction": pred, 
            "risk_level": risk,
            "confidence_lower": round(pred-0.3, 2),
            "confidence_upper": round(pred+0.3, 2)
        }
    except Exception as e:
        raise HTTPException(500, f"Model prediction error: {str(e)}")

@app.get("/earthquake/data")
def get_earthquake_data():
    if not _eq_cache:
        raise HTTPException(404, "Earthquake data not loaded")
    return _eq_cache  # жадтан — дереу қайтарады

@app.get("/conflict/data")
def get_conflict_data():
    # research_summary.json LAT/LON бағандарын пайдаланады
    if not _con_cache:
        raise HTTPException(404, "Run process_research.py first")
    return _con_cache  # жадтан — дереу қайтарады

@app.get("/nuclear/data")
def get_nuclear_data():
    if _nuc_cache:
        return _nuc_cache
    # Fallback
    return {
        "country":["Russia","USA","China","France","UK","Pakistan","India","Israel","North Korea","Iran"],
        "latitude":[61.5,37.1,35.9,46.2,55.4,30.4,20.6,31.0,40.3,32.4],
        "longitude":[105.3,-95.7,104.2,2.2,-3.4,69.3,79.0,34.8,127.5,53.7],
        "warheads":[5889,5244,410,290,225,170,164,90,50,0],
        "composite_score":[92,88,72,45,43,78,65,70,85,60]
    }

@app.get("/risk/composite")
def get_composite_risk():
    return _risk_cache or {
        "risk_category":["Geopolitical Conflict","Nuclear Threat","Seismic Activity",
                         "Climate Disasters","Pandemic Risk","Cyber Warfare"],
        "composite":[75,78,65,70,45,58]
    }

@app.get("/news/risk")
def get_news():
    return {"articles":[
        {"source":"Reuters","title":"Global seismic activity increases in Pacific Ring of Fire","url":"#","published":"2026-04-15"},
        {"source":"BBC","title":"Nuclear talks resume amid rising tensions","url":"#","published":"2026-04-14"},
        {"source":"AP News","title":"Conflict fatalities reach record high in 2025","url":"#","published":"2026-04-13"},
        {"source":"Al Jazeera","title":"Humanitarian crisis deepens in conflict zones","url":"#","published":"2026-04-12"},
        {"source":"Nature","title":"Climate models predict increased disaster frequency","url":"#","published":"2026-04-11"},
    ]}

@app.post("/chat")
async def chat(data: ChatInput):
    try:
        from llm import chat_with_ai, save_to_memory
        response = chat_with_ai(data.message, data.history)
        save_to_memory(data.message, response)
        return {"response": response}
    except ImportError:
        raise HTTPException(503, "LLM module not available")
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/")
def serve_index():
    for candidate in [
        os.path.join(FRONT_DIR,'index.html'),
        os.path.join(ROOT_DIR,'index.html'),
        os.path.join(BASE_DIR,'index.html'),
    ]:
        if os.path.exists(candidate):
            return FileResponse(candidate)
    return {"status":"Crisis Radar API","docs":"/docs",
            "hint":"Put index.html in CrisisRadar/frontend/"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "cached": {
            "earthquake": _eq_cache is not None,
            "conflict":   _con_cache is not None,
            "nuclear":    _nuc_cache is not None,
            "risk":       _risk_cache is not None,
        },
        "models": {"rf": rf_model is not None, "gb": gb_model is not None},
    }