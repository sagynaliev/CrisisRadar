from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import requests as req
from llm import chat_with_ai, save_to_memory, build_strict_prompt

app = FastAPI(title="Crisis Radar API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --- ДИНАМИКАЛЫҚ ДЕРЕКТЕРДІ АЛУ ФУНКЦИЯЛАРЫ ---

def load_earthquake_df():
    try:
        # Статикалық тізім емес, файлдан оқимыз
        return pd.read_csv('../notebooks/earthquake_data.csv')
    except Exception:
        return None

def load_nuclear_data_from_file():
    try:
        # Ядролық деректерді де бөлек JSON немесе CSV-ге шығарғаныңыз жөн
        # Қазірше файл жоқ болса, бос қайтарады
        return pd.read_json('../data/nuclear_inventory.json')
    except:
        return None

# --- API ENDPOINTS ---

@app.get("/earthquake/data")
def get_earthquake_data():
    df = load_earthquake_df()
    if df is not None:
        df = df.dropna().head(500)
        return {
            "lats": df['latitude'].tolist(),
            "lons": df['longitude'].tolist(),
            "mags": df['mag'].tolist(),
            "places": df['place'].tolist()
        }
    return {"error": "Data file not found"}

@app.get("/nuclear/data")
def get_nuclear_data():
    try:
        # Файлдан оқып, бірден frontend-ке жіберу
        df = pd.read_csv('../data/nuclear_inventory.csv')
        return df.to_dict(orient='list')
    except:
        raise HTTPException(status_code=404, detail="Data file not found")

@app.post("/chat")
async def chat(data: ChatInput):
    # build_strict_prompt функциясы llm.py ішінде файлдарды өзі оқиды
    prompt = build_strict_prompt(data.message)
    
    if prompt is None:
        return {"response": "No verified data available for this query in the system files."}
    
    try:
        response = chat_with_ai(prompt, data.history)
        save_to_memory(data.message, response)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))