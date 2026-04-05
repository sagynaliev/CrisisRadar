from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
from llm import chat_with_ai, save_to_memory, search_memory

app = FastAPI(title="Crisis Radar API", version="1.0")

# CORS — frontend-пен байланысу үшін
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Модельдерді жүктеу
with open('../models/rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('../models/gb_model.pkl', 'rb') as f:
    gb_model = pickle.load(f)

# Input схемасы
class EarthquakeInput(BaseModel):
    latitude: float
    longitude: float
    depth: float
    month: int

@app.get("/")
def root():
    return {"message": "🌍 Crisis Radar API жұмыс істеп тұр!"}

@app.post("/predict/earthquake")
def predict_earthquake(data: EarthquakeInput):
    features = np.array([[
        data.latitude,
        data.longitude,
        data.depth,
        np.log1p(data.depth),
        abs(data.latitude),
        data.month
    ]])
    
    rf_pred = rf_model.predict(features)[0]
    
    # Confidence interval
    tree_preds = np.array([tree.predict(features)[0] for tree in rf_model.estimators_])
    conf_lower = float(tree_preds.mean() - 1.96 * tree_preds.std())
    conf_upper = float(tree_preds.mean() + 1.96 * tree_preds.std())
    
    return {
        "prediction": round(float(rf_pred), 2),
        "confidence_lower": round(conf_lower, 2),
        "confidence_upper": round(conf_upper, 2),
        "risk_level": "🔴 Жоғары" if rf_pred >= 6.0 else "🟡 Орташа" if rf_pred >= 5.5 else "🟢 Төмен"
    }


class ChatInput(BaseModel):
    message: str
    history: list = []

@app.post("/chat")
async def chat(data: ChatInput):
    # Жадтан іздеу
    memory = search_memory(data.message)
    
    # Контекст қосу
    enhanced_message = data.message
    if memory:
        enhanced_message = f"Алдыңғы талдаулар:\n{memory}\n\nЖаңа сұрақ: {data.message}"
    
    # AI жауабы
    response = chat_with_ai(enhanced_message, data.history)
    
    # Жадқа сақтау
    save_to_memory(data.message, response)
    
    return {"response": response}


@app.get("/earthquake/data")
def get_earthquake_data():
    import pandas as pd
    import numpy as np
    
    try:
        df = pd.read_csv('../notebooks/earthquake_data.csv')
        df = df[['latitude', 'longitude', 'mag', 'place', 'time']].dropna().head(500)
        return {
            "lats": df['latitude'].tolist(),
            "lons": df['longitude'].tolist(),
            "mags": df['mag'].tolist(),
            "places": df['place'].tolist()
        }
    except:
        return {"lats": [], "lons": [], "mags": [], "places": []}
    


# Nuclear + Composite Risk endpoints
@app.get("/nuclear/data")
def get_nuclear_data():
    nuclear_data = {
        'country': ['Russia', 'USA', 'China', 'North Korea', 'Pakistan',
                    'India', 'Israel', 'France', 'UK', 'Iran'],
        'nuclear_warheads': [5889, 5244, 410, 40, 170, 164, 90, 290, 225, 0],
        'risk_score': [85, 45, 65, 95, 80, 60, 70, 30, 25, 88],
        'stability_index': [3.2, 7.8, 5.5, 1.2, 3.8, 5.2, 6.1, 8.5, 8.8, 2.1],
        'latitude': [61.5, 37.1, 35.8, 40.3, 30.3, 20.5, 31.0, 46.2, 55.3, 32.4],
        'longitude': [105.3, -95.7, 104.1, 127.5, 69.3, 78.9, 34.8, 2.2, -3.4, 53.6],
        'composite_score': [86.6, 43.6, 65.1, 101.6, 81.1, 64.1, 64.6, 32.6, 28.6, 93.6]
    }
    return nuclear_data

@app.get("/news/risk")
async def get_news_risk():
    import requests as req
    NEWS_API_KEY = "18b01397d6624e468325daa4e844b5d8"
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": "nuclear war OR earthquake OR WW3 OR catastrophe",
            "apiKey": NEWS_API_KEY,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 10
        }
        response = req.get(url, params=params)
        articles = response.json().get('articles', [])
        return {
            "articles": [
                {
                    "title": a['title'],
                    "source": a['source']['name'],
                    "published": a['publishedAt'],
                    "url": a['url']
                } for a in articles
            ]
        }
    except:
        return {"articles": []}
    
@app.get("/conflict/data")
def get_conflict_data():
    return {
        'country': ['Ukraine', 'Sudan', 'Myanmar', 'Ethiopia', 'Syria',
                   'Yemen', 'Somalia', 'Mali', 'Nigeria', 'Afghanistan'],
        'conflict_events': [15420, 8930, 6720, 5430, 4210, 3890, 2340, 1870, 4560, 3210],
        'fatalities': [45230, 12400, 8900, 6700, 5400, 8900, 3400, 2100, 5600, 4300],
        'risk_score': [95, 85, 80, 75, 88, 82, 70, 65, 72, 78],
        'latitude': [49.0, 15.5, 17.0, 9.1, 34.8, 15.5, 5.1, 17.5, 9.0, 33.9],
        'longitude': [31.0, 32.5, 96.0, 40.4, 38.9, 48.5, 46.2, -1.5, 8.6, 67.7]
    }

@app.get("/emdat/data")
def get_emdat_data():
    import urllib.parse
    import requests as req
    
    try:
        name = "Natural disasters (EMDAT)"
        encoded = urllib.parse.quote(name)
        url = f"https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/{encoded}/{encoded}.csv"
        
        df = pd.read_csv(url)
        df_recent = df[df['Year'] >= 2000]
        
        disaster_deaths = {
            'Earthquake': float(df_recent['deaths_earthquake'].sum()),
            'Flood': float(df_recent['deaths_flood'].sum()),
            'Storm': float(df_recent['deaths_storm'].sum()),
            'Drought': float(df_recent['deaths_drought'].sum()),
            'Wildfire': float(df_recent['deaths_wildfire'].sum()),
            'Landslide': float(df_recent['deaths_landslide'].sum()),
            'Volcanic': float(df_recent['deaths_volcanic'].sum()),
            'Extreme Temp': float(df_recent['deaths_temperature'].sum()),
        }
        
        disaster_damages = {
            'Earthquake': float(df_recent['total_damages_earthquake'].sum()),
            'Flood': float(df_recent['total_damages_flood'].sum()),
            'Storm': float(df_recent['total_damages_storm'].sum()),
            'Drought': float(df_recent['total_damages_drought'].sum()),
            'Wildfire': float(df_recent['total_damages_wildfire'].sum()),
        }
        
        return {
            'disaster_type': list(disaster_deaths.keys()),
            'total_deaths': list(disaster_deaths.values()),
            'events': [450, 320, 280, 150, 200, 45, 30, 180],
            'affected_millions': [120, 45, 67, 890, 12, 3, 1.5, 34],
            'economic_loss_billion': list(disaster_damages.values()) + [0, 0, 0]
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/conflict/gdelt")
def get_gdelt_data():
    import zipfile, io
    import requests as req
    from datetime import datetime, timedelta
    
    # Try last 3 days
    for days_back in range(1, 4):
        try:
            date = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
            url = f"http://data.gdeltproject.org/events/{date}.export.CSV.zip"
            
            r = req.get(url, timeout=60)
            if r.status_code != 200:
                continue
                
            z = zipfile.ZipFile(io.BytesIO(r.content))
            csv_name = z.namelist()[0]
            
            df = pd.read_csv(z.open(csv_name), sep='\t', header=None, low_memory=False)
            
            df['lat'] = pd.to_numeric(df[46], errors='coerce')
            df['lon'] = pd.to_numeric(df[47], errors='coerce')
            df['goldstein'] = pd.to_numeric(df[30], errors='coerce')
            df['num_mentions'] = pd.to_numeric(df[31], errors='coerce')
            df['country'] = df[44].fillna('Unknown')
            
            df = df.dropna(subset=['lat', 'lon', 'goldstein'])
            df = df[df['lat'].between(-90, 90) & df['lon'].between(-180, 180)]
            df_conflict = df[df['goldstein'] < 0]
            
            if len(df_conflict) == 0:
                continue
                
            df_sample = df_conflict.sample(min(2000, len(df_conflict)))
            
            return {
                'lats': df_sample['lat'].tolist(),
                'lons': df_sample['lon'].tolist(),
                'goldstein': df_sample['goldstein'].tolist(),
                'num_mentions': df_sample['num_mentions'].fillna(1).tolist(),
                'country': df_sample['country'].tolist(),
                'total_events': len(df_conflict),
                'date': date
            }
        except Exception as e:
            continue
    
    return {"error": "GDELT unavailable", "lats": [], "lons": [], "goldstein": [], "num_mentions": [], "country": []}

@app.get("/risk/composite")
def get_composite_risk():
    return {
        'risk_category': ['Climate Disaster', 'Major Earthquake', 'WW3 / Conflict',
                         'Cyber Attack', 'Tsunami', 'Pandemic', 'Nuclear War', 'Nuclear Accident'],
        'composite': [70.0, 64.0, 61.0, 58.0, 57.0, 55.0, 54.0, 50.0]
    }