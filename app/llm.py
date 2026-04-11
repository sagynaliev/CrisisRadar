import os
import json
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
import chromadb

# .env файлынан GROQ_API_KEY-ді оқимыз
load_dotenv()

# Groq клиенті
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ChromaDB клиенті
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("crisis_radar_memory")

def get_verified_data(query: str):
    """
    Барлық CSV және JSON файлдарынан қажетті контекстті жинайды.
    """
    context = {}
    q = query.lower()
    
    # Файл жолдары (Relative paths)
    data_dir = os.path.join(os.path.dirname(__file__), '../data/')

    # 1. ACLED ЗЕРТТЕУ ДЕРЕКТЕРІ (Соғыстар мен қақтығыстар)
    if any(word in q for word in ["conflict", "war", "соғыс", "қақтығыс", "research", "зерттеу"]):
        summary_path = os.path.join(data_dir, 'research_summary.json')
        if os.path.exists(summary_path):
            with open(summary_path, 'r', encoding='utf-8') as f:
                context["conflict_research"] = json.load(f)
                context["conflict_source"] = "ACLED Regional Aggregated Data (2026)"

    # 2. ЯДРОЛЫҚ ҚАУІП
    if any(word in q for word in ["nuclear", "warhead", "ядролық", "атом"]):
        nuke_path = os.path.join(data_dir, 'nuclear_inventory.csv')
        if os.path.exists(nuke_path):
            try:
                df_nuke = pd.read_csv(nuke_path)
                # AI-ға тым көп дерек жібермеу үшін тек маңызды бағандарды аламыз
                context["nuclear_data"] = df_nuke[['country', 'nuclear_warheads', 'risk_score']].to_dict(orient='records')
                context["nuclear_source"] = "Federation of American Scientists (FAS) / NTI"
            except Exception as e:
                print(f"Nuclear data error: {e}")

    # 3. ЖЕР СІЛКІНІСІ
    if any(word in q for word in ["earthquake", "жер сілкінісі", "магнитуда"]):
        eq_path = os.path.join(data_dir, 'earthquake_data.csv') # Егер notebooks-та болса, жолын түзетіңіз
        if os.path.exists(eq_path):
            try:
                df_eq = pd.read_csv(eq_path)
                context["earthquake_stats"] = {
                    "max_magnitude": float(df_eq['mag'].max()),
                    "recent_count": len(df_eq),
                    "latest_events": df_eq[['place', 'mag', 'time']].head(5).to_dict(orient='records')
                }
                context["earthquake_source"] = "USGS Real-time API"
            except:
                pass

    return context

def chat_with_ai(user_message: str, history: list = []):
    """
    Чат функциясы. Тек файлдан алынған деректерді пайдаланады.
    """
    # 1. Файлдардан деректерді жинау
    verified_context = get_verified_data(user_message)
    
    # 2. Егер контекст бос болса (сұрақ тақырыптан тыс болса)
    if not verified_context:
        return ("Кешіріңіз, жүйеде бұл сұрақ бойынша нақты деректер табылмады. "
                "Мен тек Жер сілкінісі, Ядролық қауіп және Қақтығыстар зерттеуі бойынша жауап бере аламын.")

    # 3. SYSTEM PROMPT - AI-ға қатаң шекара қою
    STRICT_SYSTEM_PROMPT = f"""Сен CrisisRadar жүйесінің стратегиялық аналитигісің.
Сенің ЖАЛҒЫЗ ақпарат көзің — төменде берілген JSON деректері.

ҚАТАҢ ТӘРТІП:
1. Егер сұрақ берілген деректерде ҚАМТЫЛМАҒАН болса, "Мәлімет жоқ" деп жауап бер.
2. Өз ойыңнан немесе ішкі біліміңнен ешқандай сан шығарма (мысалы, оқтұмсық санын тек файлдан ал).
3. Жауаптың соңында міндетті түрде дереккөзді (source) көрсет.
4. Тіл: Пайдаланушы қай тілде сұраса, сол тілде жауап бер.
5. Тон: Кәсіби, нақты және қысқа.

БЕРІЛГЕН ДЕРЕКТЕР (JSON):
{json.dumps(verified_context, indent=2, ensure_ascii=False)}
"""

    messages = [{"role": "system", "content": STRICT_SYSTEM_PROMPT}]
    messages += history[-4:]  # Соңғы 4 хабарламаны ғана жадта ұстау (контекст үзілмеуі үшін)
    messages.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.0, # Галлюцинацияны болдырмау үшін нөлге теңестіреміз
            max_tokens=1000
        )
        
        ai_response = response.choices[0].message.content
        return ai_response

    except Exception as e:
        return f"Жүйелік қате орын алды: {str(e)}"

# Мемори функциялары (ChromaDB)
def save_to_memory(query: str, response: str):
    collection.add(
        documents=[f"Q: {query}\nA: {response}"],
        ids=[f"chat_{collection.count()}"]
    )