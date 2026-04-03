import os
from dotenv import load_dotenv # .env-мен жұмыс істеу үшін қажет
from groq import Groq
import chromadb

# 1. .env файлынан деректерді оқимыз
load_dotenv()

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("crisis_radar")

# 2. Тырнақшадағы мәтін емес, айнымалыны қолданамыз
# os.getenv() функциясы .env файлындағы GROQ_API_KEY мәнін алады
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """Сен Crisis Radar платформасының AI анализаторысың.
Сенің міндетің — жаһандық апаттар мен тәуекелдерді талдау:
- Жер сілкінісі (Earthquake)
- Ядролық қауіп (Nuclear Risk)
- Геосаяси қақтығыстар (WW3 Risk)
- Климаттық апаттар (Climate Crisis)

Әрбір жауапта:
1. Нақты анализ бер
2. Тәуекел деңгейін көрсет (🔴 Жоғары / 🟡 Орташа / 🟢 Төмен)
3. Деректерге негізделген қорытынды жаса
4. Қысқа және нақты жауап бер

Жауапты ағылшын тілінде бер."""

def chat_with_ai(user_message: str, history: list = []) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1000,
        messages=messages
    )

    return response.choices[0].message.content

def save_to_memory(query: str, response: str):
    collection.add(
        documents=[f"Q: {query}\nA: {response}"],
        ids=[f"chat_{collection.count()}"]
    )

def search_memory(query: str) -> str:
    if collection.count() == 0:
        return ""
    results = collection.query(query_texts=[query], n_results=min(3, collection.count()))
    return "\n".join(results['documents'][0]) if results['documents'] else ""