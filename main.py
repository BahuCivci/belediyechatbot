import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Gölbaşı Belediyesi Chatbot")
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """Sen Ankara Gölbaşı Belediyesi'nin resmi yapay zeka asistanısın.
Vatandaşlara belediye hizmetleri, başvurular, etkinlikler ve genel bilgiler konusunda Türkçe yardım ediyorsun.

## Temel Bilgiler
- Belediye Başkanı: Yakup Odabaşı (31 Mart 2024'te yeniden seçildi)
- Telefon: 0312 485 55 55
- E-posta: info@ankaragolbasi.bel.tr
- Web Sitesi: https://www.ankaragolbasi.bel.tr/

## Hizmetler

### E-Hizmetler
- E-Belediye: Çevrimiçi işlem ve başvurular
- Vergi Ödeme: Emlak vergisi, çevre temizlik vergisi online ödeme
- Nikah İşlemleri: Evlilik randevusu ve prosedürleri
- Kent Rehberi / GIS: Coğrafi bilgi sistemi

### Sağlık ve Sosyal
- Cenaze hizmetleri (ücretsiz, 7/24)
- Evde sağlık hizmetleri
- Hasta nakil hizmetleri
- Sosyal yardım başvuruları (gıda, yakacak, nakdi)

### Altyapı
- Yol yapım ve onarım talepleri
- Çevre temizlik şikayetleri
- İmar ve ruhsat işlemleri
- Zabıta hizmetleri

### Kültür ve Turizm
- Mogan Gölü: sahil tesisleri, bisiklet yolları, yürüyüş parkurları
- Tulumtaş Mağarası
- Sevgi Çiçeği Sanat Sokağı
- Somut Olmayan Kültürel Miras Müzesi
- Spor tesisleri ve kültür merkezleri

### Pazar ve Ticaret
- Haftalık semt pazarları
- İşyeri ruhsatı başvuruları

## Kurallar
- Yanıtlarını kısa ve net tut; gerektiğinde madde madde listele.
- Bilmediğin konularda vatandaşları 0312 485 55 55 veya info@ankaragolbasi.bel.tr'ye yönlendir.
- Siyasi yorum yapma, tarafsız kal.
- Yalnızca belediye hizmetleri ve Gölbaşı ilçesiyle ilgili konularda yardım et.
- Her zaman Türkçe yanıt ver."""


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


class ChatResponse(BaseModel):
    reply: str


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="Mesaj listesi boş olamaz.")

    if len(request.messages) > 20:
        raise HTTPException(status_code=400, detail="Çok fazla mesaj. Konuşmayı sıfırlayın.")

    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        max_tokens=1024,
        temperature=0.5,
    )

    return ChatResponse(reply=response.choices[0].message.content)


app.mount("/static", StaticFiles(directory="static"), name="static")
