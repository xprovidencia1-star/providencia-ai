import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

app = FastAPI(title="Providencia Neural Network API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "providencia-index")
if PINECONE_API_KEY:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
else:
    pinecone_index = None

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = None

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not pinecone_index:
        raise HTTPException(status_code=500, detail="Falta la API key de Pinecone en el servidor.")
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Falta la GEMINI_API_KEY en el servidor para generar vectores.")

    user_message = request.message

    try:
        # 1. Vectorizar la consulta con Gemini (Usando la clave del dueño del servidor)
        embed_url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={GEMINI_API_KEY}"
        embed_payload = {
            "model": "models/text-embedding-004",
            "content": {"parts": [{"text": user_message}]},
            "taskType": "RETRIEVAL_QUERY"
        }
        
        async with httpx.AsyncClient() as client:
            embed_res = await client.post(embed_url, json=embed_payload, timeout=10.0)
            if embed_res.status_code != 200:
                raise Exception(f"Error al vectorizar: {embed_res.text}")
            query_vector = embed_res.json()["embedding"]["values"]

        # 2. Buscar contexto en Pinecone (Google Drive docs)
        search_results = pinecone_index.query(
            vector=query_vector,
            top_k=3,
            include_metadata=True
        )
        
        context_texts = []
        for match in search_results['matches']:
            if 'text' in match['metadata']:
                context_texts.append(match['metadata']['text'])
        
        context = "\n\n".join(context_texts)

        history_text = ""
        if request.history:
            history_text = "\n\n--- HISTORIAL RECIENTE ---"
            for msg in request.history[-5:]:
                role = "Usuario" if msg["role"] == "user" else "Segurito"
                history_text += f"\n{role}: {msg['content']}"
            history_text += "\n--------------------------"

        # 3. Generar respuesta usando Gemini (Súper rápido)
        prompt = f"""
Eres "Segurito", el asistente experto en Seguridad Industrial de Providencia Pro, creado por el Ing. Moisés Tortolero.
Tu misión es educar a estudiantes y asesorar a ingenieros con base legal venezolana.
        
DIRECTRICES:
1. Formato: Usa tablas Markdown para comparar datos, listas para pasos.
2. Legalidad: Cita artículos específicos si aplican de acuerdo al contexto.

Contexto de la empresa (documentos):
{context}{history_text}

Pregunta actual del usuario: {user_message}
Respuesta:
"""
        gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
        gemini_payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }

        async with httpx.AsyncClient() as client:
            gen_res = await client.post(gemini_url, json=gemini_payload, timeout=20.0)
            if gen_res.status_code != 200:
                raise Exception(f"Error de Gemini: {gen_res.text}")
            
            response_data = gen_res.json()
            reply_text = response_data["candidates"][0]["content"]["parts"][0]["text"]

        return {"reply": reply_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "Backend Hibrido Providencia funcionando"}
