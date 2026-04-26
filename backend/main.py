import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from pinecone import Pinecone
import google.generativeai as genai

load_dotenv()

app = FastAPI(title="Providencia Neural Network API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pinecone initialization (Base secreta de la empresa)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "providencia-index")
if PINECONE_API_KEY:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
else:
    pinecone_index = None

class ChatRequest(BaseModel):
    message: str
    gemini_api_key: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not pinecone_index:
        raise HTTPException(status_code=500, detail="Falta la API key de Pinecone en el servidor.")
    if not request.gemini_api_key:
        raise HTTPException(status_code=400, detail="El usuario debe proveer su propia API Key de Gemini.")

    user_message = request.message
    api_key = request.gemini_api_key

    try:
        # 1. Vectorizar la pregunta usando la API Key del usuario
        embed_url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={api_key}"
        embed_payload = {
            "model": "models/text-embedding-004",
            "content": {
                "parts": [{"text": user_message}]
            },
            "taskType": "RETRIEVAL_QUERY"
        }
        
        async with httpx.AsyncClient() as client:
            embed_res = await client.post(embed_url, json=embed_payload, timeout=10.0)
            if embed_res.status_code != 200:
                raise Exception(f"Error de Gemini al vectorizar (revisa tu API Key): {embed_res.text}")
            
            query_vector = embed_res.json()["embedding"]["values"]

        # 2. Buscar en Pinecone
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

        # 3. Generar la respuesta usando la API Key del usuario
        prompt = f"""
Eres un asistente corporativo profesional de la red neuronal "Providencia".
Usa la siguiente información de contexto (extraída de los documentos de la empresa) para responder a la pregunta del usuario. 
Si la respuesta no está en el contexto, indícalo de manera educada y profesional. No inventes información.

Contexto de documentos:
{context}

Pregunta del usuario: {user_message}
Respuesta:
"""
        generate_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent?key={api_key}"
        generate_payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }

        async with httpx.AsyncClient() as client:
            gen_res = await client.post(generate_url, json=generate_payload, timeout=30.0)
            if gen_res.status_code != 200:
                raise Exception(f"Error de Gemini al generar respuesta: {gen_res.text}")
            
            response_data = gen_res.json()
            reply_text = response_data["candidates"][0]["content"]["parts"][0]["text"]

        return {"reply": reply_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "Backend de Providencia funcionando correctamente"}
