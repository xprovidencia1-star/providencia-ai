import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
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

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "providencia-index")
if PINECONE_API_KEY:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
else:
    pinecone_index = None

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

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
        # 1. Vectorizar la consulta con el SDK oficial de Gemini
        embedding_result = genai.embed_content(
            model="models/text-embedding-004",
            content=user_message,
            task_type="retrieval_query"
        )
        query_vector = embedding_result['embedding']

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

        # 3. Generar respuesta usando Gemini (Súper rápido y Nativo)
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
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        reply_text = response.text

        return {"reply": reply_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "Backend Hibrido Providencia funcionando"}
