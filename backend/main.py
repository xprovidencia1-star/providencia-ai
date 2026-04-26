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
pinecone_index = None

if PINECONE_API_KEY:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        indexes = pc.list_indexes().names()
        if indexes:
            # Seleccionar automáticamente el primer índice de la cuenta
            pinecone_index = pc.Index(indexes[0])
        else:
            print("Advertencia: No hay ningún índice creado en tu cuenta de Pinecone.")
    except Exception as e:
        print(f"Error al conectar con Pinecone: {e}")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = None

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    global pinecone_index
    if not pinecone_index:
        # Intentar reconectar por si el índice se acaba de crear
        try:
            indexes = pc.list_indexes().names()
            if indexes:
                pinecone_index = pc.Index(indexes[0])
        except:
            pass
            
    if not pinecone_index:
        raise HTTPException(status_code=500, detail="Base de datos vacía. Visita /ingest primero.")
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

@app.get("/ingest")
def ingest_endpoint():
    import threading
    import uuid
    import gdown
    import shutil
    from pinecone import ServerlessSpec
    
    if not GEMINI_API_KEY or not PINECONE_API_KEY:
        return {"error": "Faltan las API keys en Render"}

    def background_task():
        try:
            print("[BACKGROUND] Iniciando tarea de ingesta de documentos...")
            # 1. Crear índice si no existe
            indexes = pc.list_indexes().names()
            index_name = "providencia-index"
            if index_name not in indexes:
                print(f"[BACKGROUND] Creando índice {index_name} en Pinecone...")
                pc.create_index(
                    name=index_name,
                    dimension=768,
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
            
            index = pc.Index(index_name)
            
            # 2. Descargar Drive
            print("[BACKGROUND] Descargando archivos de Google Drive...")
            DOWNLOAD_DIR = "/tmp/docs"
            if os.path.exists(DOWNLOAD_DIR):
                shutil.rmtree(DOWNLOAD_DIR)
            os.makedirs(DOWNLOAD_DIR, exist_ok=True)
            
            DRIVE_URL = "https://drive.google.com/drive/folders/1TkpK2iRzq7n47FmZVBEApg6uA7GqLplX?usp=sharing"
            gdown.download_folder(DRIVE_URL, output=DOWNLOAD_DIR, use_cookies=False)
            
            # 3. Procesar
            print("[BACKGROUND] Vectorizando documentos con Gemini...")
            vectors = []
            for filename in os.listdir(DOWNLOAD_DIR):
                if filename.endswith('.txt'):
                    with open(os.path.join(DOWNLOAD_DIR, filename), 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    chunks = [c.strip() for c in content.split('\n\n') if len(c.strip()) > 50]
                    
                    for chunk in chunks:
                        result = genai.embed_content(
                            model="models/text-embedding-004",
                            content=chunk,
                            task_type="retrieval_document"
                        )
                        vectors.append({
                            "id": str(uuid.uuid4()),
                            "values": result['embedding'],
                            "metadata": {"source": filename, "text": chunk}
                        })
                        
            # 4. Subir a Pinecone
            if vectors:
                print(f"[BACKGROUND] Subiendo {len(vectors)} vectores a Pinecone...")
                for i in range(0, len(vectors), 100):
                    index.upsert(vectors=vectors[i:i+100])
                    
            global pinecone_index
            pinecone_index = index
            print("[BACKGROUND] ¡Ingesta completada exitosamente!")
                    
        except Exception as e:
            print(f"[BACKGROUND] Error durante la ingesta: {str(e)}")

    # Iniciar la tarea en un hilo separado
    thread = threading.Thread(target=background_task)
    thread.start()
            
    return {"status": "¡Proceso de lectura iniciado en segundo plano! Espera unos 3 o 4 minutos, la IA te avisará en la consola interna de Render y luego podrás chatear."}

@app.get("/")
def read_root():
    return {"status": "Backend Hibrido Providencia funcionando"}
