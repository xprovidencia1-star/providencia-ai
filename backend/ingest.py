import os
import gdown
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import uuid

load_dotenv()

# Configuraciones
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "providencia-index")
# Enlace que me proporcionaste (la carpeta debe tener permisos de "Cualquiera con el enlace puede ver")
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1TkpK2iRzq7n47FmZVBEApg6uA7GqLplX?usp=sharing"
DOWNLOAD_DIR = "./downloaded_docs"

def main():
    if not GEMINI_API_KEY or not PINECONE_API_KEY:
        print("Error: Asegúrate de poner tus API Keys en el archivo .env primero.")
        return

    # 1. Configurar APIs
    genai.configure(api_key=GEMINI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Crear índice en Pinecone si no existe (Dimensión 768 para embeddings de Gemini)
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creando índice '{PINECONE_INDEX_NAME}' en Pinecone...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=768, 
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1' # Región por defecto para cuentas gratuitas
            )
        )
    
    index = pc.Index(PINECONE_INDEX_NAME)

    # 2. Descargar archivos de Google Drive
    print("Descargando archivos desde Google Drive...")
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    try:
        # gdown baja el contenido de la carpeta
        gdown.download_folder(DRIVE_FOLDER_URL, output=DOWNLOAD_DIR, use_cookies=False)
    except Exception as e:
        print(f"Fallo al descargar de Google Drive: {e}")
        return

    # 3. Procesar y subir a Pinecone
    print("Procesando documentos y convirtiéndolos en conocimiento...")
    vectors = []
    
    for filename in os.listdir(DOWNLOAD_DIR):
        file_path = os.path.join(DOWNLOAD_DIR, filename)
        if os.path.isfile(file_path) and filename.endswith('.txt'):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Partir el texto en trozos (chunks)
                chunks = content.split('\n\n')
                chunks = [c.strip() for c in chunks if len(c.strip()) > 50]

                for i, chunk in enumerate(chunks):
                    # Convertir texto a vectores usando Gemini Embeddings
                    result = genai.embed_content(
                        model="models/text-embedding-004",
                        content=chunk,
                        task_type="retrieval_document"
                    )
                    embedding = result['embedding']
                    
                    vector_id = str(uuid.uuid4())
                    vectors.append({
                        "id": vector_id,
                        "values": embedding,
                        "metadata": {
                            "source": filename,
                            "text": chunk
                        }
                    })
                    print(f"Procesado fragmento {i+1} de {filename}")

            except Exception as e:
                print(f"Error con archivo {filename}: {e}")

    # 4. Subir a la base de datos
    if vectors:
        print(f"Subiendo {len(vectors)} vectores a Pinecone. Esto puede tardar un poco...")
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            index.upsert(vectors=batch)
        print("¡Ingesta de datos completada con éxito!")
    else:
        print("No se encontraron textos válidos para procesar (asegúrate de que haya archivos .txt).")

if __name__ == "__main__":
    main()
