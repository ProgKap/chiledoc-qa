from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def construir_index(chunks: list) -> FAISS:
    """Genera embeddings y construye el índice FAISS"""
    
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    #descargar el modelo por primera vez 
    index = FAISS.from_documents(chunks, embedding)
    return index

def buscar(index: FAISS, query: str, k: int = 4) -> list:
    """Realiza una búsqueda en el índice FAISS y devuelve los chunks más relevantes"""
    
    return index.similarity_search(query, k=k)

