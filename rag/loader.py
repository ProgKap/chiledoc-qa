from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split(patronpdf: str) -> list:
    """Carga un PDF y lo divide en chunks con solapamiento"""
    
    loader = PyPDFLoader(patronpdf)
    paginas = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " "])
    
    chunks = splitter.split_documents(paginas)
    return chunks


