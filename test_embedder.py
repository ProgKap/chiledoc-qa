from rag.loader import load_and_split
from rag.embedder import construir_index, buscar

chunks = load_and_split("cv.pdf")
index = construir_index(chunks)


resultados = buscar(index, "¿Qué tecnologías de LLMs e inteligencia artificial domina Martín y en qué proyectos las ha aplicado?")
for r in resultados:
    print(f"Página: {r.metadata['page']}, Contenido: {r.page_content[:150]}...")
    print('---')
    
    