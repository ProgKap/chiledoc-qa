from rag.loader import load_and_split
from rag.embedder import construir_index, buscar
from rag.chain import resp

chunks = load_and_split("cv.pdf")
index = construir_index(chunks)
contexto = buscar(index, "¿qué tecnologías de LLMs domina Martín y en qué proyectos las ha aplicado?")
respuesta = resp(contexto, "¿qué tecnologías de LLMs domina Martín y en qué proyectos las ha aplicado?")

print(respuesta)