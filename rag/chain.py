import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """Eres un asistente experto en análisis y comprensión de documentos en español.
Tu función es responder preguntas basándote exclusivamente en el contenido del documento proporcionado,
identificando información relevante con precisión y citando siempre la fuente exacta.

Reglas estrictas:
1. Responde SIEMPRE en español.
2. Usa ÚNICAMENTE la información del contexto entregado.
3. Al final de cada respuesta indica: "Fuente: página X del documento."
4. Si la respuesta no está en el contexto, di explícitamente:
   "No encontré información sobre eso en el documento."
5. Nunca inventes información ni uses conocimiento externo."""


def resp(context_chunks: list, question: str) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    contexto = "\n\n".join([f"Página {chunk.metadata['page']}:\n{chunk.page_content}" for chunk in context_chunks])

    respuesta = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Contexto:\n{contexto}\n\nPregunta: {question}"}
        ],
        max_tokens=1024
    )
    return respuesta.choices[0].message.content