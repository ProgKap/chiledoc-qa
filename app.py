import gradio as gr
from rag.loader import load_and_split
from rag.embedder import construir_index, buscar
from rag.chain import resp

def proceso_pdf(pdf_file, state):
    if pdf_file is None:
        return "Por favor, sube un archivo PDF.", state
    chunks = load_and_split(pdf_file.name)
    index = construir_index(chunks)
    return f"PDF cargado y procesado: {len(chunks)} fragmentos indexados.", index

def responderPregunta(pregunta, historia, state):
    if state is None:
        return historia + [{"role": "user", "content": pregunta}, {"role": "assistant", "content": "Por favor, sube un PDF primero."}], state
    contexto = buscar(state, pregunta)
    respuesta = resp(contexto, pregunta)
    return historia + [{"role": "user", "content": pregunta}, {"role": "assistant", "content": respuesta}], state

with gr.Blocks(title="ChileDoc QA") as demo:
    gr.Markdown("# 📄 ChileDoc QA\nPregunta cualquier cosa sobre tu documento.")
    index_state = gr.State(None)

    with gr.Row():
        pdf_input = gr.File(label="Sube tu PDF", file_types=[".pdf"])
        status = gr.Textbox(label="Estado", interactive=False)

    pdf_input.change(proceso_pdf, [pdf_input, index_state], [status, index_state])

    chatbot = gr.Chatbot(height=400)
    question = gr.Textbox(placeholder="¿Qué quieres saber del documento?")
    question.submit(responderPregunta, [question, chatbot, index_state], [chatbot, index_state])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())