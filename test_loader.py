from rag.loader import load_and_split

chunks = load_and_split("cv.pdf")
print(f"Total chunks: {len(chunks)}")
print(f"Primer chunk: {chunks[0].page_content[:300]}")
print(f"Metadatos del primer chunk: {chunks[0].metadata}")

