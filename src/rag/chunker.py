import re
import uuid


def simple_chunk(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = max(end - overlap, end)
    return [c for c in chunks if c.strip()]

def chunk_document(doc: tuple[str, str], chunk_size: int, overlap: int):
    # docs: a tuple of (doc_id, text)
    chunks = []
    for ch in simple_chunk(doc[1], chunk_size, overlap):
        chunks.append({
            "id":  str(uuid.uuid4()),
            "text": ch,
            "metadata": {"source": doc[0]},
        })
    return chunks