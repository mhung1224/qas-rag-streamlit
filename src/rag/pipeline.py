import asyncio
from .config import AppConfig
from .embedder import Embedder
from .chunker import chunk_document
from .files_loader import load_file
from .llm import LLM
from .vectorstore import PineconeStore


SYSTEM_PROMPT = """
Bạn là trợ lý RAG. Chỉ sử dụng NGUYÊN VẸN thông tin trong phần NGỮ CẢNH cho sẵn để trả lời.
QUY TẮC:
- Nếu NGỮ CẢNH không chứa thông tin cần thiết để trả lời câu hỏi, hãy trả lời rằng bạn không biết.
- Không suy đoán, không dùng kiến thức ngoài NGỮ CẢNH.
- Không trích dẫn nguồn hay tên tệp. Không liệt kê đường dẫn, ID, hay metadata.
- Trả lời ngắn gọn, chính xác; nếu phù hợp, hãy trình bày theo gạch đầu dòng.
- Nếu câu hỏi mơ hồ, hãy nêu rõ điều còn thiếu dựa trên NGỮ CẢNH (không hỏi ngược).
"""

class RAGPipeline:
    def __init__(self):
        self.cfg = AppConfig()
        self.embedder = Embedder()
        self.vec_store = PineconeStore()

    def rebuild_index(self, file):
        doc = load_file(file)
        chunk_size = self.cfg.chunk_cfg.chunk_size
        chunk_overlap = self.cfg.chunk_cfg.chunk_overlap
        chunks = chunk_document(doc, chunk_size, chunk_overlap)
        if chunks:
            self.vec_store.add_texts(self.embedder, chunks)

    def delete_by_source(self, source: str) -> None:
        self.vec_store.delete_by_source(source)

    def answer(self, query: str, top_k: int = 5):
        hits = self.vec_store.similarity_search(self.embedder, query, top_k=top_k)
        context_blocks = []
        for h in hits:
            src = h.get("metadata", {}).get("source", "document")
            context_blocks.append(f"[{src}]\n{h['text']}")
        context = "\n\n---\n\n".join(context_blocks) or "(no context)"

        prompt = f"""{SYSTEM_PROMPT}

        Context: {context}
        Question: {query}
        Answer: """

        llm = LLM()

        try:
            answer = asyncio.run(llm.generate(prompt))
        except Exception as e:
            answer = f"[LLM error: {e}] Showing top passages instead:\n\n" + context

        return answer, hits