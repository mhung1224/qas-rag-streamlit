from pypdf import PdfReader
from docx import Document


# Load documents from streamlit file uploader(file-like object)
def _load_txt(file) -> str:
    return file.read().decode("utf-8", errors="ignore")

def _load_pdf(file) -> str:
    reader = PdfReader(file)
    text = []
    for page in reader.pages:
        t = page.extract_text() or ""
        text.append(t)
    return "\n".join(text)

def _load_doc(file) -> str:
    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def load_file(file):
    name = file.name
    ext = name.split(".")[-1].lower()

    if hasattr(file, "seek"):
        try:
            file.seek(0)
        except Exception:
            pass

    if ext in ("txt", "md"):
        content = _load_txt(file)
    elif ext == "pdf":
        content = _load_pdf(file)
    elif ext == "docx":
        content = _load_doc(file)
    else:
        return None

    return name, content