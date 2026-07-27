import fitz
import re

def load_document(file_path):
    if file_path.lower().endswith('.pdf'):
        text = ""
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

def chunk_text(text, chunk_size=800, overlap=100):
    paragraphs = re.split(r'\n\s*\n', text)
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) < chunk_size:
            current += para + "\n"
        else:
            if current:
                chunks.append(current.strip())
            current = current[-overlap:] + para + "\n"
    if current:
        chunks.append(current.strip())
    return chunks