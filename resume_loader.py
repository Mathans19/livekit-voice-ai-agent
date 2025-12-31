from pathlib import Path
from pypdf import PdfReader
from docx import Document


def load_resume_text(file_path: Path) -> str:
    ext = file_path.suffix.lower()

    if ext == ".txt":
        return file_path.read_text(encoding="utf-8", errors="ignore")

    if ext == ".pdf":
        reader = PdfReader(str(file_path))
        text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
        return "\n".join(text)

    if ext == ".docx":
        doc = Document(str(file_path))
        return "\n".join(p.text for p in doc.paragraphs)

    raise ValueError("Unsupported resume format")
