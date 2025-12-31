import json
from pathlib import Path
from resume_rag import ResumeRAG
from resume_loader import load_resume_text


class ResumeCache:
    """
    Disk-based resume cache.
    Handles persistence and exposes interview helpers.
    """

    def __init__(self):
        self.cache_dir = Path("resume_cache")
        self.cache_dir.mkdir(exist_ok=True)

        self.meta_file = self.cache_dir / "metadata.json"
        self.resume_rag = ResumeRAG()

        # ----------------------------
        # Load resume from disk if it exists (TXT / PDF / DOCX)
        # ----------------------------
        self.resume_file = None
        for ext in ("*.txt", "*.pdf", "*.docx"):
            files = list(self.cache_dir.glob(ext))
            if files:
                self.resume_file = files[0]
                break

        if self.resume_file and self.resume_file.exists():
            text = load_resume_text(self.resume_file).strip()
            if text:
                self.resume_rag.load_resume(text)

    # ----------------------------
    # Save resume (called by Flask)
    # ----------------------------
    def save_resume(self, resume_text: str) -> dict:
        try:
            # Always normalize saved resume to TXT
            self.resume_file = self.cache_dir / "resume.txt"
            self.resume_file.write_text(resume_text, encoding="utf-8")

            # Load into ResumeRAG (question generation happens here)
            self.resume_rag.load_resume(resume_text)

            metadata = {
                "has_resume": True,
                "num_questions": len(self.resume_rag.questions),
            }

            self.meta_file.write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8"
            )

            return {
                "success": True,
                "questions_count": metadata["num_questions"],
                "questions": self.resume_rag.questions,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    # ----------------------------
    # Resume status helpers
    # ----------------------------
    def has_resume(self) -> bool:
        return self.resume_file is not None and self.resume_file.exists()

    def get_metadata(self) -> dict:
        if not self.meta_file.exists():
            return {"has_resume": False}
        return json.loads(self.meta_file.read_text(encoding="utf-8"))

    # ----------------------------
    # Interview helpers (agent)
    # ----------------------------
    def get_question(self, idx: int) -> str:
        if idx < len(self.resume_rag.questions):
            return self.resume_rag.questions[idx]
        return ""

    def evaluate_answer(self, idx: int, answer: str):
        return self.resume_rag.evaluate_answer(idx, answer)

    # ----------------------------
    # Clear cache
    # ----------------------------
    def clear_cache(self):
        if self.resume_file and self.resume_file.exists():
            self.resume_file.unlink()

        if self.meta_file.exists():
            self.meta_file.unlink()

        self.resume_rag = ResumeRAG()