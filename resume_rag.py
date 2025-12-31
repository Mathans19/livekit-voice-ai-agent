from typing import List
import os
from groq import Groq


class ResumeRAG:
    def __init__(self):
        self.resume_text = ""
        self.questions: List[str] = []
        self.answers = []
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def load_resume(self, text: str):
        self.resume_text = text

        # 🔒 Generate many, but USE ONLY 2
        self.questions = self._generate_questions_llm(text)[:2]

        self.answers = []
        print(f"✅ {len(self.questions)} interview questions generated")

    def _generate_questions_llm(self, resume: str) -> List[str]:
        try:
            prompt = f"""Generate 5 technical interview questions from this resume.
One question per line, numbered 1-5.

Resume:
{resume[:1000]}"""

            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300
            )

            content = response.choices[0].message.content.strip()
            questions = []

            for line in content.split('\n'):
                line = line.strip()
                if line and len(line) > 10:
                    for prefix in ['1.', '2.', '3.', '4.', '5.']:
                        if line.startswith(prefix):
                            line = line[len(prefix):].strip()
                    if '?' in line:
                        questions.append(line)

            if len(questions) < 5:
                questions.extend([
                    "Tell me about your most challenging project",
                    "What are your key strengths"
                ][:5 - len(questions)])

            return questions[:5]

        except Exception as e:
            print(f"❌ Question generation failed: {e}")
            return [
                "Tell me about your work experience",
                "What are your technical skills",
                "Describe a challenging project",
                "What is your education background",
                "Why should we hire you"
            ]

    def get_question(self, idx: int) -> str:
        if idx < len(self.questions):
            return self.questions[idx]
        return ""

    def evaluate_answer(self, idx: int, answer: str):
        if not answer.strip():
            return 0.0, "No answer provided."

        if idx >= len(self.questions):
            return 0.0, "Invalid interview state."

        q = self.questions[idx]

        try:
            prompt = f"""Rate this interview answer (0-10).

Resume: {self.resume_text[:800]}
Question: {q}
Answer: {answer}

Format:
SCORE: [0-10]
FEEDBACK: [brief feedback]"""

            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150
            )

            content = response.choices[0].message.content.strip()
            score, feedback = 5.0, "Good answer."

            for line in content.split('\n'):
                if 'SCORE:' in line:
                    try:
                        score = float(line.split(':')[1].strip())
                        score = max(0, min(10, score))
                    except:
                        pass
                elif 'FEEDBACK:' in line:
                    feedback = line.split(':', 1)[1].strip()

            self.answers.append({
                "question": q,
                "answer": answer,
                "score": score / 10
            })

            return score / 10, feedback

        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return 0.5, "Answer recorded."

    def get_summary(self) -> str:
        if not self.answers:
            return "No answers recorded."

        avg = sum(a["score"] for a in self.answers) / len(self.answers) * 10
        rating = "Excellent" if avg >= 8 else "Good" if avg >= 6 else "Needs Improvement"
        return f"Score: {avg:.1f}/10. Rating: {rating}"
