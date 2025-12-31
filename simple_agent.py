from websearch import web_search
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    AgentServer,
    AgentSession,
    Agent,
    ChatContext,
    ChatMessage,
)
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins import groq
from groq import AsyncGroq

from rag import SimpleRAG
from resume_cache import ResumeCache

load_dotenv()

# ----------------------------
# RAG setup
# ----------------------------
def load_documents(path="know.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

DOCUMENTS = load_documents()
rag = SimpleRAG(DOCUMENTS)


class RAGAssistant(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a helpful AI assistant. "
                "When system messages provide you with information, USE THAT INFORMATION to answer. "
                "Answer naturally and concisely in 5-10 words unless more detail is needed. "
                "In interview mode, repeat EXACTLY what system messages tell you to say, word-for-word."
            )
        )
        self.interview_mode = False
        self.waiting_for_answer = False
        self.q_idx = 0
        self.scores = []
        self.groq_client = AsyncGroq()
        self.resume_cache = ResumeCache()

    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ):
        user_text = new_message.text_content.strip()

        print(f"\n🗣 USER: {user_text}")
        print(
            f"🔎 STATE → interview_mode={self.interview_mode}, "
            f"waiting_for_answer={self.waiting_for_answer}, q_idx={self.q_idx}, "
            f"scores={self.scores}"
        )

        if self.interview_mode and "start interview" in user_text.lower():
            print("⚠️ Start interview ignored (already in interview)")
            turn_ctx.add_message(
                role="system",
                content="REPEAT EXACTLY: Please answer the interview question."
            )
            return

        if self.interview_mode and self.waiting_for_answer:
            print(f"📝 Evaluating answer for question {self.q_idx + 1}")

            score, feedback = self.resume_cache.evaluate_answer(
                self.q_idx, user_text
            )
            self.scores.append(score)

            words = feedback.split()
            if len(words) > 10:
                short_feedback = " ".join(words[:10]) + "."
            else:
                short_feedback = feedback
            
            print(f"💬 Short feedback: {short_feedback}")
            print(f"🔢 Before check: q_idx={self.q_idx}, total_scores={len(self.scores)}")

            if len(self.scores) >= 2:
                print("✅ CONDITION MET: 2 answers received, ENDING INTERVIEW")
                
                avg = sum(self.scores) / len(self.scores)
                final = avg * 10
                rating = (
                    "Excellent" if final >= 8
                    else "Good" if final >= 6
                    else "Needs Improvement"
                )

                print(f"🏁 Interview finished | Final score={final:.1f}/10 | Rating={rating}")

                self.interview_mode = False
                self.waiting_for_answer = False
                self.q_idx = 0
                self.scores = []

                final_message = f"{short_feedback} Interview complete. Score {final:.1f} out of 10. {rating}."
                
                print(f"📢 Final message to speak: {final_message}")
                
                turn_ctx.add_message(
                    role="system",
                    content=f"REPEAT WORD-FOR-WORD WITHOUT ANY CHANGES: {final_message}"
                )
                
                print("🛑 Interview ended - NO MORE QUESTIONS")
                return

            self.q_idx += 1
            print(f"➡️ CONTINUING TO QUESTION {self.q_idx + 1}")
            
            next_q = self.resume_cache.get_question(self.q_idx)
            self.waiting_for_answer = True

            print(f"🎤 Asking question {self.q_idx + 1}: {next_q}")

            combined = f"{short_feedback} Next question: {next_q}"
            
            print(f"📢 Combined message: {combined}")

            turn_ctx.add_message(
                role="system",
                content=f"REPEAT WORD-FOR-WORD WITHOUT ANY CHANGES: {combined}"
            )
            return

        if "start interview" in user_text.lower():
            print("▶ Start interview detected")

            self.resume_cache = ResumeCache()

            if not self.resume_cache.has_resume():
                print("❌ Resume not found")
                turn_ctx.add_message(
                    role="system",
                    content="REPEAT EXACTLY: Please upload your resume before starting the interview."
                )
                return

            print("✅ Resume found → Interview mode ON")

            self.interview_mode = True
            self.waiting_for_answer = True
            self.q_idx = 0
            self.scores = []

            first_q = self.resume_cache.get_question(0)
            print(f"🎤 Asking question 1: {first_q}")
            
            welcome_msg = f"Welcome to the interview. First question: {first_q}"
            print(f"📢 Welcome message: {welcome_msg}")

            turn_ctx.add_message(
                role="system",
                content=f"REPEAT WORD-FOR-WORD WITHOUT ANY CHANGES: {welcome_msg}"
            )
            return

        if self.interview_mode:
            print("🚫 Normal chat blocked (interview active)")
            turn_ctx.add_message(
                role="system",
                content="REPEAT EXACTLY: Please answer the interview question."
            )
            return

        # ======================================================
        # 💬 NORMAL CHAT - SMART ROUTING
        # ======================================================
        
        print("🔍 Step 1: Classifying query...")
        
        classify_prompt = (
            "You are a query classifier. Analyze the user's question and determine what type it is.\n\n"
            "Categories:\n"
            "CURRENT - Questions that need real-time, up-to-date information that changes frequently\n"
            "FACTUAL - Questions about well-established, stable knowledge that doesn't change\n"
            "UNCERTAIN - Questions about specialized, niche, or ambiguous topics\n\n"
            f"User question: {user_text}\n\n"
            "Respond with only one word: CURRENT, FACTUAL, or UNCERTAIN"
        )

        response = await self.groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": classify_prompt}],
            temperature=0
        )

        category = response.choices[0].message.content.strip().upper()
        print(f"✅ Classified as: {category}")

        if category == "CURRENT":
            print("🔍 Step 2: CURRENT query → Skip RAG, use Web Search")
            web_context = web_search(user_text)
            print(f"📡 Web results: {web_context[:200] if web_context else 'None'}...")
            
            if web_context:
                turn_ctx.add_message(
                    role="system",
                    content=f"Answer in 6-8 words using this info:\n\n{web_context}"
                )
            else:
                turn_ctx.add_message(
                    role="system",
                    content="Say: No current information available."
                )
            return

        print("🔍 Step 2: Not CURRENT → Checking RAG...")
        rag_context = rag.retrieve(user_text, k=3)
        
        if rag_context:
            print(f"📚 RAG found context:\n{rag_context[:200]}...")
            turn_ctx.add_message(
                role="system",
                content=f"Use this info to answer in 6-8 words:\n{rag_context}"
            )
            return

        print("🔍 Step 3: RAG empty → Deciding next step...")

        if category == "FACTUAL":
            print("🧠 Using LLM knowledge")
            return
        
        else:
            print("🌐 Using Web Search")
            web_context = web_search(user_text)
            print(f"📡 Web results: {web_context[:200] if web_context else 'None'}...")
            
            if web_context:
                turn_ctx.add_message(
                    role="system",
                    content=f"Answer in 6-8 words using this info:\n\n{web_context}"
                )
            else:
                turn_ctx.add_message(
                    role="system",
                    content="Say: No information available."
                )
            return


server = AgentServer()

@server.rtc_session()
async def my_agent(ctx: agents.JobContext):
    session = AgentSession(
        stt=groq.STT(model="whisper-large-v3"),
        llm=groq.LLM(model="llama-3.1-8b-instant"),
        tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )
    await session.start(room=ctx.room, agent=RAGAssistant())

if __name__ == "__main__":
    agents.cli.run_app(server)

    