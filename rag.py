
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class SimpleRAG:
    def __init__(self, documents: list[str]):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.documents = documents

        # ----------------------------
        # Encode & normalize documents
        # ----------------------------
        embeddings = self.model.encode(documents, convert_to_numpy=True)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]

        # Inner Product index = cosine similarity (after normalization)
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        self.embeddings = embeddings

        print(f"✅ FAISS index built with {len(documents)} documents")

    def retrieve(
        self,
        query: str,
        k: int = 3,
        relative_threshold: float = 0.8,
    ) -> str:
        # ----------------------------
        # Encode & normalize query
        # ----------------------------
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        # ----------------------------
        # Search FAISS
        # ----------------------------
        scores, indices = self.index.search(query_embedding, k)

        best_score = scores[0][0]

        # ----------------------------
        # 🔒 Confidence gate (skip weak matches)
        # ----------------------------
        if best_score < 0.25:
            print(
                f"\n🔍 Query: {query}\n"
                f"Best score: {best_score:.3f} → ❌ Too low, skipping RAG\n"
            )
            return ""

        # ----------------------------
        # 🔑 Confidence-adaptive dynamic threshold
        # ----------------------------
        if best_score < 0.4:
            dynamic_threshold = best_score * 0.7
        else:
            dynamic_threshold = best_score * relative_threshold

        print(
            f"\n🔍 Query: {query}\n"
            f"Best score: {best_score:.3f} | "
            f"Dynamic threshold: {dynamic_threshold:.3f}\n"
        )

        results = []
        for idx, score in zip(indices[0], scores[0]):
            print(
                f"  Document: '{self.documents[idx][:50]}...' "
                f"| Cosine score: {score:.3f}"
            )

            if score >= dynamic_threshold:
                results.append(self.documents[idx])

        # ----------------------------
        # Return combined context
        # ----------------------------
        return "\n".join(results)
