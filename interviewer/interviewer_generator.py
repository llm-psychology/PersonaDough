import faiss
from typing import List, Dict
import openai
from ..module.LLM_responder import LLM_responder
import numpy as np

class Interviewer(LLM_responder):
    # ========== 1. Collect user answers ==========
    def collect_user_answers(self, question_list: List[str]) -> List[Dict[str, str]]:
        qa_pairs = []
        for q in question_list:
            a = input(f"{q}\n> ")
            qa_pairs.append({"q": q, "a": a})
        return qa_pairs

    # ========== 2. Format QA Pairs ==========
    def format_qa_pairs(self, qa_pairs: List[Dict[str, str]]) -> List[str]:
        return [f"Q：{pair['q']} A：{pair['a']}" for pair in qa_pairs]

    # ========== 3. Generate Embeddings ==========
    # generate_embedding()

    # ========== 4. Build Vector Index ==========
    def build_vector_index(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index

    # ========== 5. Retrieve Similar Docs ==========
    def retrieve_similar_docs(self, query: str, index: faiss.IndexFlatL2, docs: List[str], top_k: int = 3, similarity_threshold: float = 0.7) -> List[str]:
        res = self._call_embeddings_openai_api([query])

        query_vector = np.array([res.data[0].embedding])
        D, I = index.search(query_vector, top_k)
        return [docs[i] for i, d in zip(I[0], D[0]) if d < (1 - similarity_threshold)]

    # ========== 6. Build Simulation Prompt ==========
    def build_simulation_prompt(self, retrieved_docs: List[str], user_query: str) -> List[Dict[str, str]]:
        memory_context = "\n".join(retrieved_docs) if retrieved_docs else "（無可參考的記憶資料）"
        system_prompt = f"你是一個根據以下記憶資料模擬出來的人格：\n{memory_context}\n請根據這些資料，用一致的口吻與邏輯回答問題。"
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

    # ========== 7. Simulate Persona Answer ==========
    # simulate_persona_answer()

# ========== Main Example Flow ==========
if __name__ == "__main__":
    interviewer = Interviewer()
    # Example usage
    questions = [
        "你平常講話會不會用很多語助詞？",
        "你對人生的看法是什麼？",
        "你最喜歡的電影是哪一部？",
        "朋友跟你求助時你通常會怎麼回應？"
    ]

    print("\n=== 問卷開始 ===")
    qa_pairs = interviewer.collect_user_answers(questions)
    docs = interviewer.format_qa_pairs(qa_pairs)
    interviewer.generate_embedding(docs)
    print("\n=== 建立記憶庫 ===")
    embeddings = interviewer.generate_embedding(docs)
    index = interviewer.build_vector_index(embeddings)

    while True:
        query = input("\n請輸入你的提問（或輸入 'exit' 離開）：\n> ")
        if query.strip().lower() == "exit":
            break
        retrieved = interviewer.retrieve_similar_docs(query, index, docs)
        prompt = interviewer.build_simulation_prompt(retrieved, query)
        answer = interviewer.simulate_persona_answer(prompt)
        print(f"\n模擬人格回答：\n{answer}\n")
