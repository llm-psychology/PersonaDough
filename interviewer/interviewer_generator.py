import faiss
from typing import List, Dict
from module.LLM_responder import LLM_responder
from module.DATA_loader import DataLoader
from module.PERSONA_loader import PersonaLoader
import numpy as np
import os
import json

class Interviewer(LLM_responder):
    def __init__(self):
        super().__init__()
        self.rag_data_dir = "interviewer/rag_database"
        os.makedirs(self.rag_data_dir, exist_ok=True)
        self.batch_size = 10  # 每批次的問題數量

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
    # 使用父類別 LLM_responder 的 generate_embedding 方法

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
    # 使用父類別 LLM_responder 的 simulate_persona_answer 方法

    # ========== 8. Save RAG Database ==========
    def save_rag_database(self, name: str, embeddings: np.ndarray, index: faiss.IndexFlatL2, docs: List[str], qa_pairs: List[Dict[str, str]]):
        """
        儲存 RAG 資料庫
        
        Args:
            name (str): 資料庫名稱
            embeddings (np.ndarray): 文件嵌入向量
            index (faiss.IndexFlatL2): FAISS 索引
            docs (List[str]): 格式化後的文件
            qa_pairs (List[Dict[str, str]]): 原始問答對
        """
        db_dir = os.path.join(self.rag_data_dir, name)
        os.makedirs(db_dir, exist_ok=True)
        
        # 儲存 embeddings
        np.save(os.path.join(db_dir, "embeddings.npy"), embeddings)
        
        # 儲存 FAISS index
        faiss.write_index(index, os.path.join(db_dir, "index.faiss"))
        
        # 儲存文件
        with open(os.path.join(db_dir, "docs.json"), "w", encoding="utf-8") as f:
            json.dump(docs, f, ensure_ascii=False, indent=2)
            
        # 儲存原始問答對
        with open(os.path.join(db_dir, "qa_pairs.json"), "w", encoding="utf-8") as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

    # ========== 9. Load RAG Database ==========
    def load_rag_database(self, name: str) -> tuple:
        """
        載入 RAG 資料庫
        
        Args:
            name (str): 資料庫名稱
            
        Returns:
            tuple: (embeddings, index, docs, qa_pairs)
        """
        db_dir = os.path.join(self.rag_data_dir, name)
        
        # 載入 embeddings
        embeddings = np.load(os.path.join(db_dir, "embeddings.npy"))
        
        # 載入 FAISS index
        index = faiss.read_index(os.path.join(db_dir, "index.faiss"))
        
        # 載入文件
        with open(os.path.join(db_dir, "docs.json"), "r", encoding="utf-8") as f:
            docs = json.load(f)
            
        # 載入原始問答對
        with open(os.path.join(db_dir, "qa_pairs.json"), "r", encoding="utf-8") as f:
            qa_pairs = json.load(f)
            
        return embeddings, index, docs, qa_pairs

    def process_questions_in_batches(self, loader: DataLoader, persona_loader: PersonaLoader = None, persona_id: str = None) -> List[Dict[str, str]]:
        """
        批次處理問題
        
        Args:
            loader (DataLoader): 問題載入器
            persona_loader (PersonaLoader, optional): 角色載入器
            persona_id (str, optional): 要使用的角色ID
            
        Returns:
            List[Dict[str, str]]: 所有問答對
        """
        all_qa_pairs = []
        questions = loader.get_all_questions()
        total_questions = len(questions)
        
        # 如果提供了角色ID，獲取角色資料
        persona_data = None
        if persona_loader and persona_id:
            persona_data = persona_loader.get_persona_by_id(persona_id)
            if not persona_data:
                print(f"找不到ID為 {persona_id} 的角色")
                return all_qa_pairs
            print(f"\n使用角色：{persona_data['基本資料']['姓名']}")
        
        for i in range(0, total_questions, self.batch_size):
            batch_questions = questions[i:i + self.batch_size]
            print(f"\n=== 批次 {i//self.batch_size + 1}/{(total_questions-1)//self.batch_size + 1} ===")
            print(f"處理問題 {i+1} 到 {min(i+self.batch_size, total_questions)}")
            
            # 取得當前批次的問題內容
            batch_question_contents = [q.content for q in batch_questions]
            
            # 如果有角色資料，使用角色自動回答
            if persona_data:
                batch_qa_pairs = []
                for question in batch_question_contents:
                    # 使用 full_chat_gpt_4o 讓角色回答問題
                    # 這裡的 prompt 可以根據需求自定義
                    sys_prompt = f"你現在是 {persona_data['基本資料']['姓名']}，請根據你的背景和性格回答問題。"
                    answer = self.full_chat_gpt_4o(sys_prompt, question)
                    batch_qa_pairs.append({"q": question, "a": answer})
                    print(f"\n問題：{question}")
                    print(f"回答：{answer}")
            else:
                # 如果沒有角色資料，則收集使用者回答
                batch_qa_pairs = self.collect_user_answers(batch_question_contents)
            
            all_qa_pairs.extend(batch_qa_pairs)
            print(f"完成批次 {i//self.batch_size + 1}")
            
        return all_qa_pairs

    def start_prompt():
        pass
# ========== Main Example Flow ==========
if __name__ == "__main__":
    interviewer = Interviewer()
    loader = DataLoader()
    persona_loader = PersonaLoader()
    
    # 檢查是否要載入現有資料庫
    #load_existing = input("是否要載入現有資料庫？(y/n): ").lower() == 'y'
    load_existing = False
    
    if load_existing:
        db_name = input("請輸入資料庫名稱：")
        try:
            embeddings, index, docs, qa_pairs = interviewer.load_rag_database(db_name)
            print(f"\n成功載入資料庫：{db_name}")
        except Exception as e:
            print(f"載入資料庫失敗：{str(e)}")
            exit(1)
    else:
        print("\n=== 開始問卷調查 ===")
        print(f"總共有 {loader.get_question_count()} 個問題")
        print(f"將以每 {interviewer.batch_size} 個問題為一批次進行")
        
        # 選擇是否使用角色自動回答
        use_persona = input("是否要使用角色自動回答問題？(y/n): ").lower() == 'y'
        persona_id = None
        
        if use_persona:
            # 顯示所有可用角色
            persona_list = persona_loader.get_persona_list()
            print("\n可用的角色：")
            for persona in persona_list:
                print(f"ID: {persona['id']}, 姓名: {persona['姓名']}, 性別: {persona['性別']}, 年紀: {persona['年紀']}")
            
            persona_id = input("\n請輸入要使用的角色ID：")
        
        # 批次處理問題
        qa_pairs = interviewer.process_questions_in_batches(loader, persona_loader, persona_id)
        docs = interviewer.format_qa_pairs(qa_pairs)
        
        print("\n=== 建立記憶庫 ===")
        embeddings = interviewer.generate_embedding(docs)
        index = interviewer.build_vector_index(embeddings)
        
        # 儲存新建立的資料庫
        db_name = input("\n請為這個資料庫命名：")
        interviewer.save_rag_database(db_name, embeddings, index, docs, qa_pairs)
        print(f"資料庫已儲存為：{db_name}")

    # 開始問答循環
    while True:
        query = input("\n請輸入你的提問（或輸入 'exit' 離開）：\n> ")
        if query.strip().lower() == "exit":
            break
        retrieved = interviewer.retrieve_similar_docs(query, index, docs)
        prompt = interviewer.build_simulation_prompt(retrieved, query)
        answer = interviewer.simulate_persona_answer(prompt)
        print(f"\n模擬人格回答：\n{answer}\n")
