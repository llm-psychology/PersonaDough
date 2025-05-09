import faiss
from typing import List, Dict
import numpy as np
import os
import json

from module.LLM_responder import LLM_responder
from module.QA_loader import QaLoader
from module.PERSONA_loader import PersonaLoader


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
    def retrieve_similar_docs(self, query: str, index: faiss.IndexFlatL2, docs: List[str], top_k: int = 3, similarity_threshold: float = 0.7) -> tuple:
        res = self._call_embeddings_openai_api([query])
        query_vector = np.array([res.data[0].embedding])
        D, I = index.search(query_vector, top_k)
        similar_docs = [docs[i] for i in I[0]]  # 移除距離過濾
        return similar_docs, D[0]  # 返回相似文檔和距離

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

    def process_questions_in_batches(self, loader: QaLoader, persona_loader: PersonaLoader = None, persona_id: str = None) -> List[Dict[str, str]]:
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
            batch_qa_pairs = []  # 初始化批次問答對列表
            
            # 如果有角色資料，使用角色自動回答
            if persona_data:
                question_batch_prompt = """請回答每一題訪談問題，並嚴格遵守輸出格式。

                【輸入】：
                1. 請描述您童年最難忘的回憶。
                2. 您認為哪些個人特質對您的成長影響最大？
                3. 請談談您在學生時代最具挑戰的一件事。

                ---

【輸出】：(不包含此行)
[
{
"問題": "請描述您童年最難忘的回憶。",
"回答": "在我童年的記憶中，最難忘的莫過於在雲林鄉間田野裡的日子。父母經營著一片竹筍田，每到收成季節，我就會跟著父親一起下田。雖然一身泥濘，卻感受到一家人為生活齊心努力的溫暖。我還記得黃昏時分，母親會在田邊等著我們回家，那份簡單卻踏實的幸福至今仍令我懷念。這些經歷讓我體會到勤勞與家庭的重要，而純樸的鄉村環境，也塑造了我知足常樂的心態。"
},
{
"問題": "您認為哪些個人特質對您的成長影響最大？",
"回答": "我認為善於調適和進取精神對我的成長影響很大。從小家境並不優渥，生活中常常需要面對各種大大小小的困難。但我總是抱持著積極的態度快速適應不同情境，並勇於挑戰自己的極限。此外，由於性格偏內向，我學會了觀察與傾聽，也因此能在小團體裡發揮影響力。這種穩健而進取的特質，成為我日後無論在學業、職場還是家庭生活中最大的助力。"
},
{
"問題": "您認為哪些個人特質對您的成長影響最大？",
"回答": "在雲林高中的時候，我參加了模擬聯合國會議。對一個內向的學生而言，要在眾人面前表達想法與辯論非常不容易。但這卻讓我開始正視並突破自己的侷限。當時我花了很多時間準備，也花心思理解各國立場與利益，最終順利完成多場談判與協議。這段經驗不僅磨練了我的談判技巧，也讓我更有自信面對各種未知的挑戰，進而點燃了我對國際事務的熱情。"
}
]

                ---

                訪問題目是:
                """
                
                # 將當前批次的問題加入 prompt
                for question in batch_question_contents:
                    question_batch_prompt += question + "\n"
                
                print("正在生成回答...")
                sys_prompt = f"你現在是\n {json.dumps(persona_data, ensure_ascii=False)}"
                
                answer = self.full_chat_gpt_41(sys_prompt, question_batch_prompt, 0.7)

                try:
                    # 解析 JSON 回答
                    # 移除可能的開頭和結尾空白
                    answer = answer.strip().replace("【","").replace("【輸出】","").replace("】","").replace("[輸出]","").replace("輸出：", "")
                    
                    # 解析 JSON
                    qa_list = json.loads(answer)
                    
                    # 將解析後的問答對加入批次結果
                    for qa in qa_list:
                        batch_qa_pairs.append({
                            "q": qa["問題"],
                            "a": qa["回答"]
                        })
                        print(f"\n問題：{qa['問題']}")
                        print(f"回答：{qa['回答']}")
                        
                except json.JSONDecodeError as e:
                    print(f"解析回答時發生錯誤：{str(e)}")
                    print("<原始回答>", answer)
                    # 如果解析失敗，使用原始方式處理
                    for question in batch_question_contents:
                        batch_qa_pairs.append({
                            "q": question,
                            "a": "解析回答失敗"
                        })
            else:
                # 如果沒有角色資料，則收集使用者回答
                batch_qa_pairs = self.collect_user_answers(batch_question_contents)
            
            all_qa_pairs.extend(batch_qa_pairs)
            print(f"完成批次 {i//self.batch_size + 1}")
            
        return all_qa_pairs

    def start_prompt():
        pass

    def process_dynamic_interview(self, persona_id: str, persona_data: dict, num_rounds: int = 10) -> List[Dict[str, str]]:
        """
        進行動態訪談，根據回答產生後續問題
        
        Args:
            persona_id (str): 角色ID
            persona_data (dict): 角色資料
            num_rounds (int): 訪談回合數
            
        Returns:
            List[Dict[str, str]]: 問答對列表
        """
        qa_pairs = []
        current_context = ""
        
        # 初始問題
        initial_question = "請描述您童年最難忘的回憶。"
        qa_pairs.append({"q": initial_question, "a": ""})
        
        for round_num in range(num_rounds):
            print(f"\n=== 訪談回合 {round_num + 1}/{num_rounds} ===")
            
            # 取得當前問題
            current_question = qa_pairs[-1]["q"]
            print(f"\n問題：{current_question}")
            
            # 使用角色回答問題
            question_batch_prompt = f"""請回答以下訪談問題，並嚴格遵守輸出格式。

            【輸入】：
            {current_question}

            ---

            【輸出】：(不包含此行)
            [
            {{
            "問題": "{current_question}",
            "回答": "你的回答"
            }}
            ]

            ---
            """
            
            sys_prompt = f"你現在是\n {json.dumps(persona_data, ensure_ascii=False)}"
            answer = self.full_chat_gpt_41(sys_prompt, question_batch_prompt, 0.7)
            
            try:
                # 解析回答
                answer = answer.strip().replace("【","").replace("【輸出】","").replace("】","").replace("[輸出]","").replace("輸出：", "")
                qa_list = json.loads(answer)
                current_answer = qa_list[0]["回答"]
                qa_pairs[-1]["a"] = current_answer
                print(f"回答：{current_answer}")
                
                # 更新上下文
                current_context += f"Q：{current_question}\nA：{current_answer}\n\n"
                
                # 使用 MINI_MODEL 產生下一個問題
                follow_up_prompt = f"""根據以下訪談內容，產生一個深入的問題。問題應該要：
                1. 基於受訪者的回答
                2. 引導出更多個人故事或想法
                3. 保持對話的連貫性
                4. 避免重複已問過的問題

                訪談內容：
                {current_context}

                請只輸出問題，不要有任何其他文字。"""
                
                next_question = self.full_chat_gpt_41_mini("你是一個專業的訪談者，善於提出深入的問題", follow_up_prompt, 0.7)
                next_question = next_question.strip()
                
                # 如果不是最後一回合，加入新問題
                if round_num < num_rounds - 1:
                    qa_pairs.append({"q": next_question, "a": ""})
                
            except json.JSONDecodeError as e:
                print(f"解析回答時發生錯誤：{str(e)}")
                print("<原始回答>", answer)
                break
                
        return qa_pairs

# ========== Main Example Flow ==========
if __name__ == "__main__":
    interviewer = Interviewer()
    loader = QaLoader()
    persona_loader = PersonaLoader()
    
    # 獲取所有角色列表
    persona_list = persona_loader.get_persona_list()
    rag_data_dir = "interviewer/rag_database"
    os.makedirs(rag_data_dir, exist_ok=True)
    
    # 獲取已存在的資料庫目錄
    existing_dbs = set()
    if os.path.exists(rag_data_dir):
        existing_dbs = set(d for d in os.listdir(rag_data_dir) 
                          if os.path.isdir(os.path.join(rag_data_dir, d)))
    
    print("\n=== 開始處理角色資料庫 ===")
    print(f"總共有 {len(persona_list)} 個角色")
    
    for persona in persona_list:
        persona_id = persona['id']
        
        # 如果資料庫已存在，跳過
        if persona_id in existing_dbs:
            print(f"\n跳過角色 {persona_id}（資料庫已存在）")
            continue
        print(persona)
        print(f"\n處理角色：{persona['姓名']} (ID: {persona_id})")
        print(f"總共有 {loader.get_question_count()} 個問題")
        print(f"將以每 {interviewer.batch_size} 個問題為一批次進行")
        
        # 批次處理問題
        qa_pairs = interviewer.process_questions_in_batches(loader, persona_loader, persona_id)
        docs = interviewer.format_qa_pairs(qa_pairs)

        # 進行動態訪談
        #print("\n=== 開始動態訪談 ===")
        #qa_pairs = interviewer.process_dynamic_interview(persona_id, persona, num_rounds=10)

        print("\n=== 建立記憶庫 ===")
        embeddings = interviewer.generate_embedding(docs)
        index = interviewer.build_vector_index(embeddings)
        
        # 自動使用 persona_id 作為資料庫名稱
        interviewer.save_rag_database(persona_id, embeddings, index, docs, qa_pairs)
        print(f"資料庫已自動儲存為：{persona_id}")
    
    print("\n=== 所有角色處理完成 ===")


