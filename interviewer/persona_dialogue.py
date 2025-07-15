import logging
import json
import os
import uuid
from typing import List, Dict, Optional, Tuple
import aiofiles

from .chat_with_persona import ChatWith
from .interviewer_generator import generate_persona_rag_by_id

logger = logging.getLogger(__name__)

class PersonaDialogueManager:
    def __init__(self):
        self.conversation_history_path = "conversation_history"
        os.makedirs(self.conversation_history_path, exist_ok=True)
        self.chatbot = None

    async def initialize(self):
        """初始化聊天處理器"""
        if not self.chatbot:
            self.chatbot = ChatWith()
            await self.chatbot.wait_until_ready()

    async def load_dialogue_history(self, session_id: str) -> List[Dict[str, str]]:
        """載入對話歷史"""
        history_file = os.path.join(self.conversation_history_path, f"dialogue_{session_id}.json")
        if os.path.exists(history_file):
            async with aiofiles.open(history_file, "r", encoding="utf-8") as f:
                content = await f.read()
                return json.loads(content)
        return []

    async def save_dialogue_history(self, session_id: str, history: List[Dict[str, str]]):
        """保存對話歷史"""
        history_file = os.path.join(self.conversation_history_path, f"dialogue_{session_id}.json")
        async with aiofiles.open(history_file, "w", encoding="utf-8") as f:
            await f.write(json.dumps(history, ensure_ascii=False, indent=2))

    async def create_session_id(self, persona1_id: str, persona2_id: str) -> str:
        """創建新的會話 ID"""
        return f"dialogue_{persona1_id}_{persona2_id}_{str(uuid.uuid4())[:8]}"

    async def get_persona_context(
        self,
        persona_id: str,
        current_topic: str,
        conversation_history: List[Dict[str, str]],
        other_persona_name: str
    ) -> Tuple[str, List[str]]:
        """獲取 persona 的對話上下文和相關記憶"""
        index, docs, _ = await self.chatbot.interviewer_obj.load_rag_database(persona_id)
        persona_data = self.chatbot.persona_loader_obj.personas[persona_id]
        
        # 獲取相關記憶
        retrieved_docs, distances = await self.chatbot.interviewer_obj.retrieve_similar_docs(
            current_topic, index, docs
        )
        
        # 如果沒有足夠相關的記憶，返回空列表
        if len(distances) > 0 and all(dist < 0.5 for dist in distances):
            retrieved_docs = []
        
        # 建立對話上下文
        conversation_context = ""
        if conversation_history:
            conversation_context = "之前的對話:\n" + "\n".join([
                f"{exchange['speaker']}: {exchange['content']}" 
                for exchange in conversation_history[-5:]
            ])
        
        return conversation_context, retrieved_docs

    async def generate_persona_response(
        self,
        persona_id: str,
        current_topic: str,
        conversation_context: str,
        retrieved_docs: List[str],
        other_persona_name: str
    ) -> str:
        """生成 persona 的回應"""
        persona_data = self.chatbot.persona_loader_obj.personas[persona_id]
        
        prompt = [
            {
                "role": "system",
                "content": f"""你是一個根據以下記憶資料模擬出來的人格：{persona_data['簡化行為']}
                你正在與 {other_persona_name} 對話。
                記憶資料：
                {"\n".join(retrieved_docs) if retrieved_docs else "（無可參考的記憶資料）"}
                {conversation_context}
                請根據上下文判斷是否需要自我介紹，除非是初次見面或需要自我介紹的情境，否則請避免重複自我介紹或說「大家好，我是OOO，請多多指教」這類話。
                根據這些資料，用一致的口吻、用字遣詞、邏輯、口語化、對話模式來回答。你現在就是這個人，當有人問你是誰的時候，不能說你是AI助手，你要說出角色的名字。
                需要仔細衡量人格的背景，年齡和人生經歷須符合，回答的內容講求台灣的真實歷史性，不要產生不一致或不合邏輯的資料。
                仔細評估人格的語氣，情緒可以有點起伏，不要過度的正向，維持上下文的情緒。如果沒有上下文就維持平平的，無需過度開心或是憤怒。
                回應要保持連貫性，記住前面的對話內容，讓整個對話流程自然。
                """
            },
            {"role": "user", "content": current_topic}
        ]
        
        return await self.chatbot.interviewer_obj.simulate_persona_answer(prompt)

    async def conduct_dialogue(
        self,
        persona1_id: str,
        persona2_id: str,
        num_rounds: int,
        initial_topic: str,
        session_id: Optional[str] = None
    ) -> Tuple[List[Dict[str, str]], str]:
        """執行兩個 persona 之間的對話"""
        await self.initialize()
        
        # 獲取或創建會話 ID
        if not session_id:
            session_id = await self.create_session_id(persona1_id, persona2_id)
            logger.info(f"創建新的對話會話: {session_id}")
            history = []
        else:
            history = await self.load_dialogue_history(session_id)
            logger.info(f"載入會話 {session_id} 的 {len(history)} 條歷史對話")
        
        # 獲取 persona 資料
        persona1_data = self.chatbot.persona_loader_obj.personas[persona1_id]
        persona2_data = self.chatbot.persona_loader_obj.personas[persona2_id]
        
        dialogue_rounds = []
        current_topic = initial_topic
        
        for _ in range(num_rounds):
            # Persona 1 的回合
            conv_context, retrieved_docs = await self.get_persona_context(
                persona1_id, current_topic, history, persona2_data['基本資料']['姓名']
            )
            
            persona1_response = await self.generate_persona_response(
                persona1_id, current_topic, conv_context, retrieved_docs,
                persona2_data['基本資料']['姓名']
            )
            
            dialogue_rounds.append({
                "speaker": persona1_data['基本資料']['姓名'],
                "content": persona1_response
            })
            current_topic = persona1_response
            
            # Persona 2 的回合
            conv_context, retrieved_docs = await self.get_persona_context(
                persona2_id, current_topic, dialogue_rounds, persona1_data['基本資料']['姓名']
            )
            
            persona2_response = await self.generate_persona_response(
                persona2_id, current_topic, conv_context, retrieved_docs,
                persona1_data['基本資料']['姓名']
            )
            
            dialogue_rounds.append({
                "speaker": persona2_data['基本資料']['姓名'],
                "content": persona2_response
            })
            current_topic = persona2_response
        
        # 保存對話歷史
        history.extend(dialogue_rounds)
        await self.save_dialogue_history(session_id, history)
        
        return dialogue_rounds, session_id 