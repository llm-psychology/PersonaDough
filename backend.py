# backend.py
import logging
from pydantic import BaseModel
import aiofiles
from fastapi import FastAPI, Query, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import json
import os
from typing import List, Optional, Dict, Any
import uvicorn
import glob
import uuid
import asyncio
from datetime import datetime, timedelta

# Logging initialization
logging.basicConfig(
    level=logging.INFO,  # You can change to DEBUG or WARNING as needed
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

from module.QA_loader import QaLoader
from module.PERSONA_loader import PersonaLoader
from module.LLM_responder import (
    LLM_responder
)
from module.PERSONA_loader import (
    PersonaLoader
)
from module.QA_loader import (
    QaLoader
)
from interviewer.chat_with_persona import ( 
    ChatWith
)
from interviewer.interviewer_generator import (
    Interviewer,
    generate_persona_rag_by_id
)
from humanoid.humanoid_generator import (
    BaseInfoGenerator, 
    AttributeInjector, 
    StoryGenerator, 
    ToneGenerator, 
    SummarizedBehaviorGenerator, 
    AIParameterAnalyzer,
    CharacterGenerator,
    generate_a_persona
)
from interviewer.persona_dialogue import PersonaDialogueManager
from location_function_tools import location_manager
from llm_scheduler import llm_scheduler, ActivityRequest, ContextInfo

# =======================================================

# Database path
DATABASE_PATH = "humanoid/humanoid_database"
os.makedirs(DATABASE_PATH, exist_ok=True)
PHOTO_DIR = DATABASE_PATH + "/photo/"
os.makedirs(PHOTO_DIR, exist_ok=True)
character_generator = CharacterGenerator() # Initialize character generator

# New: conversation history database
CONVERSATION_HISTORY_PATH = "conversation_history"
os.makedirs(CONVERSATION_HISTORY_PATH, exist_ok=True)

# Unity memory database
UNITY_MEMORY_PATH = "unity_memory"
os.makedirs(UNITY_MEMORY_PATH, exist_ok=True)

# New: 火災模擬相關導入和初始化
from scenario_simulator import FireScenarioSimulator
import asyncio
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

# 全域火災模擬管理器
active_simulations = {}

# =======================================================

app = FastAPI(
    title="Humanoid Agent API",
    description="API for generating and managing humanoid agents, and persona interview simulator",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
'''
# Mount static files for output_internet directory
app.mount("/unity", StaticFiles(directory="output_internet", html=True), name="unity")
'''
# =======================================================
# frontend
@app.get("/")
def serve_frontend():
    return FileResponse("frontend.html", media_type="text/html")

# Unity WebGL application - redirect to proper path
# 手動掛接資料夾
@app.get("/unity-app")
def serve_unity_app():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/unity/", status_code=302)

# fire simulation
@app.get("/fire_simulation")
def serve_fire_simulation():
    return FileResponse("fire_simulation.html", media_type="text/html")

# fire simulation
@app.get("/fire_simulation_sse")
def serve_fire_simulation_sse():
    return FileResponse("fire_simulation_sse.html", media_type="text/html")

# =======================================================
# LLM 智能排程 API 端點
# =======================================================

@app.post("/api/llm-schedule/generate-activity")
async def generate_llm_activity(request: dict):
    """使用 LLM 生成單個活動"""
    try:
        persona_id = request.get("persona_id")
        time_slot = request.get("time_slot", "現在")
        duration_minutes = request.get("duration_minutes", 60)
        constraints = request.get("constraints", [])
        
        if not persona_id:
            raise HTTPException(status_code=400, detail="缺少必要參數: persona_id")
        
        # 建立上下文資訊
        context = ContextInfo(
            current_time=datetime.now().strftime("%H:%M"),
            weather=request.get("weather", "晴朗"),
            season=request.get("season", "春季"),
            available_locations=list(location_manager.locations.keys()),
            other_characters=request.get("other_characters", []),
            special_events=request.get("special_events", [])
        )
        
        # 建立活動請求
        activity_request = ActivityRequest(
            persona_id=persona_id,
            time_slot=time_slot,
            duration_minutes=duration_minutes,
            context=context,
            constraints=constraints
        )
        
        # 生成活動
        activity = await llm_scheduler.generate_activity(activity_request)
        
        return {
            "success": True,
            "activity": {
                "name": activity.name,
                "activity_type": activity.activity_type,
                "location": activity.location,
                "duration": activity.duration,
                "priority": activity.priority,
                "reasoning": activity.reasoning,
                "social_participants": activity.social_participants,
                "required_items": activity.required_items
            }
        }
        
    except Exception as e:
        logger.error(f"LLM 活動生成失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"LLM 活動生成失敗: {str(e)}")

@app.post("/api/llm-schedule/generate-daily")
async def generate_llm_daily_schedule(request: dict):
    """使用 LLM 生成一整天的排程"""
    try:
        persona_id = request.get("persona_id")
        date = request.get("date")  # YYYY-MM-DD 格式
        
        if not persona_id:
            raise HTTPException(status_code=400, detail="缺少必要參數: persona_id")
        
        # 生成一整天的排程
        activities = await llm_scheduler.generate_daily_schedule(persona_id, date)
        
        # 轉換為 API 回應格式
        schedule_data = []
        current_time = datetime.strptime("07:00", "%H:%M")
        
        for activity in activities:
            start_time = current_time.strftime("%H:%M")
            end_time = (current_time + timedelta(minutes=activity.duration)).strftime("%H:%M")
            
            schedule_data.append({
                "name": activity.name,
                "activity_type": activity.activity_type,
                "location": activity.location,
                "start_time": start_time,
                "end_time": end_time,
                "duration": activity.duration,
                "priority": activity.priority,
                "reasoning": activity.reasoning,
                "social_participants": activity.social_participants,
                "required_items": activity.required_items
            })
            
            current_time += timedelta(minutes=activity.duration)
        
        return {
            "success": True,
            "persona_id": persona_id,
            "date": date or datetime.now().strftime("%Y-%m-%d"),
            "total_activities": len(schedule_data),
            "schedule": schedule_data
        }
        
    except Exception as e:
        logger.error(f"LLM 日程生成失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"LLM 日程生成失敗: {str(e)}")

@app.post("/api/llm-schedule/apply-to-agent")
async def apply_llm_schedule_to_agent(request: dict):
    """將 LLM 生成的排程應用到代理人"""
    try:
        persona_id = request.get("persona_id")
        date = request.get("date")
        
        if not persona_id:
            raise HTTPException(status_code=400, detail="缺少必要參數: persona_id")
        
        # 生成 LLM 排程
        activities = await llm_scheduler.generate_daily_schedule(persona_id, date)
        
        # 轉換為生活模擬系統的格式
        schedule_tasks = []
        current_time = datetime.strptime("07:00", "%H:%M")
        
        for activity in activities:
            task = {
                "name": activity.name,
                "activity_type": activity.activity_type,
                "location": activity.location,
                "start_time": current_time.strftime("%H:%M"),
                "duration": activity.duration
            }
            schedule_tasks.append(task)
            current_time += timedelta(minutes=activity.duration)
        
        # 找到對應的代理人
        agent = None
        for agent_id, agent_data in life_simulation_manager.agents.items():
            if agent_data.persona_id == persona_id:
                agent = agent_data
                break
        
        if not agent:
            raise HTTPException(status_code=404, detail=f"找不到 persona_id {persona_id} 對應的代理人")
        
        # 創建並應用排程
        from life_simulation_backend import DailySchedule
        schedule_id = f"llm_schedule_{agent.agent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        schedule = DailySchedule(
            agent_id=agent.agent_id,
            schedule_id=schedule_id,
            tasks=schedule_tasks,
            is_active=True
        )
        
        life_simulation_manager.schedules[agent.agent_id] = schedule
        
        # 開始第一個任務
        if schedule_tasks:
            first_task = schedule_tasks[0]
            await life_simulation_manager.start_scheduled_task(agent, first_task)
        
        return {
            "success": True,
            "message": f"已為 {agent.name} 應用 LLM 生成的排程",
            "agent_id": agent.agent_id,
            "schedule_id": schedule_id,
            "task_count": len(schedule_tasks),
            "activities": schedule_tasks
        }
        
    except Exception as e:
        logger.error(f"應用 LLM 排程失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"應用 LLM 排程失敗: {str(e)}")

@app.get("/api/llm-schedule/personality/{persona_id}")
async def get_personality_profile(persona_id: str):
    """獲取人物個性檔案"""
    try:
        profile = await llm_scheduler.load_personality_profile(persona_id)
        
        return {
            "success": True,
            "profile": {
                "persona_id": profile.persona_id,
                "name": profile.name,
                "age": profile.age,
                "occupation": profile.occupation,
                "personality_traits": profile.personality_traits,
                "interests": profile.interests,
                "daily_preferences": profile.daily_preferences,
                "energy_pattern": profile.energy_pattern,
                "social_level": profile.social_level
            }
        }
        
    except Exception as e:
        logger.error(f"獲取個性檔案失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"獲取個性檔案失敗: {str(e)}")

# schedule control
@app.get("/schedule-control")
def serve_schedule_control():
    return FileResponse("schedule_control.html", media_type="text/html")

# LLM schedule control
@app.get("/llm-schedule-control")
def serve_llm_schedule_control():
    return FileResponse("llm_schedule_control.html", media_type="text/html")

# =======================================================

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

class HumanoidResponse(BaseModel):
    id: str
    name: str
    file_path: str

class HumanoidFilter(BaseModel):
    age_min: Optional[int] = None
    age_max: Optional[int] = None
    gender: Optional[str] = None
    birthplace: Optional[str] = None
    traits: Optional[List[str]] = None
    social_abilities: Optional[List[str]] = None
    ability_attributes: Optional[List[str]] = None

class InterviewRequest(BaseModel):
    persona_id: str | None = None
    num_rounds: int = 10
    static_interview_data: str = ""  # User's input question
    session_id: Optional[str] = None  # New: session ID to maintain conversation history

class InterviewResponse(BaseModel):
    qa_pairs: List[Dict[str, str]]
    session_id: str  # Return the session ID for the client to use in subsequent requests

class PersonaDialogueRequest(BaseModel):
    persona1_id: str
    persona2_id: str
    num_rounds: int = 5
    initial_topic: str = "你好，很高興認識你"
    session_id: Optional[str] = None
    memory_context: str = ""  # 新增記憶上下文
    is_unity_dialogue: bool = False  # 標識是否為Unity對話

class PersonaDialogueResponse(BaseModel):
    dialogue_rounds: List[Dict[str, str]]
    session_id: str

class GroupDialogueRequest(BaseModel):
    persona_ids: List[str]  # 參與群組對話的 persona ID 列表
    num_rounds: int = 5
    initial_topic: str = "大家好，很高興認識各位"
    session_id: Optional[str] = None
    # speaking_order 已廢棄，現在使用動態對話池模式

class GroupDialogueResponse(BaseModel):
    dialogue_rounds: List[Dict[str, str]]
    session_id: str
    participants: List[Dict[str, str]]  # 參與者資訊

# Unity記憶管理相關模型
class UnityMemoryEntry(BaseModel):
    persona1_id: str
    persona1_name: str
    persona2_id: str
    persona2_name: str
    speaker: str
    content: str
    timestamp: str
    round: str

class UnityMemoryRequest(BaseModel):
    persona1_id: str
    persona1_name: str
    persona2_id: str
    persona2_name: str
    speaker: str
    content: str
    round: str

class UnityMemoryResponse(BaseModel):
    success: bool
    message: str
    memory_count: int

class UnityMemoryContextRequest(BaseModel):
    persona1_id: str
    persona2_id: str

class UnityMemoryContextResponse(BaseModel):
    success: bool
    context: str
    memory_count: int
    memories: List[UnityMemoryEntry]

class GroupDialogueManager:
    def __init__(self):
        self.conversation_history_path = "conversation_history"
        os.makedirs(self.conversation_history_path, exist_ok=True)
        self.chatbot = None
        self._initialization_lock = None

    async def initialize(self):
        """初始化聊天處理器（線程安全）"""
        if self._initialization_lock is None:
            import asyncio
            self._initialization_lock = asyncio.Lock()
        
        async with self._initialization_lock:
            if not self.chatbot:
                logger.info("正在初始化 GroupDialogueManager 的 ChatWith 實例...")
                from interviewer.chat_with_persona import ChatWith
                self.chatbot = ChatWith()
                await self.chatbot.wait_until_ready()
                logger.info(f"GroupDialogueManager 初始化完成，載入了 {len(self.chatbot.persona_loader_obj.personas)} 個 persona")

    async def load_group_dialogue_history(self, session_id: str) -> List[Dict[str, str]]:
        """載入群組對話歷史"""
        history_file = os.path.join(self.conversation_history_path, f"group_dialogue_{session_id}.json")
        if os.path.exists(history_file):
            async with aiofiles.open(history_file, "r", encoding="utf-8") as f:
                content = await f.read()
                return json.loads(content)
        return []

    async def save_group_dialogue_history(self, session_id: str, history: List[Dict[str, str]]):
        """保存群組對話歷史"""
        history_file = os.path.join(self.conversation_history_path, f"group_dialogue_{session_id}.json")
        async with aiofiles.open(history_file, "w", encoding="utf-8") as f:
            await f.write(json.dumps(history, ensure_ascii=False, indent=2))

    async def create_group_session_id(self, persona_ids: List[str]) -> str:
        """創建新的群組會話 ID"""
        persona_ids_sorted = sorted(persona_ids)  # 確保順序一致
        return f"group_{'_'.join(persona_ids_sorted[:3])}_{str(uuid.uuid4())[:8]}"

    async def get_persona_context_for_group(
        self,
        persona_id: str,
        current_topic: str,
        conversation_history: List[Dict[str, str]],
        other_participants: List[str]
    ) -> tuple:
        """獲取 persona 在群組對話中的上下文和相關記憶"""
        index, docs, _ = await self.chatbot.interviewer_obj.load_rag_database(persona_id)
        persona_data = self.chatbot.persona_loader_obj.personas[persona_id]
        
        # 獲取相關記憶
        retrieved_docs, distances = await self.chatbot.interviewer_obj.retrieve_similar_docs(
            current_topic, index, docs
        )
        
        # 如果沒有足夠相關的記憶，返回空列表
        if len(distances) > 0 and all(dist < 0.5 for dist in distances):
            retrieved_docs = []
        
        # 建立群組對話上下文
        conversation_context = ""
        if conversation_history:
            conversation_context = "群組對話歷史:\n" + "\n".join([
                f"{exchange['speaker']}: {exchange['content']}" 
                for exchange in conversation_history[-8:]  # 群組對話顯示更多歷史
            ])
        
        return conversation_context, retrieved_docs

    async def generate_group_persona_response(
        self,
        persona_id: str,
        current_topic: str,
        conversation_context: str,
        retrieved_docs: List[str],
        other_participants: List[str],
        is_first_round: bool = False
    ) -> str:
        """生成 persona 在群組中的回應"""
        persona_data = self.chatbot.persona_loader_obj.personas[persona_id]
        
        participants_info = ", ".join(other_participants)
        
        group_instruction = ""
        if is_first_round:
            group_instruction = "這是群組對話的開始，你可以向大家介紹自己或回應話題。"
        else:
            group_instruction = "這是群組對話，請根據前面的對話內容自然地回應，可以對其他人的發言作出反應或提出新的話題。"
        
        from module.prompt_loader import get_user_prompt
        
        system_content = get_user_prompt("group_dialogue_system",
                                       simplified_behavior=persona_data['簡化行為'],
                                       participants_info=participants_info,
                                       memory_context="\n".join(retrieved_docs) if retrieved_docs else "（無可參考的記憶資料）",
                                       conversation_context=conversation_context,
                                       group_instruction=group_instruction)
        
        prompt = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": current_topic}
        ]
        
        return await self.chatbot.interviewer_obj.simulate_persona_answer(prompt)

    def determine_speaking_order(self, persona_ids: List[str], speaking_order: Optional[List[str]] = None) -> List[str]:
        """決定發言順序"""
        if speaking_order and len(speaking_order) == len(persona_ids):
            # 檢查指定的順序是否包含所有參與者
            if set(speaking_order) == set(persona_ids):
                return speaking_order
        
        # 如果沒有指定順序或順序不正確，則使用原始順序
        return persona_ids

    async def determine_who_wants_to_speak(
        self,
        persona_ids: List[str],
        current_topic: str,
        recent_dialogue: List[Dict[str, str]],
        persona_data_map: Dict,
        round_num: int
    ) -> List[str]:
        """決定誰想要在這一輪發言"""
        potential_speakers = []
        
        for persona_id in persona_ids:
            # 獲取角色最近的發言時間（避免同一人連續發言太多次）
            recent_speaker_count = sum(1 for d in recent_dialogue[-3:] if d.get('speaker_id') == persona_id)
            
            # 如果最近發言太多次，降低發言意願
            if recent_speaker_count >= 2:
                continue
                
            # 第一輪確保每個人至少說一句話
            if round_num == 1:
                has_spoken = any(d.get('speaker_id') == persona_id for d in recent_dialogue)
                if not has_spoken:
                    potential_speakers.append(persona_id)
                    continue
            
            # 根據角色個性和話題相關性決定發言意願
            conv_context, retrieved_docs = await self.get_persona_context_for_group(
                persona_id, current_topic, recent_dialogue, 
                [persona_data_map[pid]['基本資料']['姓名'] for pid in persona_ids if pid != persona_id]
            )
            
            # 使用AI判斷是否想要發言
            wants_to_speak = await self.check_speaking_desire(
                persona_id, current_topic, conv_context, retrieved_docs, persona_data_map[persona_id]
            )
            
            if wants_to_speak:
                potential_speakers.append(persona_id)
        
        # 如果沒有人想發言，隨機選一個人
        if not potential_speakers and persona_ids:
            import random
            potential_speakers = [random.choice(persona_ids)]
            
        return potential_speakers

    async def check_speaking_desire(
        self,
        persona_id: str,
        current_topic: str,
        conversation_context: str,
        retrieved_docs: List[str],
        persona_data: Dict
    ) -> bool:
        """檢查角色是否想要對當前話題發言"""
        
        from module.prompt_loader import get_system_prompt, get_user_prompt
        
        system_content = get_system_prompt("speaking_desire_check",
                                         simplified_behavior=persona_data['簡化行為'],
                                         memory_context="\n".join(retrieved_docs) if retrieved_docs else "（無可參考的記憶資料）",
                                         conversation_context=conversation_context)
        
        user_content = get_user_prompt("speaking_desire_check", current_topic=current_topic)
        
        prompt = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        try:
            response = await self.chatbot.interviewer_obj.simulate_persona_answer(prompt)
            # 判斷回應是否包含肯定的意思
            return "是" in response or "想" in response or "要" in response or "願意" in response
        except:
            # 如果AI判斷失敗，隨機決定（30%機率發言）
            import random
            return random.random() < 0.3

    async def conduct_group_dialogue(
        self,
        persona_ids: List[str],
        num_rounds: int,
        initial_topic: str,
        session_id: Optional[str] = None
    ) -> tuple:
        """執行群組對話 - 使用動態對話池概念"""
        await self.initialize()
        
        if len(persona_ids) < 2:
            raise ValueError("群組對話至少需要 2 個參與者")
        
        if len(persona_ids) > 10:
            raise ValueError("群組對話最多支援 10 個參與者")
        
        # 獲取或創建會話 ID
        if not session_id:
            session_id = await self.create_group_session_id(persona_ids)
            logger.info(f"創建新的群組對話會話: {session_id}")
            history = []
        else:
            history = await self.load_group_dialogue_history(session_id)
            logger.info(f"載入群組會話 {session_id} 的 {len(history)} 條歷史對話")
        
        # 獲取參與者資訊
        participants = []
        persona_data_map = {}
        for persona_id in persona_ids:
            persona_data = self.chatbot.persona_loader_obj.personas[persona_id]
            participants.append({
                "id": persona_id,
                "name": persona_data['基本資料']['姓名']
            })
            persona_data_map[persona_id] = persona_data
        
        dialogue_rounds = []
        current_topic = initial_topic
        
        # 對話池概念：每一輪可能有多人發言，也可能只有一人發言
        for round_num in range(1, num_rounds + 1):
            logger.info(f"開始第 {round_num} 輪群組對話")
            
            round_speakers = 0
            max_speakers_per_round = min(len(persona_ids), 3)  # 每輪最多3人發言
            
            while round_speakers < max_speakers_per_round:
                # 決定這一輪誰想要發言
                potential_speakers = await self.determine_who_wants_to_speak(
                    persona_ids, current_topic, history + dialogue_rounds, persona_data_map, round_num
                )
                
                if not potential_speakers:
                    break
                
                # 從想發言的人中選擇一個（可以加入優先級邏輯）
                import random
                speaker_id = random.choice(potential_speakers)
                
                # 獲取其他參與者的名字
                other_participants = [
                    persona_data_map[pid]['基本資料']['姓名'] 
                    for pid in persona_ids if pid != speaker_id
                ]
                
                # 獲取對話上下文
                conv_context, retrieved_docs = await self.get_persona_context_for_group(
                    speaker_id, 
                    current_topic, 
                    history + dialogue_rounds,
                    other_participants
                )
                
                # 生成回應
                is_first_round = (round_num == 1 and len(dialogue_rounds) == 0)
                response = await self.generate_group_persona_response(
                    speaker_id, 
                    current_topic, 
                    conv_context, 
                    retrieved_docs,
                    other_participants,
                    is_first_round
                )
                
                dialogue_rounds.append({
                    "speaker": persona_data_map[speaker_id]['基本資料']['姓名'],
                    "speaker_id": speaker_id,
                    "content": response,
                    "round": str(round_num)
                })
                
                # 更新話題為最新的回應
                current_topic = response
                round_speakers += 1
                
                logger.info(f"第{round_num}輪 - {persona_data_map[speaker_id]['基本資料']['姓名']}: {response[:50]}...")
                
                # 如果這輪已經有人發言，有機率結束這輪（讓對話更自然）
                if round_speakers >= 1 and random.random() < 0.4:  # 40%機率結束這輪
                    break
        
        # 保存對話歷史
        history.extend(dialogue_rounds)
        await self.save_group_dialogue_history(session_id, history)
        
        return dialogue_rounds, session_id, participants

# =======================================================

# New: Helper functions for conversation history management
async def load_conversation_history(session_id: str) -> List[Dict[str, str]]:
    """Load conversation history for a given session"""
    history_file = os.path.join(CONVERSATION_HISTORY_PATH, f"{session_id}.json")
    if os.path.exists(history_file):
        async with aiofiles.open(history_file, "r", encoding="utf-8") as f:
            content = await f.read()
            return json.loads(content)
    return []

async def save_conversation_history(session_id: str, history: List[Dict[str, str]]):
    """Save conversation history for a given session"""
    history_file = os.path.join(CONVERSATION_HISTORY_PATH, f"{session_id}.json")
    async with aiofiles.open(history_file, "w", encoding="utf-8") as f:
        await f.write(json.dumps(history, ensure_ascii=False, indent=2))

# Unity記憶管理輔助函數
def create_unity_memory_pair_key(persona1_id: str, persona2_id: str) -> str:
    """創建角色對的標準化鍵值"""
    if persona1_id <= persona2_id:
        return f"{persona1_id}_{persona2_id}"
    else:
        return f"{persona2_id}_{persona1_id}"

async def load_unity_memories(persona1_id: str, persona2_id: str) -> List[UnityMemoryEntry]:
    """載入兩個角色之間的記憶"""
    pair_key = create_unity_memory_pair_key(persona1_id, persona2_id)
    memory_file = os.path.join(UNITY_MEMORY_PATH, f"{pair_key}.json")
    
    if os.path.exists(memory_file):
        try:
            async with aiofiles.open(memory_file, "r", encoding="utf-8") as f:
                content = await f.read()
                # 檢查文件是否為空
                if not content.strip():
                    logger.warning(f"Unity記憶文件為空: {memory_file}")
                    return []
                data = json.loads(content)
                return [UnityMemoryEntry(**entry) for entry in data]
        except json.JSONDecodeError as e:
            logger.error(f"Unity記憶文件JSON解析錯誤: {memory_file}, 錯誤: {e}")
            # 如果文件損壞，創建備份並重新開始
            backup_file = f"{memory_file}.backup"
            try:
                import shutil
                shutil.copy2(memory_file, backup_file)
                logger.info(f"已創建備份文件: {backup_file}")
            except Exception as backup_e:
                logger.warning(f"創建備份文件失敗: {backup_e}")
            return []
        except Exception as e:
            logger.error(f"載入Unity記憶失敗: {e}")
            return []
    return []

async def save_unity_memories(persona1_id: str, persona2_id: str, memories: List[UnityMemoryEntry]):
    """保存兩個角色之間的記憶"""
    pair_key = create_unity_memory_pair_key(persona1_id, persona2_id)
    memory_file = os.path.join(UNITY_MEMORY_PATH, f"{pair_key}.json")
    
    # 轉換為字典格式保存
    memory_data = [memory.dict() for memory in memories]
    async with aiofiles.open(memory_file, "w", encoding="utf-8") as f:
        await f.write(json.dumps(memory_data, ensure_ascii=False, indent=2))

def format_unity_memories_as_context(memories: List[UnityMemoryEntry]) -> str:
    """格式化記憶為對話上下文字串"""
    if not memories:
        return "這是你們的第一次見面。"
    
    context_lines = [f"你們之前有過 {len(memories)} 次互動記錄："]
    for memory in memories:
        context_lines.append(f"- {memory.speaker}: {memory.content}")
    context_lines.append("根據這些記憶，自然地繼續對話。")
    
    return "\n".join(context_lines)

# =======================================================

@app.get("/image/{id}")
async def get_image_by_id(id: str):
    image_path = os.path.join(PHOTO_DIR, f"{id}.png")
    if os.path.exists(image_path):
        return FileResponse(image_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Image not found")

@app.post("/items")
async def create_item(item: Item):
    return item

@app.post("/interviews", response_model=InterviewResponse)
async def create_interview(request: InterviewRequest):
    """
    Handle interview requests with conversation history support.
    Uses session_id to maintain continuity between requests.
    """
    try:
        if not request.persona_id:
            raise HTTPException(status_code=400, detail="Persona ID is required")
            
        # Initialize chat handler
        chatbot = ChatWith()
        await chatbot.wait_until_ready()  # important
        
        # Get or create session ID
        session_id = request.session_id
        if not session_id:
            # Generate a new session ID if none provided
            import uuid
            session_id = f"{request.persona_id}_{str(uuid.uuid4())[:8]}"
            logger.info(f"Created new session: {session_id}")
            history = []
        else:
            # Load existing conversation history
            history = await load_conversation_history(session_id)
            logger.info(f"Loaded session {session_id} with {len(history)} previous exchanges")
        
        # Get user input from request
        user_question = request.static_interview_data
        logger.info(f"User input: {user_question}")
        
        # Load persona data and RAG database
        index, docs, qa_docs = await chatbot.interviewer_obj.load_rag_database(request.persona_id)
        
        # Format conversation history for context
        conversation_context = ""
        if history:
            conversation_context = "Previous conversation:\n" + "\n".join([
                f"User: {exchange['q']}\nPersona: {exchange['a']}" 
                for exchange in history[-5:]  # Include last 5 exchanges for context
            ])
        
        # Retrieve similar documents for context
        retrieved, distances = await chatbot.interviewer_obj.retrieve_similar_docs(user_question, index, docs)
        
        # If no similar documents with good confidence, set retrieved to None
        if len(distances) > 0 and all(dist < 0.5 for dist in distances):  # dist越小，離原話越近
            retrieved = None
        
        # Build prompt with persona data, conversation history, and context
        persona_data = chatbot.persona_loader_obj.personas[request.persona_id]
        
        # Build enhanced prompt that includes conversation history
        from module.prompt_loader import get_user_prompt
        
        system_content = get_user_prompt("interview_system_prompt",
                                       simplified_behavior=persona_data['簡化行為'],
                                       memory_context="\n".join(retrieved) if retrieved else "（無可參考的記憶資料）",
                                       conversation_context=conversation_context if conversation_context else "")
        
        prompt = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_question}
        ]
        
        # Generate AI response
        answer = await chatbot.interviewer_obj.simulate_persona_answer(prompt)
        logger.info(f"Response output: {answer}")
        
        # Add new exchange to history
        new_exchange = {"q": user_question, "a": answer}
        history.append(new_exchange)
        
        # Save updated history
        await save_conversation_history(session_id, history)
        
        # Format response
        qa_pairs = [new_exchange]  # Just return the latest exchange
        
        return InterviewResponse(qa_pairs=qa_pairs, session_id=session_id)
        
    except Exception as e:
        logger.error(f"Interview API failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# New: endpoint to clear conversation history
@app.delete("/interviews/{session_id}")
async def clear_conversation_history(session_id: str):
    """Clear the conversation history for a given session"""
    history_file = os.path.join(CONVERSATION_HISTORY_PATH, f"{session_id}.json")
    if os.path.exists(history_file):
        os.remove(history_file)
        return {"success": True, "message": f"Conversation history for session {session_id} cleared"}
    raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

@app.get("/interviews/{session_id}")
async def get_conversation_history(session_id: str):
    """Get the full conversation history for a given session"""
    try:
        history = await load_conversation_history(session_id)
        return {"success": True, "session_id": session_id, "history": history}
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving conversation history: {str(e)}")

@app.post("/generate", response_class=JSONResponse)
async def generate_humanoid(count: int = Query(1, gt=0, le=10, description="Number of humanoids to generate")):
    """Generate one or more new humanoid agents"""
    try:
        id = await generate_a_persona()
        await generate_persona_rag_by_id(id)
        return {"success": True, "count": 1, "humanoids": "please reflash the website"}
    
    except Exception as e:
        logger.error(f"Error generating humanoid: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating humanoid: {str(e)}")

@app.post("/generate-rag/{persona_id}")
async def generate_rag_database(persona_id: str = Path(..., description="ID of the persona")):
    """
    為指定的 persona 生成 RAG 資料庫
    
    ## 功能描述
    此 API 為指定的 persona 生成 RAG（Retrieval-Augmented Generation）資料庫，
    包含訪談問答、向量化處理和索引建立，用於後續的對話生成。
    
    ## 輸入參數
    - `persona_id` (str, 必需): 要生成 RAG 資料庫的 persona ID
    
    ## 輸出結果
    - `success` (bool): 操作是否成功
    - `message` (str): 結果訊息
    - `persona_id` (str): 處理的 persona ID
    - `rag_database_path` (str): 生成的資料庫路徑
    
    ## 注意事項
    - 此過程可能需要較長時間（幾分鐘）
    - 如果 RAG 資料庫已存在，將會跳過生成
    - 生成失敗時會返回詳細錯誤訊息
    """
    try:
        # 檢查 persona 是否存在
        files = glob.glob(os.path.join(DATABASE_PATH, f"{persona_id}_*.json"))
        if not files:
            raise HTTPException(status_code=404, detail=f"Persona ID {persona_id} 不存在")
        
        # 檢查 RAG 資料庫是否已存在
        rag_database_path = "interviewer/rag_database"
        persona_rag_dir = os.path.join(rag_database_path, persona_id)
        index_file = os.path.join(persona_rag_dir, "index.faiss")
        
        if os.path.exists(index_file):
            return {
                "success": True,
                "message": f"RAG 資料庫已存在，無需重新生成",
                "persona_id": persona_id,
                "rag_database_path": persona_rag_dir
            }
        
        logger.info(f"開始為 {persona_id} 生成 RAG 資料庫...")
        await generate_persona_rag_by_id(persona_id)
        logger.info(f"成功為 {persona_id} 生成 RAG 資料庫")
        
        return {
            "success": True,
            "message": f"成功為 persona {persona_id} 生成 RAG 資料庫",
            "persona_id": persona_id,
            "rag_database_path": persona_rag_dir
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"為 {persona_id} 生成 RAG 資料庫失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"生成 RAG 資料庫失敗: {str(e)}")

@app.get("/rag-status")
async def get_rag_status():
    """
    獲取所有 persona 的 RAG 資料庫狀態
    
    ## 功能描述
    檢查所有現有 persona 的 RAG 資料庫是否已建立，
    協助用戶了解哪些 persona 可以立即用於對話。
    
    ## 輸出結果
    - `total_personas` (int): 總 persona 數量
    - `rag_ready_count` (int): 已有 RAG 資料庫的 persona 數量
    - `personas` (List): 詳細的 persona 狀態列表
      - `id` (str): Persona ID
      - `name` (str): Persona 姓名
      - `rag_ready` (bool): RAG 資料庫是否就緒
      - `rag_path` (str): RAG 資料庫路徑（如果存在）
    """
    try:
        files = glob.glob(os.path.join(DATABASE_PATH, "*.json"))
        rag_database_path = "interviewer/rag_database"
        
        personas_status = []
        rag_ready_count = 0
        
        for file_path in files:
            try:
                async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                    content = await f.read()
                    data = json.loads(content)
                    
                    persona_id = data["基本資料"]["id"]
                    persona_name = data["基本資料"]["姓名"]
                    
                    # 檢查 RAG 資料庫狀態
                    persona_rag_dir = os.path.join(rag_database_path, persona_id)
                    index_file = os.path.join(persona_rag_dir, "index.faiss")
                    qa_file = os.path.join(persona_rag_dir, "qa_pairs.json")
                    
                    rag_ready = os.path.exists(index_file) and os.path.exists(qa_file)
                    if rag_ready:
                        rag_ready_count += 1
                    
                    personas_status.append({
                        "id": persona_id,
                        "name": persona_name,
                        "rag_ready": rag_ready,
                        "rag_path": persona_rag_dir if rag_ready else None
                    })
                    
            except Exception as e:
                logger.warning(f"處理檔案 {file_path} 時發生錯誤: {e}")
                continue
        
        return {
            "success": True,
            "total_personas": len(personas_status),
            "rag_ready_count": rag_ready_count,
            "personas": personas_status
        }
        
    except Exception as e:
        logger.error(f"獲取 RAG 狀態失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"獲取 RAG 狀態失敗: {str(e)}")

@app.get("/humanoids", response_class=JSONResponse)
async def list_humanoids():
    '''
    @return
    {
    "success":true,"count":6,
    "humanoids":[
    {"id":"1955f47a6e","name":"王文欽","file_path":"humanoid/humanoid_database\\1955f47a6e_王文欽.json"},
    {"id":"278c94ca40","name":"陳秀蓮","file_path":"humanoid/humanoid_database\\278c94ca40_陳秀蓮.json"},
    {"id":"64ade2a584","name":"張麗卿","file_path":"humanoid/humanoid_database\\64ade2a584_張麗卿.json"},
    {"id":"d3e6670d22","name":"陳玉梅","file_path":"humanoid/humanoid_database\\d3e6670d22_陳玉梅.json"},
    {"id":"d4c097d155","name":"許文傑","file_path":"humanoid/humanoid_database\\d4c097d155_許文傑.json"},
    {"id":"f3ed0707e3","name":"曾麗華","file_path":"humanoid/humanoid_database\\f3ed0707e3_曾麗華.json"}]}
    '''
    try:
        files = glob.glob(os.path.join(DATABASE_PATH, "*.json"))
        humanoids = []
        
        for file_path in files:
            try:
                async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                    content = await f.read()
                    data = json.loads(content)
                    
                    humanoids.append({
                        "id": data["基本資料"]["id"],
                        "name": data["基本資料"]["姓名"],
                        "file_path": file_path,
                        "traits": data.get("人格屬性", {}).get("人格特質", []),
                        "social_abilities": data.get("人格屬性", {}).get("社交能力", []),
                        "ability_attributes": data.get("人格屬性", {}).get("能力屬性", [])
                    })
            except Exception as e:
                logger.warning(f"Error reading file {file_path}: {e}")
                continue
        logger.info(f"Listed {len(humanoids)} humanoids.")
        return {"success": True, "count": len(humanoids), "humanoids": humanoids}
    
    except Exception as e:
        logger.error(f"Error listing humanoids: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing humanoids: {str(e)}")

@app.get("/humanoids/{humanoid_id}", response_class=JSONResponse)
async def get_humanoid(humanoid_id: str = Path(..., description="ID of the humanoid agent")):
    '''
    @return
    {
    "基本資料": {
        "id": "d4c097d155",
        "姓名": "許文傑",
        "年紀": 61,
        "性別": "男",
        "出生地": "花蓮"
    },
    "人格屬性": {
        "人格特質": [
        "足智多謀"
        ],
        "社交能力": [
        "關懷他人"
        ],
        "能力屬性": [
        "衝突解決",
        "生存技能"
        ]
    },
    "生平故事": "許文傑，生於花蓮，這個四面環山的地方孕育...",
    "語言行為": "許文傑是一位充滿智慧和關懷的律師...",
    "簡化行為": "許文傑，...",
    "AI參數": {
        "temperature": 0.7,
        "top_p": 0.8,
        "reason": "故事內容豐富且具有創意，需較高的多樣性與表現力。"
    }
    }
    '''
    try:
        # Find the file with the specified ID
        files = glob.glob(os.path.join(DATABASE_PATH, f"{humanoid_id}_*.json"))
        if not files:
            logger.warning(f"Humanoid with ID {humanoid_id} not found")
            raise HTTPException(status_code=404, detail=f"Humanoid with ID {humanoid_id} not found")
        
        file_path = files[0]
        
        # Read the file
        import aiofiles
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            content = await f.read()
            data = json.loads(content) 
        logger.info(f"Fetched humanoid {humanoid_id}")
        return {"success": True, "humanoid": data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting humanoid: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting humanoid: {str(e)}")

@app.post("/search", response_class=JSONResponse)
async def search_humanoids(filter_params: HumanoidFilter):
    """Search for humanoid agents using filters"""
    try:
        files = glob.glob(os.path.join(DATABASE_PATH, "*.json"))
        matching_humanoids = []
        
        for file_path in files:
            try:
                async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                    content = await f.read()
                    data = json.loads(content)
                    
                    # Check if the humanoid matches all filters
                    matches = True
                    
                    # Age filter
                    if filter_params.age_min is not None and data["基本資料"]["年紀"] < filter_params.age_min:
                        matches = False
                    if filter_params.age_max is not None and data["基本資料"]["年紀"] > filter_params.age_max:
                        matches = False
                    
                    # Gender filter
                    if filter_params.gender and data["基本資料"]["性別"] != filter_params.gender:
                        matches = False
                    
                    # Birthplace filter
                    if filter_params.birthplace and filter_params.birthplace not in data["基本資料"]["出生地"]:
                        matches = False
                    
                    # Traits filter
                    if filter_params.traits:
                        if not all(trait in data["人格屬性"]["人格特質"] for trait in filter_params.traits):
                            matches = False
                    
                    # Social abilities filter
                    if filter_params.social_abilities:
                        if not all(ability in data["人格屬性"]["社交能力"] for ability in filter_params.social_abilities):
                            matches = False
                    
                    # Ability attributes filter
                    if filter_params.ability_attributes:
                        if not all(attr in data["人格屬性"]["能力屬性"] for attr in filter_params.ability_attributes):
                            matches = False
                    
                    if matches:
                        matching_humanoids.append({
                            "id": data["基本資料"]["id"],
                            "name": data["基本資料"]["姓名"],
                            "file_path": file_path,
                            "data": data
                        })
            except Exception as e:
                logger.warning(f"Error processing file {file_path}: {e}")
                continue
        logger.info(f"Search result count: {len(matching_humanoids)}")
        return {
            "success": True, 
            "count": len(matching_humanoids), 
            "humanoids": matching_humanoids
        }
    
    except Exception as e:
        logger.error(f"Error searching humanoids: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error searching humanoids: {str(e)}")

# 初始化對話管理器
dialogue_manager = PersonaDialogueManager()
group_dialogue_manager = GroupDialogueManager()

@app.get("/persona/{persona_id}/name")
async def get_persona_name(persona_id: str):
    """
    根據 persona ID 獲取角色姓名
    
    ## 功能描述
    此 API 端點可以根據 persona ID 返回對應角色的姓名。
    主要用於 Unity 客戶端自動獲取角色姓名，避免手動設定。
    
    ## 輸入參數 (Input)
    - `persona_id` (str, 必需): persona 的唯一 ID
    
    ## 輸出結果 (Output)
    - `persona_id` (str): 輸入的 persona ID
    - `name` (str): 角色姓名
    - `success` (bool): 是否成功獲取姓名
    
    ## 使用範例
    **請求範例:**
    ```
    GET /persona/867b21d0fe/name
    ```
    
    **回應範例:**
    ```json
    {
        "persona_id": "867b21d0fe",
        "name": "何素珍",
        "success": true
    }
    ```
    
    ## 注意事項
    - 如果 persona ID 不存在，返回 404 錯誤
    - 確保 persona_loader 已經初始化
    """
    try:
        # 確保 persona loader 已初始化
        if not hasattr(app.state, 'persona_loader'):
            from module.PERSONA_loader import PersonaLoader
            app.state.persona_loader = PersonaLoader()
        
        await app.state.persona_loader.wait_until_ready()
        
        # 獲取 persona 資料
        persona_data = app.state.persona_loader.get_persona_by_id(persona_id)
        if not persona_data:
            raise HTTPException(status_code=404, detail=f"找不到 persona ID: {persona_id}")
        
        # 提取姓名
        name = persona_data.get('基本資料', {}).get('姓名', '')
        if not name:
            raise HTTPException(status_code=500, detail=f"persona {persona_id} 缺少姓名資料")
        
        return {
            "persona_id": persona_id,
            "name": name,
            "success": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get persona name API failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/persona-dialogue", response_model=PersonaDialogueResponse)
async def create_persona_dialogue(request: PersonaDialogueRequest):
    """
    讓兩個 persona 進行對話
    
    ## 功能描述
    此 API 端點可以讓兩個不同的 persona（人格角色）進行自然的對話交流。
    系統會根據每個 persona 的人格特質、背景故事和記憶資料來生成符合其特色的對話內容。
    
    ## 輸入參數 (Input)
    **PersonaDialogueRequest:**
    - `persona1_id` (str, 必需): 第一個對話者的 persona ID
    - `persona2_id` (str, 必需): 第二個對話者的 persona ID  
    - `num_rounds` (int, 可選): 對話輪數，預設為 5 輪
    - `initial_topic` (str, 可選): 對話起始話題，預設為 "你好，很高興認識你"
    - `session_id` (str, 可選): 會話 ID，用於延續之前的對話，若未提供則創建新會話
    
    ## 輸出結果 (Output)
    **PersonaDialogueResponse:**
    - `dialogue_rounds` (List[Dict[str, str]]): 對話輪次列表，每輪包含：
      - `speaker`: 發言者姓名
      - `content`: 發言內容
      - `round`: 輪次編號
    - `session_id` (str): 會話 ID，可用於後續延續對話
    
    ## 使用範例
    **請求範例:**
    ```json
    {
        "persona1_id": "abc123",
        "persona2_id": "def456", 
        "num_rounds": 3,
        "initial_topic": "今天天氣真好呢",
        "session_id": null
    }
    ```
    
    **回應範例:**
    ```json
    {
        "dialogue_rounds": [
            {
                "speaker": "王小明",
                "content": "是啊，今天陽光普照，讓人心情都變好了。",
                "round": "1"
            },
            {
                "speaker": "李小華", 
                "content": "沒錯！這種天氣最適合出去走走了。",
                "round": "1"
            }
        ],
        "session_id": "dialogue_abc123_def456_x1y2z3"
    }
    ```
    
    ## 注意事項
    - 兩個 persona ID 必須存在於系統中
    - 對話內容會根據 persona 的人格特質和背景產生
    - 可以使用返回的 session_id 來延續對話
    - 若 persona 的 RAG 資料庫不存在，系統會自動生成（可能需要較長時間）
    - 首次使用新 persona 進行對話時，回應時間可能較長
    """
    try:
        # 檢查並確保兩個 persona 的 RAG 資料庫存在
        rag_database_path = "interviewer/rag_database"
        
        for persona_id in [request.persona1_id, request.persona2_id]:
            persona_rag_dir = os.path.join(rag_database_path, persona_id)
            index_file = os.path.join(persona_rag_dir, "index.faiss")
            
            if not os.path.exists(index_file):
                logger.info(f"RAG 資料庫不存在於 {persona_id}，正在自動生成...")
                try:
                    await generate_persona_rag_by_id(persona_id)
                    logger.info(f"成功為 {persona_id} 生成 RAG 資料庫")
                except Exception as e:
                    logger.error(f"為 {persona_id} 生成 RAG 資料庫失敗: {e}")
                    raise HTTPException(
                        status_code=500, 
                        detail=f"無法為 persona {persona_id} 生成必要的資料庫: {str(e)}"
                    )
        
        # 確保 dialogue_manager 已初始化
        await dialogue_manager.initialize()
        
        # 如果是Unity對話，獲取Unity記憶上下文
        unity_memory_context = ""
        if request.is_unity_dialogue:
            try:
                unity_memories = await load_unity_memories(request.persona1_id, request.persona2_id)
                unity_memory_context = format_unity_memories_as_context(unity_memories)
                logger.info(f"載入Unity記憶上下文，共 {len(unity_memories)} 條記憶")
            except Exception as e:
                logger.warning(f"載入Unity記憶失敗: {e}")
                unity_memory_context = "這是你們的第一次見面。"
        
        # 如果有記憶上下文，將其整合到 initial_topic 中
        enhanced_topic = request.initial_topic
        if request.memory_context:
            enhanced_topic = f"{request.memory_context}\n\n當前話題: {request.initial_topic}"
            logger.info(f"整合記憶上下文到對話中，長度: {len(request.memory_context)} 字元")
        
        dialogue_rounds, session_id = await dialogue_manager.conduct_dialogue(
            persona1_id=request.persona1_id,
            persona2_id=request.persona2_id,
            num_rounds=request.num_rounds,
            initial_topic=enhanced_topic,
            session_id=request.session_id,
            is_unity_dialogue=request.is_unity_dialogue,
            unity_memory_context=unity_memory_context
        )
        
        return PersonaDialogueResponse(
            dialogue_rounds=dialogue_rounds,
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"Persona dialogue API failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/group-dialogue", response_model=GroupDialogueResponse)
async def create_group_dialogue(request: GroupDialogueRequest):
    """
    讓多個 persona 進行群組對話
    """
    try:
        # 驗證參與者數量
        if len(request.persona_ids) < 2:
            raise HTTPException(status_code=400, detail="群組對話至少需要 2 個參與者")
        
        if len(request.persona_ids) > 10:
            raise HTTPException(status_code=400, detail="群組對話最多支援 10 個參與者")
        
        # 驗證所有 persona_id 是否存在
        for persona_id in request.persona_ids:
            if persona_id not in group_dialogue_manager.chatbot.persona_loader_obj.personas if group_dialogue_manager.chatbot else True:
                # 先初始化以檢查 persona 是否存在
                await group_dialogue_manager.initialize()
                if persona_id not in group_dialogue_manager.chatbot.persona_loader_obj.personas:
                    raise HTTPException(status_code=404, detail=f"Persona ID {persona_id} 不存在")
        
        dialogue_rounds, session_id, participants = await group_dialogue_manager.conduct_group_dialogue(
            persona_ids=request.persona_ids,
            num_rounds=request.num_rounds,
            initial_topic=request.initial_topic,
            session_id=request.session_id
        )
        
        return GroupDialogueResponse(
            dialogue_rounds=dialogue_rounds,
            session_id=session_id,
            participants=participants
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Group dialogue API failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/group-dialogue/{session_id}")
async def get_group_dialogue_history(session_id: str):
    """獲取群組對話的完整歷史記錄"""
    try:
        history = await group_dialogue_manager.load_group_dialogue_history(session_id)
        return {"success": True, "session_id": session_id, "history": history}
    except Exception as e:
        logger.error(f"Error retrieving group dialogue history: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving group dialogue history: {str(e)}")

@app.delete("/group-dialogue/{session_id}")
async def clear_group_dialogue_history(session_id: str):
    """清除群組對話的歷史記錄"""
    history_file = os.path.join(CONVERSATION_HISTORY_PATH, f"group_dialogue_{session_id}.json")
    if os.path.exists(history_file):
        os.remove(history_file)
        return {"success": True, "message": f"群組對話歷史記錄 {session_id} 已清除"}
    raise HTTPException(status_code=404, detail=f"會話 {session_id} 不存在")

# 測試端點
@app.get("/test-unity")
async def test_unity_endpoint():
    """測試Unity端點是否工作"""
    return {"message": "Unity測試端點正常工作", "success": True}

# =======================================================
# 生活模擬 API 端點
# =======================================================

from life_simulation_backend import life_simulation_manager, LifeSimulationRequest, LifeSimulationResponse

@app.get("/api/life-simulation/start")
async def start_life_simulation():
    """啟動生活模擬"""
    try:
        request = LifeSimulationRequest(agent_count=3, time_speed=60.0)
        response = await life_simulation_manager.start_simulation(request)
        return response.dict()
    except Exception as e:
        logger.error(f"啟動生活模擬失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"啟動生活模擬失敗: {str(e)}")

@app.get("/api/life-simulation/auto-start")
async def auto_start_life_simulation():
    """自動啟動生活模擬（當Unity第一次連接時）"""
    try:
        # 檢查是否已經有活躍的模擬
        if life_simulation_manager.agents:
            return {
                "success": True,
                "message": "生活模擬已經在運行",
                "agent_count": len(life_simulation_manager.agents)
            }
        
        # 自動啟動模擬
        request = LifeSimulationRequest(agent_count=3, time_speed=60.0)
        response = await life_simulation_manager.start_simulation(request)
        
        logger.info(f"自動啟動生活模擬: {response.message}")
        return {
            "success": response.success,
            "message": response.message,
            "agent_count": len(response.agents)
        }
        
    except Exception as e:
        logger.error(f"自動啟動生活模擬失敗: {e}", exc_info=True)
        return {
            "success": False,
            "message": f"自動啟動失敗: {str(e)}",
            "agent_count": 0
        }

@app.get("/api/life-simulation/agent-status/{agent_id}")
async def get_agent_status(agent_id: str):
    """獲取指定角色的狀態"""
    try:
        # 首先嘗試直接通過agent_id獲取
        agent_status = life_simulation_manager.get_agent_status(agent_id)
        if agent_status:
            from dataclasses import asdict
            return asdict(agent_status)
        
        # 如果找不到，嘗試通過persona_id獲取
        agent_status = life_simulation_manager.get_agent_by_persona_id(agent_id)
        if agent_status:
            from dataclasses import asdict
            return asdict(agent_status)
        
        # 如果還是找不到，創建一個默認狀態
        logger.warning(f"角色 {agent_id} 不存在，創建默認狀態")
        return await life_simulation_manager.create_default_agent_status(agent_id)
        
    except Exception as e:
        logger.error(f"獲取角色狀態失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"獲取角色狀態失敗: {str(e)}")

@app.get("/api/life-simulation/all-agents-status")
async def get_all_agents_status():
    """獲取所有角色狀態"""
    try:
        agents = life_simulation_manager.get_all_agents_status()
        return {
            "success": True,
            "agent_count": len(agents),
            "agents": agents
        }
    except Exception as e:
        logger.error(f"獲取所有角色狀態失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"獲取所有角色狀態失敗: {str(e)}")

@app.post("/api/life-simulation/agent/{agent_id}/command")
async def execute_agent_command(agent_id: str, command_data: dict):
    """執行代理人指令"""
    try:
        command = command_data.get("command")
        params = command_data.get("params", {})
        
        result = await life_simulation_manager.execute_agent_command(agent_id, command, params)
        return result
        
    except Exception as e:
        logger.error(f"執行代理人指令失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"執行指令失敗: {str(e)}")

@app.post("/api/life-simulation/stop")
async def stop_life_simulation():
    """停止生活模擬"""
    try:
        life_simulation_manager.stop_simulation()
        return {"success": True, "message": "生活模擬已停止"}
    except Exception as e:
        logger.error(f"停止生活模擬失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"停止生活模擬失敗: {str(e)}")

# =======================================================
# 每日行程管理 API 端點
# =======================================================

@app.post("/api/schedule/apply-to-all")
async def apply_schedule_to_all_agents(schedule_type: str = "balanced"):
    """
    一鍵為所有代理人安排每日行程
    
    ## 功能描述
    為所有現有的代理人自動安排指定的每日行程，讓他們按照固定時間表執行日常活動。
    
    ## 輸入參數
    - `schedule_type` (str): 行程類型，可選值：
      - `student`: 學生行程（上學、放學、踢足球）
      - `worker`: 上班族行程（上班、下班、社交）
      - `balanced`: 平衡行程（學習、運動、社交）
    
    ## 輸出結果
    - `success` (bool): 是否成功
    - `message` (str): 結果訊息
    - `total_agents` (int): 總代理人數量
    - `success_count` (int): 成功安排行程的代理人數量
    - `results` (List): 每個代理人的安排結果
    """
    try:
        result = await life_simulation_manager.apply_schedule_to_all_agents(schedule_type)
        return result
        
    except Exception as e:
        logger.error(f"批量安排行程失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"批量安排行程失敗: {str(e)}")

@app.post("/api/schedule/apply-to-agent/{agent_id}")
async def apply_schedule_to_single_agent(agent_id: str, schedule_type: str = "balanced"):
    """
    為指定代理人安排每日行程
    
    ## 功能描述
    為指定的代理人安排每日行程，讓他按照固定時間表執行日常活動。
    
    ## 輸入參數
    - `agent_id` (str): 代理人ID
    - `schedule_type` (str): 行程類型
    
    ## 輸出結果
    - `success` (bool): 是否成功
    - `message` (str): 結果訊息
    - `schedule_id` (str): 行程ID
    - `task_count` (int): 任務數量
    """
    try:
        result = await life_simulation_manager.apply_schedule_to_agent(agent_id, schedule_type)
        return result
        
    except Exception as e:
        logger.error(f"安排代理人行程失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"安排代理人行程失敗: {str(e)}")

@app.get("/api/schedule/agent/{agent_id}")
async def get_agent_schedule(agent_id: str):
    """
    獲取指定代理人的行程安排
    
    ## 功能描述
    獲取指定代理人的詳細行程安排，包括所有任務和當前執行狀態。
    
    ## 輸入參數
    - `agent_id` (str): 代理人ID
    
    ## 輸出結果
    - `agent_id` (str): 代理人ID
    - `schedule_id` (str): 行程ID
    - `is_active` (bool): 行程是否啟用
    - `created_at` (str): 創建時間
    - `tasks` (List): 任務列表
    - `current_task` (Dict): 當前任務
    """
    try:
        schedule_info = life_simulation_manager.get_agent_schedule(agent_id)
        if not schedule_info:
            raise HTTPException(status_code=404, detail=f"代理人 {agent_id} 沒有行程安排")
        
        return schedule_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"獲取代理人行程失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"獲取代理人行程失敗: {str(e)}")

@app.get("/api/schedule/all")
async def get_all_schedules():
    """
    獲取所有代理人的行程安排
    
    ## 功能描述
    獲取所有代理人的行程安排概覽，包括有行程和沒有行程的代理人統計。
    
    ## 輸出結果
    - `total_agents` (int): 總代理人數量
    - `scheduled_agents` (int): 有行程的代理人數量
    - `schedules` (Dict): 所有行程詳細資訊
    """
    try:
        schedules_info = life_simulation_manager.get_all_schedules()
        return schedules_info
        
    except Exception as e:
        logger.error(f"獲取所有行程失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"獲取所有行程失敗: {str(e)}")

@app.post("/api/schedule/toggle/{agent_id}")
async def toggle_agent_schedule(agent_id: str, is_active: bool = None):
    """
    切換代理人的行程狀態
    
    ## 功能描述
    啟用或停用指定代理人的行程安排。
    
    ## 輸入參數
    - `agent_id` (str): 代理人ID
    - `is_active` (bool, 可選): 是否啟用，如果不提供則切換當前狀態
    
    ## 輸出結果
    - `success` (bool): 是否成功
    - `message` (str): 結果訊息
    - `is_active` (bool): 當前狀態
    """
    try:
        result = life_simulation_manager.toggle_schedule(agent_id, is_active)
        return result
        
    except Exception as e:
        logger.error(f"切換行程狀態失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"切換行程狀態失敗: {str(e)}")

@app.delete("/api/schedule/agent/{agent_id}")
async def clear_agent_schedule(agent_id: str):
    """
    清除指定代理人的行程安排
    
    ## 功能描述
    清除指定代理人的所有行程安排，讓他回到自主決策模式。
    
    ## 輸入參數
    - `agent_id` (str): 代理人ID
    
    ## 輸出結果
    - `success` (bool): 是否成功
    - `message` (str): 結果訊息
    """
    try:
        result = life_simulation_manager.clear_agent_schedule(agent_id)
        return result
        
    except Exception as e:
        logger.error(f"清除代理人行程失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"清除代理人行程失敗: {str(e)}")

@app.delete("/api/schedule/clear-all")
async def clear_all_schedules():
    """
    清除所有代理人的行程安排
    
    ## 功能描述
    清除所有代理人的行程安排，讓他們都回到自主決策模式。
    
    ## 輸出結果
    - `success` (bool): 是否成功
    - `message` (str): 結果訊息
    """
    try:
        result = life_simulation_manager.clear_all_schedules()
        return result
        
    except Exception as e:
        logger.error(f"清除所有行程失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"清除所有行程失敗: {str(e)}")

@app.get("/api/schedule/templates")
async def get_schedule_templates():
    """
    獲取可用的行程模板
    
    ## 功能描述
    獲取所有可用的每日行程模板，包括學生、上班族和平衡型行程。
    
    ## 輸出結果
    - `templates` (Dict): 行程模板列表
    - `template_count` (int): 模板數量
    """
    try:
        templates = life_simulation_manager.default_schedules
        return {
            "templates": templates,
            "template_count": len(templates),
            "available_types": list(templates.keys())
        }
        
    except Exception as e:
        logger.error(f"獲取行程模板失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"獲取行程模板失敗: {str(e)}")

# =======================================================
# 位置控制 API 端點
# =======================================================

@app.post("/api/location/move-character")
async def move_character_to_location_api(request: dict):
    """讓角色移動到指定位置"""
    try:
        character_id = request.get("character_id")
        location_id = request.get("location_id")
        movement_speed = request.get("movement_speed", 1.0)
        
        if not character_id or not location_id:
            raise HTTPException(status_code=400, detail="缺少必要參數: character_id, location_id")
        
        # 執行移動指令
        result = await life_simulation_manager.execute_agent_command(
            character_id, 
            "move_to", 
            {"location": location_id, "speed": movement_speed}
        )
        
        logger.info(f"角色移動請求: {character_id} -> {location_id}")
        return result
        
    except Exception as e:
        logger.error(f"角色移動失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"角色移動失敗: {str(e)}")

@app.post("/api/location/move-coordinates")
async def move_character_to_coordinates_api(request: dict):
    """讓角色移動到指定座標"""
    try:
        character_id = request.get("character_id")
        x = request.get("x")
        y = request.get("y")
        movement_speed = request.get("movement_speed", 1.0)
        
        if character_id is None or x is None or y is None:
            raise HTTPException(status_code=400, detail="缺少必要參數: character_id, x, y")
        
        result = await life_simulation_manager.execute_agent_command(
            character_id,
            "move_to_coordinates",
            {"x": float(x), "y": float(y), "speed": movement_speed}
        )
        
        logger.info(f"角色移動到座標: {character_id} -> ({x}, {y})")
        return result
        
    except Exception as e:
        logger.error(f"角色移動到座標失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"角色移動到座標失敗: {str(e)}")

@app.get("/api/location/character/{character_id}")
async def get_character_location_api(character_id: str):
    """獲取角色位置資訊"""
    try:
        agent = life_simulation_manager.get_agent_by_persona_id(character_id)
        if not agent:
            agent = life_simulation_manager.get_agent_status(character_id)
        
        if agent:
            return {
                "success": True,
                "character_id": character_id,
                "current_location": agent.current_location,
                "position": agent.position,
                "is_moving": agent.is_moving
            }
        else:
            return {
                "success": False,
                "message": f"找不到角色: {character_id}"
            }
    except Exception as e:
        logger.error(f"獲取角色位置失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"獲取角色位置失敗: {str(e)}")

@app.get("/api/location/all-locations")
async def get_all_locations_api():
    """獲取所有可用位置"""
    try:
        # 使用 location_manager 獲取所有位置
        locations = []
        for location in location_manager.locations.values():
            locations.append({
                "id": location.id,
                "name": location.name,
                "type": location.type,
                "position": location.position,
                "description": location.description
            })
        
        return {
            "success": True,
            "total_count": len(locations),
            "locations": locations
        }
    except Exception as e:
        logger.error(f"獲取位置列表失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"獲取位置列表失敗: {str(e)}")

@app.get("/api/location/{location_id}")
async def get_location_by_id_api(location_id: str):
    """根據ID獲取特定位置資訊"""
    try:
        location = location_manager.get_location(location_id)
        if not location:
            raise HTTPException(status_code=404, detail=f"位置 {location_id} 不存在")
        
        return {
            "success": True,
            "location": {
                "id": location.id,
                "name": location.name,
                "type": location.type,
                "position": location.position,
                "description": location.description
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"獲取位置資訊失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"獲取位置資訊失敗: {str(e)}")

@app.get("/api/location/type/{location_type}")
async def get_locations_by_type_api(location_type: str):
    """根據類型獲取所有位置"""
    try:
        locations = location_manager.get_all_locations_by_type(location_type)
        location_list = []
        
        for location in locations:
            location_list.append({
                "id": location.id,
                "name": location.name,
                "type": location.type,
                "position": location.position,
                "description": location.description
            })
        
        return {
            "success": True,
            "location_type": location_type,
            "total_count": len(location_list),
            "locations": location_list
        }
    except Exception as e:
        logger.error(f"獲取類型位置失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"獲取類型位置失敗: {str(e)}")

@app.get("/api/location/character/{character_id}/preferred")
async def get_character_preferred_locations_api(character_id: str):
    """獲取角色的偏好位置"""
    try:
        memory = location_manager.get_character_memory(character_id)
        if not memory:
            return {
                "success": True,
                "character_id": character_id,
                "message": "角色沒有位置記憶",
                "preferred_locations": {}
            }
        
        preferred_locations = {}
        for location_type, location_id in memory.preferred_locations.items():
            location = location_manager.get_location(location_id)
            if location:
                preferred_locations[location_type] = {
                    "id": location.id,
                    "name": location.name,
                    "type": location.type,
                    "position": location.position,
                    "description": location.description
                }
        
        return {
            "success": True,
            "character_id": character_id,
            "preferred_locations": preferred_locations,
            "known_locations": memory.known_locations,
            "home_location": memory.home_location_id
        }
    except Exception as e:
        logger.error(f"獲取角色偏好位置失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"獲取角色偏好位置失敗: {str(e)}")

@app.post("/api/location/character/{character_id}/preferred")
async def set_character_preferred_location_api(character_id: str, request: dict):
    """設定角色的偏好位置"""
    try:
        location_type = request.get("location_type")
        location_id = request.get("location_id")
        
        if not location_type or not location_id:
            raise HTTPException(status_code=400, detail="缺少必要參數: location_type, location_id")
        
        # 檢查位置是否存在
        location = location_manager.get_location(location_id)
        if not location:
            raise HTTPException(status_code=404, detail=f"位置 {location_id} 不存在")
        
        # 設定偏好位置
        location_manager.set_character_preferred_location(character_id, location_type, location_id)
        
        return {
            "success": True,
            "message": f"已設定角色 {character_id} 的 {location_type} 偏好為 {location_id}",
            "character_id": character_id,
            "location_type": location_type,
            "location_id": location_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"設定角色偏好位置失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"設定角色偏好位置失敗: {str(e)}")

# =======================================================
# Unity專用對話API端點
# =======================================================

@app.post("/unity-api/dialogue", response_model=PersonaDialogueResponse)
async def create_unity_dialogue(request: PersonaDialogueRequest):
    """
    Unity專用的對話API
    
    ## 功能描述
    專為Unity遊戲設計的對話API，自動使用Unity記憶系統，
    並生成以'unity'結尾的會話ID。
    
    ## 特點
    - 自動載入Unity記憶上下文
    - 會話ID格式: dialogue_{persona1_id}_{persona2_id}_unity
    - 整合Unity記憶系統
    - 自動保存對話到Unity記憶中
    """
    try:
        # 強制設定為Unity對話
        request.is_unity_dialogue = True
        
        # 檢查並確保兩個 persona 的 RAG 資料庫存在
        rag_database_path = "interviewer/rag_database"
        
        for persona_id in [request.persona1_id, request.persona2_id]:
            persona_rag_dir = os.path.join(rag_database_path, persona_id)
            index_file = os.path.join(persona_rag_dir, "index.faiss")
            
            if not os.path.exists(index_file):
                logger.info(f"RAG 資料庫不存在於 {persona_id}，正在自動生成...")
                try:
                    await generate_persona_rag_by_id(persona_id)
                    logger.info(f"成功為 {persona_id} 生成 RAG 資料庫")
                except Exception as e:
                    logger.error(f"為 {persona_id} 生成 RAG 資料庫失敗: {e}")
                    raise HTTPException(
                        status_code=500, 
                        detail=f"無法為 persona {persona_id} 生成必要的資料庫: {str(e)}"
                    )
        
        # 確保 dialogue_manager 已初始化
        await dialogue_manager.initialize()
        
        # 載入Unity記憶上下文
        unity_memory_context = ""
        try:
            unity_memories = await load_unity_memories(request.persona1_id, request.persona2_id)
            unity_memory_context = format_unity_memories_as_context(unity_memories)
            logger.info(f"Unity對話載入記憶上下文，共 {len(unity_memories)} 條記憶")
        except Exception as e:
            logger.warning(f"載入Unity記憶失敗: {e}")
            unity_memory_context = "這是你們的第一次見面。"
        
        # 整合記憶上下文到初始話題
        enhanced_topic = request.initial_topic
        if request.memory_context:
            enhanced_topic = f"{request.memory_context}\n\n當前話題: {request.initial_topic}"
            logger.info(f"整合額外記憶上下文到Unity對話中，長度: {len(request.memory_context)} 字元")
        
        # 執行對話
        dialogue_rounds, session_id = await dialogue_manager.conduct_dialogue(
            persona1_id=request.persona1_id,
            persona2_id=request.persona2_id,
            num_rounds=request.num_rounds,
            initial_topic=enhanced_topic,
            session_id=request.session_id,
            is_unity_dialogue=True,
            unity_memory_context=unity_memory_context
        )
        
        logger.info(f"Unity對話完成，會話ID: {session_id}")
        
        return PersonaDialogueResponse(
            dialogue_rounds=dialogue_rounds,
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"Unity對話API失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# =======================================================
# Unity記憶管理 API 端點
# =======================================================

@app.post("/unity-api/memory/add", response_model=UnityMemoryResponse)
async def add_unity_memory(request: UnityMemoryRequest):
    """
    添加Unity對話記憶
    
    ## 功能描述
    為Unity遊戲中的角色對話添加記憶記錄，支援角色間的互動歷史追蹤。
    
    ## 輸入參數
    - `persona1_id` (str): 第一個角色ID
    - `persona1_name` (str): 第一個角色姓名
    - `persona2_id` (str): 第二個角色ID
    - `persona2_name` (str): 第二個角色姓名
    - `speaker` (str): 發言者姓名
    - `content` (str): 對話內容
    - `round` (str): 對話輪次
    
    ## 輸出結果
    - `success` (bool): 是否成功添加記憶
    - `message` (str): 操作結果訊息
    - `memory_count` (int): 該角色對的總記憶數量
    """
    try:
        # 載入現有記憶
        existing_memories = await load_unity_memories(request.persona1_id, request.persona2_id)
        
        # 創建新記憶條目
        from datetime import datetime
        new_memory = UnityMemoryEntry(
            persona1_id=request.persona1_id,
            persona1_name=request.persona1_name,
            persona2_id=request.persona2_id,
            persona2_name=request.persona2_name,
            speaker=request.speaker,
            content=request.content,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            round=request.round
        )
        
        # 添加到記憶列表
        existing_memories.append(new_memory)
        
        # 限制記憶數量（最多保留10條）
        max_memories = 10
        if len(existing_memories) > max_memories:
            existing_memories = existing_memories[-max_memories:]
        
        # 保存記憶
        await save_unity_memories(request.persona1_id, request.persona2_id, existing_memories)
        
        logger.info(f"Unity記憶已添加: {request.speaker} -> {request.content[:50]}...")
        
        return UnityMemoryResponse(
            success=True,
            message=f"成功添加記憶，{request.speaker}: {request.content[:30]}...",
            memory_count=len(existing_memories)
        )
        
    except Exception as e:
        logger.error(f"添加Unity記憶失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"添加記憶失敗: {str(e)}")

@app.post("/unity-api/memory/context", response_model=UnityMemoryContextResponse)
async def get_unity_memory_context(request: UnityMemoryContextRequest):
    """
    獲取Unity角色對話的記憶上下文
    
    ## 功能描述
    獲取兩個角色之間的對話記憶，格式化為上下文字串，用於Unity中的對話系統。
    
    ## 輸入參數
    - `persona1_id` (str): 第一個角色ID
    - `persona2_id` (str): 第二個角色ID
    
    ## 輸出結果
    - `success` (bool): 是否成功獲取記憶
    - `context` (str): 格式化的記憶上下文字串
    - `memory_count` (int): 記憶條目數量
    - `memories` (List[UnityMemoryEntry]): 詳細的記憶列表
    """
    try:
        # 載入記憶
        memories = await load_unity_memories(request.persona1_id, request.persona2_id)
        
        # 格式化為上下文
        context = format_unity_memories_as_context(memories)
        
        logger.info(f"Unity記憶上下文獲取: {request.persona1_id} <-> {request.persona2_id}, 共 {len(memories)} 條記憶")
        
        return UnityMemoryContextResponse(
            success=True,
            context=context,
            memory_count=len(memories),
            memories=memories
        )
        
    except Exception as e:
        logger.error(f"獲取Unity記憶上下文失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"獲取記憶上下文失敗: {str(e)}")

@app.delete("/unity-api/memory/clear/{persona1_id}/{persona2_id}")
async def clear_unity_memory(persona1_id: str, persona2_id: str):
    """
    清除Unity角色對的記憶
    
    ## 功能描述
    清除兩個特定角色之間的所有對話記憶。
    
    ## 輸入參數
    - `persona1_id` (str): 第一個角色ID
    - `persona2_id` (str): 第二個角色ID
    
    ## 輸出結果
    - `success` (bool): 是否成功清除記憶
    - `message` (str): 操作結果訊息
    """
    try:
        pair_key = create_unity_memory_pair_key(persona1_id, persona2_id)
        memory_file = os.path.join(UNITY_MEMORY_PATH, f"{pair_key}.json")
        
        if os.path.exists(memory_file):
            os.remove(memory_file)
            message = f"已清除 {persona1_id} 和 {persona2_id} 之間的所有記憶"
        else:
            message = f"{persona1_id} 和 {persona2_id} 之間沒有記憶記錄"
        
        logger.info(f"Unity記憶清除: {message}")
        
        return {
            "success": True,
            "message": message
        }
        
    except Exception as e:
        logger.error(f"清除Unity記憶失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"清除記憶失敗: {str(e)}")

@app.delete("/unity-api/memory/clear-all")
async def clear_all_unity_memories():
    """
    清除所有Unity記憶
    
    ## 功能描述
    清除所有角色對的記憶記錄。
    
    ## 輸出結果
    - `success` (bool): 是否成功清除所有記憶
    - `message` (str): 操作結果訊息
    - `cleared_count` (int): 清除的記憶文件數量
    """
    try:
        import glob
        memory_files = glob.glob(os.path.join(UNITY_MEMORY_PATH, "*.json"))
        cleared_count = 0
        
        for file_path in memory_files:
            try:
                os.remove(file_path)
                cleared_count += 1
            except Exception as e:
                logger.warning(f"無法刪除記憶文件 {file_path}: {e}")
        
        message = f"已清除 {cleared_count} 個記憶文件"
        logger.info(f"Unity記憶全部清除: {message}")
        
        return {
            "success": True,
            "message": message,
            "cleared_count": cleared_count
        }
        
    except Exception as e:
        logger.error(f"清除所有Unity記憶失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"清除所有記憶失敗: {str(e)}")

@app.get("/unity-api/memory/stats")
async def get_unity_memory_stats():
    """
    獲取Unity記憶統計資訊
    
    ## 功能描述
    獲取Unity記憶系統的統計資訊，包括角色對數量和記憶總數。
    
    ## 輸出結果
    - `success` (bool): 是否成功獲取統計資訊
    - `total_pairs` (int): 角色對總數
    - `total_memories` (int): 記憶條目總數
    - `pairs_info` (List): 每個角色對的詳細資訊
    """
    try:
        import glob
        memory_files = glob.glob(os.path.join(UNITY_MEMORY_PATH, "*.json"))
        
        total_pairs = len(memory_files)
        total_memories = 0
        pairs_info = []
        
        for file_path in memory_files:
            try:
                pair_key = os.path.basename(file_path).replace('.json', '')
                async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                    content = await f.read()
                    data = json.loads(content)
                    memory_count = len(data)
                    total_memories += memory_count
                    
                    # 獲取角色名稱（從第一條記憶中）
                    if data:
                        first_memory = data[0]
                        pairs_info.append({
                            "pair_key": pair_key,
                            "persona1_name": first_memory.get("persona1_name", "未知"),
                            "persona2_name": first_memory.get("persona2_name", "未知"),
                            "memory_count": memory_count,
                            "last_interaction": data[-1].get("timestamp", "未知") if data else "無"
                        })
            except Exception as e:
                logger.warning(f"處理記憶文件 {file_path} 時發生錯誤: {e}")
        
        return {
            "success": True,
            "total_pairs": total_pairs,
            "total_memories": total_memories,
            "pairs_info": pairs_info
        }
        
    except Exception as e:
        logger.error(f"獲取Unity記憶統計失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"獲取記憶統計失敗: {str(e)}")

# =======================================================
# 火災模擬 API 端點
# =======================================================

# 火災模擬房間名稱與前端地圖代碼對應
# 布局：上排(A1-A4): 臥室A、臥室B、臥室C、浴室
#       中間(B1-B4): 走廊（貫穿）
#       下排(C1-C4): 客廳、廚房、大門(出口)、空
FIRE_SIMULATION_ROOM_ID_MAP: Dict[str, str] = {
    "臥室A": "A1",
    "臥室B": "A2",
    "臥室C": "A3",
    "浴室": "A4",
    "走廊": "B2",  # 下排第一個位置，跨2列
    "客廳": "C1",
    "廚房": "C2",
    "大門": "C3"
}
def _room_name_to_id(room_name: str) -> str:
    return FIRE_SIMULATION_ROOM_ID_MAP.get(room_name, room_name)


def _normalize_agent_for_stream(agent: Dict[str, Any]) -> Dict[str, Any]:
    agent_payload = agent.copy()
    original_room = agent.get("current_room") or agent.get("location")
    room_id = _room_name_to_id(original_room)
    agent_payload["current_room"] = room_id
    agent_payload["location_id"] = room_id
    agent_payload["current_room_name"] = original_room
    # 保留語言行為與簡化行為（若存在）
    if "語言行為" in agent:
        agent_payload["語言行為"] = agent["語言行為"]
    if "簡化行為" in agent:
        agent_payload["簡化行為"] = agent["簡化行為"]
    # 英文別名，方便未來前端使用
    if "language_behavior" in agent:
        agent_payload["language_behavior"] = agent["language_behavior"]
    if "simplified_behavior" in agent:
        agent_payload["simplified_behavior"] = agent["simplified_behavior"]
    return agent_payload


def _normalize_rooms_for_stream(rooms: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    normalized: Dict[str, Dict[str, Any]] = {}
    for name, data in rooms.items():
        room_id = _room_name_to_id(name)
        room_payload = data.copy()
        room_payload["id"] = room_id
        room_payload["display_name"] = name
        room_payload["fire_level"] = 1 if room_payload.get("on_fire") else 0
        smoke_value = room_payload.get("smoke")
        if isinstance(smoke_value, (int, float)):
            room_payload["smoke_level"] = smoke_value
        else:
            room_payload["smoke_level"] = 1 if smoke_value else 0
        normalized[room_id] = room_payload
    return normalized


class CustomAgentConfig(BaseModel):
    persona_id: str
    name: str
    initial_room: str = "客廳"

class FireSimulationRequest(BaseModel):
    num_agents: int = 5
    max_rounds: int = 15
    simulation_speed: Optional[float] = 1.0
    fire_start_room: Optional[str] = "廚房"  # 起火房間名稱
    custom_agents: Optional[List[CustomAgentConfig]] = None  # 自訂代理人配置

class FireSimulationResponse(BaseModel):
    simulation_id: str
    status: str
    message: str

@app.post("/start-fire-simulation", response_model=FireSimulationResponse)
async def start_fire_simulation(request: FireSimulationRequest):
    """啟動火災模擬"""
    try:
        # 創建新的模擬器
        simulator = FireScenarioSimulator()
        await simulator.initialize()
        
        # 生成模擬 ID
        simulation_id = f"fire_sim_{str(uuid.uuid4())[:8]}"
        
        # 儲存到活躍模擬中
        active_simulations[simulation_id] = {
            "simulator": simulator,
            "status": "running",
            "tick": 0,
            "agents_data": [],
            "rooms_data": {},
            "events": [],
            "finished": False
        }
        
        # 設定起火點
        simulator.fire_start_room = request.fire_start_room or "廚房"
        
        # 準備自訂代理人配置（如果有的話）
        custom_agents_config = None
        if request.custom_agents:
            custom_agents_config = [
                {"persona_id": agent.persona_id, "initial_room": agent.initial_room}
                for agent in request.custom_agents
            ]
        
        # 在背景執行模擬
        asyncio.create_task(run_fire_simulation_background(
            simulation_id, 
            simulator, 
            request.num_agents, 
            request.max_rounds,
            request.simulation_speed or 1.0,
            custom_agents_config
        ))
        
        return FireSimulationResponse(
            simulation_id=simulation_id,
            status="started",
            message=f"火災模擬已啟動，ID: {simulation_id}"
        )
        
    except Exception as e:
        logger.error(f"啟動火災模擬失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"啟動模擬失敗: {str(e)}")

async def run_fire_simulation_background(simulation_id: str, simulator: FireScenarioSimulator, 
                                       num_agents: int, max_rounds: int, speed: float,
                                       custom_agents_config: Optional[List[Dict]] = None):
    """在背景執行火災模擬"""
    try:
        sim_data = active_simulations[simulation_id]
        
        # 建立代理人（支援自訂配置）
        if custom_agents_config:
            await simulator.create_agents_from_config(custom_agents_config)
        else:
            await simulator.create_agents_from_database(num_agents)
        
        # 開始模擬循環
        for round_num in range(max_rounds):
            if sim_data["status"] != "running":
                break
                
            # 執行一個 tick
            tick_events = await simulator.fire_scenario_tick()
            
            # 更新模擬資料
            sim_data["tick"] = simulator.tick
            sim_data["events"] = tick_events
            sim_data["agents_data"] = [
                {
                    "id": agent.id,
                    "name": agent.name,
                    "current_room": agent.location,  # 前端期望的欄位名稱
                    "location": agent.location,  # 保留原有欄位
                    "health": agent.health,
                    "panic_level": agent.panic_level,
                    "escaped": agent.escaped,
                    "injured": agent.injured,
                    "status": "已逃生" if agent.escaped else ("受傷" if agent.injured else "危險中"),
                    "current_action": getattr(agent, 'current_action', '等待中...'),
                    "traits": agent.traits,
                    "social_abilities": getattr(agent, 'social_abilities', []),
                    "ability_attributes": getattr(agent, 'ability_attributes', []),
                    # 即時監控：內心獨白與記憶片段
                    "internal_monologue": getattr(agent, 'internal_monologue', ""),
                    "memory_snippet": getattr(agent, 'memory_snippet', ""),
                    # 新增語言行為與簡化行為（提供中英文鍵）
                    "語言行為": agent.persona_data.get("語言行為"),
                    "簡化行為": agent.persona_data.get("簡化行為"),
                    "language_behavior": agent.persona_data.get("語言行為"),
                    "simplified_behavior": agent.persona_data.get("簡化行為")
                }
                for agent in simulator.agents
            ]
            sim_data["rooms_data"] = {
                name: {
                    "name": name,
                    "on_fire": room.on_fire,
                    "smoke": room.smoke,
                    "temperature": room.temperature,
                    "visibility": room.visibility,
                    "is_exit": room.is_exit,
                    "blocked": room.blocked
                }
                for name, room in simulator.rooms.items()
            }
            
            # 檢查是否結束
            active_agents = [a for a in simulator.agents if not a.escaped and not a.injured]
            if not active_agents:
                sim_data["finished"] = True
                sim_data["status"] = "completed"
                sim_data["events"] = ["所有代理人都已逃生或受傷，模擬結束"]
                break
                
            # 控制模擬速度
            await asyncio.sleep(speed)
            
        # 模擬結束
        if sim_data["status"] == "running":
            sim_data["status"] = "completed"
            sim_data["finished"] = True
            
        # 計算最終統計
        total_agents = len(simulator.agents)
        escaped_count = sum(1 for a in simulator.agents if a.escaped)
        injured_count = sum(1 for a in simulator.agents if a.injured)
        
        sim_data["final_stats"] = {
            "total_agents": total_agents,
            "escaped_count": escaped_count,
            "injured_count": injured_count,
            "escape_rate": escaped_count / total_agents if total_agents > 0 else 0,
            "injury_rate": injured_count / total_agents if total_agents > 0 else 0,
            "total_ticks": simulator.tick
        }
        
        logger.info(f"火災模擬 {simulation_id} 完成")
        
    except Exception as e:
        logger.error(f"火災模擬 {simulation_id} 執行失敗: {e}", exc_info=True)
        sim_data["status"] = "error"
        sim_data["error"] = str(e)

@app.get("/simulation-stream/{simulation_id}")
async def get_simulation_stream(simulation_id: str):
    """提供模擬的實時串流資料"""
    from fastapi.responses import StreamingResponse
    
    if simulation_id not in active_simulations:
        raise HTTPException(status_code=404, detail="模擬不存在")
    
    async def event_generator():
        connection_id = str(uuid.uuid4())[:8]
        logger.info(f"新的SSE連接建立: {connection_id} for simulation {simulation_id}")
        
        try:
            # 發送連接確認
            connected_data = json.dumps({
                'message': '連接成功', 
                'simulation_id': simulation_id, 
                'connection_id': connection_id
            }, ensure_ascii=False)
            yield f"event: connected\ndata: {connected_data}\n\n"
            
            last_tick = -1
            last_events_count = 0
            heartbeat_counter = 0
            
            while simulation_id in active_simulations:
                try:
                    sim_data = active_simulations[simulation_id]
                    current_tick = sim_data.get("tick", 0)
                    current_events = sim_data.get("events", [])
                    current_status = sim_data.get("status", "unknown")
                    
                    # 檢查是否有更新需要發送
                    has_update = (
                        current_tick != last_tick or 
                        len(current_events) > last_events_count or
                        sim_data.get("finished", False)
                    )
                    
                    if has_update:
                        # 準備發送的資料
                        new_events = current_events[last_events_count:] if len(current_events) > last_events_count else []
                        
                        normalized_agents = [
                            _normalize_agent_for_stream(agent)
                            for agent in sim_data.get("agents_data", [])
                        ]
                        normalized_rooms = _normalize_rooms_for_stream(
                            sim_data.get("rooms_data", {})
                        )

                        update_data = {
                            "tick": current_tick,
                            "status": current_status,
                            "agents": normalized_agents,
                            "rooms": normalized_rooms,
                            "events": new_events,
                            "finished": sim_data.get("finished", False)
                        }
                        
                        if sim_data.get("finished", False):
                            update_data["stats"] = sim_data.get("final_stats", {})
                        
                        # 發送更新
                        update_json = json.dumps(update_data, ensure_ascii=False)
                        yield f"event: simulation_update\ndata: {update_json}\n\n"
                        
                        logger.debug(f"SSE {connection_id} 發送更新: tick={current_tick}, events={len(new_events)}")
                        
                        last_tick = current_tick
                        last_events_count = len(current_events)
                        
                        # 如果模擬結束
                        if sim_data.get("finished", False):
                            end_data = json.dumps({'message': '模擬結束'}, ensure_ascii=False)
                            yield f"event: simulation_ended\ndata: {end_data}\n\n"
                            logger.info(f"SSE連接 {connection_id} 模擬結束，正常關閉")
                            break
                    
                    # 每10次循環發送一次心跳
                    heartbeat_counter += 1
                    if heartbeat_counter % 10 == 0:
                        yield f": heartbeat-{connection_id}-{heartbeat_counter}\n\n"
                    
                except KeyError as ke:
                    logger.warning(f"SSE {connection_id} 缺少資料鍵: {ke}")
                except Exception as inner_e:
                    logger.error(f"SSE內部循環錯誤 {connection_id}: {inner_e}")
                
                # 等待
                await asyncio.sleep(0.5)
                
        except asyncio.CancelledError:
            logger.info(f"SSE連接 {connection_id} 被客戶端取消")
            return
        except Exception as e:
            logger.error(f"SSE嚴重錯誤 {connection_id}: {e}", exc_info=True)
            try:
                error_data = json.dumps({'error': str(e), 'connection_id': connection_id}, ensure_ascii=False)
                yield f"event: error\ndata: {error_data}\n\n"
            except:
                pass
        finally:
            logger.info(f"SSE連接 {connection_id} 關閉")
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Methods": "*",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/fire-simulation/{simulation_id}/status")
async def get_simulation_status(simulation_id: str):
    """獲取模擬狀態"""
    if simulation_id not in active_simulations:
        raise HTTPException(status_code=404, detail="模擬不存在")
    
    sim_data = active_simulations[simulation_id]
    
    response = {
        "simulation_id": simulation_id,
        "status": sim_data["status"],
        "tick": sim_data["tick"],
        "finished": sim_data.get("finished", False),
        "agents_data": sim_data.get("agents_data", []),
        "rooms_data": sim_data.get("rooms_data", {}),
        "events": sim_data.get("events", [])
    }
    
    if sim_data.get("final_stats"):
        response["final_stats"] = sim_data["final_stats"]
    
    if sim_data.get("error"):
        response["error"] = sim_data["error"]
    
    return response

@app.post("/fire-simulation/{simulation_id}/control")
async def control_simulation(simulation_id: str, action: str = Query(...)):
    """控制模擬（暫停、繼續、停止）"""
    if simulation_id not in active_simulations:
        raise HTTPException(status_code=404, detail="模擬不存在")
    
    sim_data = active_simulations[simulation_id]
    
    if action == "pause":
        sim_data["status"] = "paused"
    elif action == "resume":
        sim_data["status"] = "running"
    elif action == "stop":
        sim_data["status"] = "stopped"
    else:
        raise HTTPException(status_code=400, detail="無效的控制動作")
    
    return {"message": f"模擬 {action} 成功"}

@app.delete("/fire-simulation/{simulation_id}")
async def delete_simulation(simulation_id: str):
    """刪除模擬"""
    if simulation_id not in active_simulations:
        raise HTTPException(status_code=404, detail="模擬不存在")
    
    del active_simulations[simulation_id]
    return {"message": "模擬已刪除"}

@app.get("/fire-simulations")
async def list_active_simulations():
    """列出所有活躍的模擬"""
    simulations = []
    for sim_id, sim_data in active_simulations.items():
        simulations.append({
            "simulation_id": sim_id,
            "status": sim_data["status"],
            "tick": sim_data["tick"],
            "finished": sim_data.get("finished", False),
            "num_agents": len(sim_data["agents_data"])
        })
    
    return {"simulations": simulations}

# =======================================================

# 新增模組化模擬相關的數據模型
class ScenarioInfo(BaseModel):
    id: str
    name: str
    description: str
    scenario_type: str
    complexity: str
    features: List[str]
    parameters: Dict[str, Any]

class MapConfiguration(BaseModel):
    rooms: Dict[str, Dict[str, Any]]
    connections: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = {}

class AgentConfiguration(BaseModel):
    count: int
    types: Dict[str, int]  # 類型分布
    spawn_location: str
    custom_traits: Optional[List[str]] = []

class SimulationParameters(BaseModel):
    max_rounds: int = 50
    speed: float = 1.0
    random_seed: Optional[int] = None
    initial_temperature: float = 25.0
    initial_visibility: float = 100.0
    panic_spread_rate: float = 0.3
    speed_variation: float = 0.2
    scenario_specific: Optional[Dict[str, Any]] = {}

class ModularSimulationRequest(BaseModel):
    scenario_id: str
    map_config: MapConfiguration
    agent_config: AgentConfiguration
    parameters: SimulationParameters
    session_id: Optional[str] = None

class ModularSimulationResponse(BaseModel):
    simulation_id: str
    status: str
    message: str
    session_id: str

# 在現有的 app 定義後添加新的路由

# =======================================================
# 模組化模擬框架 API 端點
# =======================================================

@app.get("/simulation-frontend")  
def serve_simulation_frontend():
    """提供模組化模擬前端界面"""
    return FileResponse("simulation_frontend.html", media_type="text/html")

@app.get("/api/scenarios", response_model=List[ScenarioInfo])
async def get_available_scenarios():
    """獲取可用的模擬情境列表"""
    try:
        scenarios = [
            ScenarioInfo(
                id="fire",
                name="火災情境",
                description="建築物內火災蔓延與人員疏散模擬",
                scenario_type="emergency",
                complexity="中等",
                features=["火勢蔓延", "煙霧擴散", "結構損壞", "緊急疏散"],
                parameters={
                    "fire_spread_rate": {"type": "float", "default": 0.1, "min": 0.0, "max": 1.0},
                    "smoke_density": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
                    "structural_damage": {"type": "bool", "default": True},
                    "emergency_exits": {"type": "int", "default": 2, "min": 1, "max": 10}
                }
            ),
            ScenarioInfo(
                id="earthquake",
                name="地震情境",
                description="地震發生時的建築物搖晃與人員避難",
                scenario_type="natural_disaster",
                complexity="高",
                features=["地震波動", "建築搖晃", "物品掉落", "避難行為"],
                parameters={
                    "magnitude": {"type": "float", "default": 6.0, "min": 1.0, "max": 9.0},
                    "duration": {"type": "int", "default": 60, "min": 10, "max": 300},
                    "aftershocks": {"type": "bool", "default": True},
                    "building_stability": {"type": "float", "default": 0.8, "min": 0.1, "max": 1.0}
                }
            ),
            ScenarioInfo(
                id="flood",
                name="水災情境",
                description="洪水淹沒區域的人員撤離模擬",
                scenario_type="natural_disaster",
                complexity="中等",
                features=["水位上升", "區域淹沒", "交通阻斷", "高地避難"],
                parameters={
                    "water_rise_rate": {"type": "float", "default": 0.05, "min": 0.01, "max": 0.5},
                    "max_water_level": {"type": "float", "default": 2.0, "min": 0.5, "max": 10.0},
                    "evacuation_warning": {"type": "int", "default": 30, "min": 0, "max": 120}
                }
            ),
            ScenarioInfo(
                id="power_outage",
                name="停電情境",
                description="大規模停電時的人員行為與應對",
                scenario_type="infrastructure",
                complexity="低",
                features=["照明中斷", "電梯停擺", "通訊受限", "緊急照明"],
                parameters={
                    "emergency_lighting": {"type": "bool", "default": True},
                    "backup_power_duration": {"type": "int", "default": 120, "min": 0, "max": 600},
                    "communication_loss": {"type": "float", "default": 0.8, "min": 0.0, "max": 1.0}
                }
            ),
            ScenarioInfo(
                id="terror_attack",
                name="恐攻情境",
                description="恐怖攻擊事件的人員疏散與應對",
                scenario_type="security",
                complexity="高",
                features=["威脅來源", "恐慌傳播", "封鎖區域", "安全疏散"],
                parameters={
                    "threat_locations": {"type": "int", "default": 1, "min": 1, "max": 5},
                    "lockdown_zones": {"type": "bool", "default": True},
                    "panic_amplification": {"type": "float", "default": 1.5, "min": 1.0, "max": 3.0},
                    "security_response_time": {"type": "int", "default": 300, "min": 60, "max": 1800}
                }
            ),
            ScenarioInfo(
                id="pandemic",
                name="疫情情境",
                description="傳染病擴散的人員流動控制模擬",
                scenario_type="health",
                complexity="高",
                features=["病毒傳播", "隔離措施", "社交距離", "檢疫流程"],
                parameters={
                    "transmission_rate": {"type": "float", "default": 0.15, "min": 0.01, "max": 1.0},
                    "incubation_period": {"type": "int", "default": 14, "min": 1, "max": 30},
                    "social_distancing": {"type": "float", "default": 2.0, "min": 1.0, "max": 5.0},
                    "quarantine_effectiveness": {"type": "float", "default": 0.9, "min": 0.1, "max": 1.0}
                }
            )
        ]
        
        logger.info(f"返回 {len(scenarios)} 個可用情境")
        return scenarios
        
    except Exception as e:
        logger.error(f"獲取情境列表失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"獲取情境列表失敗: {str(e)}")

@app.get("/api/scenarios/{scenario_id}", response_model=ScenarioInfo)
async def get_scenario_details(scenario_id: str):
    """獲取特定情境的詳細資訊"""
    try:
        scenarios = await get_available_scenarios()
        scenario = next((s for s in scenarios if s.id == scenario_id), None)
        
        if not scenario:
            raise HTTPException(status_code=404, detail=f"情境 {scenario_id} 不存在")
        
        return scenario
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"獲取情境詳情失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"獲取情境詳情失敗: {str(e)}")

@app.post("/api/map/validate")
async def validate_map_configuration(map_config: MapConfiguration):
    """驗證地圖配置的有效性"""
    try:
        rooms = map_config.rooms
        connections = map_config.connections
        
        # 基本驗證
        errors = []
        warnings = []
        
        # 檢查是否有房間
        if not rooms:
            errors.append("地圖必須至少包含一個房間")
        
        # 檢查是否有出口
        exits = [room for room in rooms.values() if room.get("is_exit", False)]
        if not exits:
            warnings.append("建議至少設置一個出口房間")
        
        # 檢查連接完整性
        room_ids = set(rooms.keys())
        for conn in connections:
            if conn.get("from") not in room_ids:
                errors.append(f"連接中的房間 {conn.get('from')} 不存在")
            if conn.get("to") not in room_ids:
                errors.append(f"連接中的房間 {conn.get('to')} 不存在")
        
        # 檢查連通性（簡化版）
        if len(rooms) > 1 and not connections:
            warnings.append("多個房間之間應該有連接路徑")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "room_count": len(rooms),
            "connection_count": len(connections),
            "exit_count": len(exits)
        }
        
    except Exception as e:
        logger.error(f"地圖驗證失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"地圖驗證失敗: {str(e)}")

@app.post("/api/agents/generate")
async def generate_simulation_agents(agent_config: AgentConfiguration):
    """根據配置生成模擬代理人"""
    try:
        # 初始化角色生成器
        if not hasattr(app.state, 'persona_loader'):
            from module.PERSONA_loader import PersonaLoader
            app.state.persona_loader = PersonaLoader()
        
        # 獲取現有的 persona 列表
        existing_personas = list(app.state.persona_loader.personas.keys())
        
        if len(existing_personas) < agent_config.count:
            # 如果現有角色不足，生成新的
            needed = agent_config.count - len(existing_personas)
            logger.info(f"需要生成 {needed} 個新角色")
            
            for _ in range(needed):
                try:
                    new_id = await generate_a_persona()
                    await generate_persona_rag_by_id(new_id)
                    existing_personas.append(new_id)
                except Exception as e:
                    logger.warning(f"生成角色失敗: {e}")
        
        # 從現有角色中選擇
        import random
        selected_personas = random.sample(existing_personas, min(agent_config.count, len(existing_personas)))
        
        # 為每個選中的 persona 生成代理人資料
        agents = []
        for i, persona_id in enumerate(selected_personas):
            try:
                persona_data = app.state.persona_loader.personas[persona_id]
                agent = {
                    "id": f"agent_{i}",
                    "persona_id": persona_id,
                    "name": persona_data['基本資料']['姓名'],
                    "age": persona_data['基本資料']['年紀'],
                    "gender": persona_data['基本資料']['性別'],
                    "traits": persona_data['人格屬性']['人格特質'],
                    "social_abilities": persona_data['人格屬性']['社交能力'],
                    "ability_attributes": persona_data['人格屬性']['能力屬性'],
                    "spawn_location": agent_config.spawn_location,
                    "status": "active",
                    "health": 100,
                    "panic_level": 0
                }
                agents.append(agent)
            except Exception as e:
                logger.warning(f"處理 persona {persona_id} 失敗: {e}")
        
        return {
            "success": True,
            "agent_count": len(agents),
            "agents": agents,
            "message": f"成功生成 {len(agents)} 個代理人"
        }
        
    except Exception as e:
        logger.error(f"生成代理人失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"生成代理人失敗: {str(e)}")

@app.post("/api/simulation/start", response_model=ModularSimulationResponse)
async def start_modular_simulation(request: ModularSimulationRequest):
    """啟動模組化模擬"""
    try:
        # 驗證情境
        scenario = await get_scenario_details(request.scenario_id)
        
        # 驗證地圖
        map_validation = await validate_map_configuration(request.map_config)
        if not map_validation["valid"]:
            raise HTTPException(status_code=400, detail=f"地圖配置無效: {', '.join(map_validation['errors'])}")
        
        # 生成模擬 ID
        simulation_id = f"sim_{request.scenario_id}_{str(uuid.uuid4())[:8]}"
        session_id = request.session_id or f"session_{str(uuid.uuid4())[:8]}"
        
        # 根據情境類型創建對應的模擬器
        if request.scenario_id == "fire":
            # 使用現有的火災模擬器
            simulator = FireScenarioSimulator()
            await simulator.initialize()
            
            # 注意：當前的FireScenarioSimulator使用預設的房間配置
            # 未來版本將支援自訂地圖配置
            
            # 創建代理人
            await simulator.create_agents_from_database(request.agent_config.count)
        
        else:
            # 其他情境類型的模擬器（待實作）
            raise HTTPException(status_code=501, detail=f"情境 {request.scenario_id} 的模擬器尚未實作")
        
        # 儲存模擬配置
        simulation_data = {
            "simulator": simulator,
            "config": request.dict(),
            "status": "running",
            "tick": 0,
            "start_time": asyncio.get_event_loop().time(),
            "agents_data": [],
            "environment_data": {},
            "events": [],
            "finished": False
        }
        
        active_simulations[simulation_id] = simulation_data
        
        # 在背景執行模擬
        asyncio.create_task(run_modular_simulation_background(
            simulation_id,
            request.parameters.max_rounds,
            request.parameters.speed
        ))
        
        logger.info(f"啟動模組化模擬: {simulation_id}")
        
        return ModularSimulationResponse(
            simulation_id=simulation_id,
            status="started",
            message=f"模擬已啟動 - 情境: {scenario.name}",
            session_id=session_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"啟動模組化模擬失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"啟動模擬失敗: {str(e)}")

async def run_modular_simulation_background(simulation_id: str, max_rounds: int, speed: float):
    """在背景執行模組化模擬"""
    try:
        sim_data = active_simulations[simulation_id]
        simulator = sim_data["simulator"]
        
        logger.info(f"開始執行模組化模擬 {simulation_id}")
        
        for round_num in range(max_rounds):
            if sim_data["status"] != "running":
                break
            
            # 執行一個模擬步驟
            if hasattr(simulator, 'fire_scenario_tick'):
                # 火災情境
                tick_events = await simulator.fire_scenario_tick()
            else:
                # 其他情境的模擬邏輯
                tick_events = []
            
            # 更新模擬資料
            sim_data["tick"] = simulator.tick if hasattr(simulator, 'tick') else round_num
            sim_data["events"] = tick_events
            
            # 更新代理人資料
            if hasattr(simulator, 'agents'):
                sim_data["agents_data"] = [
                    {
                        "id": agent.id,
                        "name": agent.name,
                        "location": agent.location,
                        "health": agent.health,
                        "panic_level": getattr(agent, 'panic_level', 0),
                        "escaped": getattr(agent, 'escaped', False),
                        "injured": getattr(agent, 'injured', False),
                        "status": get_agent_status(agent),
                        "current_action": getattr(agent, 'current_action', '等待中...'),
                    }
                    for agent in simulator.agents
                ]
            
            # 更新環境資料
            if hasattr(simulator, 'rooms'):
                sim_data["environment_data"] = {
                    "rooms": {
                        name: {
                            "name": name,
                            "on_fire": getattr(room, 'on_fire', False),
                            "smoke": getattr(room, 'smoke', 0),
                            "temperature": getattr(room, 'temperature', 25),
                            "visibility": getattr(room, 'visibility', 100),
                            "is_exit": getattr(room, 'is_exit', False),
                            "blocked": getattr(room, 'blocked', False)
                        }
                        for name, room in simulator.rooms.items()
                    }
                }
            
            # 檢查結束條件
            if hasattr(simulator, 'agents'):
                active_agents = [a for a in simulator.agents 
                               if not getattr(a, 'escaped', False) and not getattr(a, 'injured', False)]
                if not active_agents:
                    sim_data["finished"] = True
                    sim_data["status"] = "completed"
                    break
            
            # 控制模擬速度
            await asyncio.sleep(speed)
        
        # 模擬結束處理
        if sim_data["status"] == "running":
            sim_data["status"] = "completed"
            sim_data["finished"] = True
        
        # 計算最終統計
        if hasattr(simulator, 'agents'):
            total_agents = len(simulator.agents)
            escaped_count = sum(1 for a in simulator.agents if getattr(a, 'escaped', False))
            injured_count = sum(1 for a in simulator.agents if getattr(a, 'injured', False))
            
            sim_data["final_stats"] = {
                "total_agents": total_agents,
                "escaped_count": escaped_count,
                "injured_count": injured_count,
                "escape_rate": escaped_count / total_agents if total_agents > 0 else 0,
                "injury_rate": injured_count / total_agents if total_agents > 0 else 0,
                "total_rounds": sim_data["tick"],
                "simulation_time": asyncio.get_event_loop().time() - sim_data["start_time"]
            }
        
        logger.info(f"模組化模擬 {simulation_id} 完成")
        
    except Exception as e:
        logger.error(f"模組化模擬 {simulation_id} 執行失敗: {e}", exc_info=True)
        sim_data["status"] = "error"
        sim_data["error"] = str(e)

def get_agent_status(agent):
    """獲取代理人狀態描述"""
    if getattr(agent, 'escaped', False):
        return "已逃生"
    elif getattr(agent, 'injured', False):
        return "受傷"
    elif getattr(agent, 'panic_level', 0) > 0.7:
        return "恐慌中"
    elif getattr(agent, 'health', 100) < 50:
        return "健康不佳"
    else:
        return "正常"

@app.get("/api/simulation/{simulation_id}/status")
async def get_modular_simulation_status(simulation_id: str):
    """獲取模組化模擬狀態"""
    if simulation_id not in active_simulations:
        raise HTTPException(status_code=404, detail="模擬不存在")
    
    sim_data = active_simulations[simulation_id]
    
    response = {
        "simulation_id": simulation_id,
        "status": sim_data["status"],
        "tick": sim_data["tick"],
        "finished": sim_data.get("finished", False),
        "agents": sim_data.get("agents_data", []),
        "environment": sim_data.get("environment_data", {}),
        "events": sim_data.get("events", []),
        "config": sim_data.get("config", {})
    }
    
    if sim_data.get("final_stats"):
        response["final_stats"] = sim_data["final_stats"]
    
    if sim_data.get("error"):
        response["error"] = sim_data["error"]
    
    return response

@app.get("/api/simulation/{simulation_id}/stream")
async def get_modular_simulation_stream(simulation_id: str):
    """提供模組化模擬的實時串流資料"""
    if simulation_id not in active_simulations:
        raise HTTPException(status_code=404, detail="模擬不存在")
    
    async def event_generator():
        connection_id = str(uuid.uuid4())[:8]
        logger.info(f"新的模組化模擬SSE連接: {connection_id} for {simulation_id}")
        
        try:
            # 發送連接確認
            yield f"event: connected\ndata: {json.dumps({'simulation_id': simulation_id, 'connection_id': connection_id}, ensure_ascii=False)}\n\n"
            
            last_tick = -1
            
            while simulation_id in active_simulations:
                sim_data = active_simulations[simulation_id]
                current_tick = sim_data.get("tick", 0)
                
                if current_tick != last_tick or sim_data.get("finished", False):
                    # 準備更新資料
                    update_data = {
                        "tick": current_tick,
                        "status": sim_data.get("status", "unknown"),
                        "agents": sim_data.get("agents_data", []),
                        "environment": sim_data.get("environment_data", {}),
                        "events": sim_data.get("events", []),
                        "finished": sim_data.get("finished", False)
                    }
                    
                    if sim_data.get("finished", False):
                        update_data["final_stats"] = sim_data.get("final_stats", {})
                    
                    # 發送更新
                    yield f"event: simulation_update\ndata: {json.dumps(update_data, ensure_ascii=False)}\n\n"
                    
                    last_tick = current_tick
                    
                    if sim_data.get("finished", False):
                        yield f"event: simulation_ended\ndata: {json.dumps({'message': '模擬結束'}, ensure_ascii=False)}\n\n"
                        break
                
                await asyncio.sleep(0.5)
                
        except Exception as e:
            logger.error(f"模組化模擬SSE錯誤 {connection_id}: {e}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
        finally:
            logger.info(f"模組化模擬SSE連接 {connection_id} 關閉")
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*"
        }
    )

@app.post("/api/simulation/{simulation_id}/control")
async def control_modular_simulation(simulation_id: str, action: str = Query(...)):
    """控制模組化模擬（暫停、繼續、停止）"""
    if simulation_id not in active_simulations:
        raise HTTPException(status_code=404, detail="模擬不存在")
    
    sim_data = active_simulations[simulation_id]
    
    if action == "pause":
        sim_data["status"] = "paused"
    elif action == "resume":
        sim_data["status"] = "running"
    elif action == "stop":
        sim_data["status"] = "stopped"
    else:
        raise HTTPException(status_code=400, detail="無效的控制動作")
    
    logger.info(f"模擬 {simulation_id} 控制動作: {action}")
    return {"message": f"模擬 {action} 成功", "status": sim_data["status"]}

@app.delete("/api/simulation/{simulation_id}")
async def delete_modular_simulation(simulation_id: str):
    """刪除模組化模擬"""
    if simulation_id not in active_simulations:
        raise HTTPException(status_code=404, detail="模擬不存在")
    
    del active_simulations[simulation_id]
    logger.info(f"刪除模組化模擬: {simulation_id}")
    return {"message": "模擬已刪除"}

@app.get("/api/simulations")
async def list_all_simulations():
    """列出所有活躍的模擬"""
    simulations = []
    for sim_id, sim_data in active_simulations.items():
        config = sim_data.get("config", {})
        simulations.append({
            "simulation_id": sim_id,
            "scenario_id": config.get("scenario_id", "unknown"),
            "status": sim_data["status"],
            "tick": sim_data["tick"],
            "finished": sim_data.get("finished", False),
            "agent_count": len(sim_data.get("agents_data", [])),
            "start_time": sim_data.get("start_time", 0)
        })
    
    return {"simulations": simulations}

# =======================================================

# Start the server
if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=20000, reload=True)