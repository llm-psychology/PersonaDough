import random
import json
import asyncio
import logging
import os
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import aiofiles

# 整合現有模組
from module.PERSONA_loader import PersonaLoader
from interviewer.chat_with_persona import ChatWith
from interviewer.interviewer_generator import Interviewer

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =======================================================
# 環境定義
# =======================================================
class Room:
    def __init__(self, name, is_exit=False):
        self.name = name
        self.is_exit = is_exit
        self.on_fire = False
        self.smoke = False
        self.blocked = False
        self.temperature = 25  # 新增：溫度
        self.visibility = 100  # 新增：能見度 (0-100)
        self.agents = []  # 代理人ID清單

    def update_environment(self, tick):
        """根據時間更新環境狀態"""
        if self.on_fire:
            self.temperature += random.randint(10, 20)
            self.visibility = max(0, self.visibility - random.randint(10, 30))
        if self.smoke:
            self.visibility = max(0, self.visibility - random.randint(5, 15))

    def __repr__(self):
        status = []
        if self.is_exit: status.append('出口')
        if self.on_fire: status.append(f'著火({self.temperature}°C)')
        if self.smoke: status.append(f'煙霧(能見度{self.visibility}%)')
        if self.blocked: status.append('封鎖')
        return f"{self.name} ({'/'.join(status) if status else '正常'})"

# =======================================================
# 代理人定義
# =======================================================
class Agent:
    def __init__(self, persona_id, persona_data, initial_location):
        self.id = persona_id
        self.persona_data = persona_data
        self.name = persona_data['基本資料']['姓名']
        self.age = persona_data['基本資料']['年紀']
        self.traits = persona_data['人格屬性']['人格特質']
        self.social_abilities = persona_data['人格屬性']['社交能力']
        self.ability_attributes = persona_data['人格屬性']['能力屬性']
        
        # 狀態
        self.location = initial_location
        self.known_fire = False
        self.panic_level = 0  # 0-100
        self.health = 100  # 0-100
        self.escaped = False
        self.injured = False
        self.can_move = True
        self.current_action = "初始化中..."  # 當前行動描述
        # 可視化監控：內心獨白與記憶片段
        self.internal_monologue: str = ""
        self.memory_snippet: str = ""
        # 新增：允許逃生後重返救援
        self.allow_reentry: bool = True
        self.reentered_times: int = 0
        
        # 互動記錄
        self.log = []
        self.known_agents = set()  # 已知的其他代理人
        self.messages_received = []  # 收到的訊息
        self.last_action = {}  # 最後一次行動

    def update_status(self, room):
        """根據環境更新代理人狀態"""
        if room.on_fire:
            self.health -= random.randint(5, 15)
            self.panic_level = min(100, self.panic_level + random.randint(10, 20))
        if room.smoke:
            self.health -= random.randint(1, 5)
            self.panic_level = min(100, self.panic_level + random.randint(5, 10))
        
        if self.health <= 0:
            self.injured = True
            self.can_move = False

    def add_log(self, tick, action, details=""):
        """添加行為記錄"""
        log_entry = {
            "tick": tick,
            "action": action,
            "details": details,
            "location": self.location,
            "panic_level": self.panic_level,
            "health": self.health
        }
        self.log.append(log_entry)
        logger.info(f"[{tick}] {self.name}: {action} {details}")

    def __repr__(self):
        status = []
        if self.escaped: status.append('已逃生')
        if self.injured: status.append('受傷')
        if self.panic_level > 70: status.append('恐慌')
        elif self.panic_level > 40: status.append('緊張')
        return f"{self.name}@{self.location} ({'/'.join(status) if status else '正常'})"

# =======================================================
# 火災模擬器主類
# =======================================================
class FireScenarioSimulator:
    def __init__(self):
        self.persona_loader = None
        self.chatbot = None
        self.interviewer = None
        self.rooms = {}
        self.connections = {}
        self.agents = []
        self.tick = 0
        self.simulation_log = []
        self.fire_start_room = "廚房"  # 預設起火房間
        
    async def initialize(self):
        """初始化所有模組"""
        logger.info("初始化人格載入器...")
        self.persona_loader = PersonaLoader()
        await self.persona_loader.wait_until_ready()
        
        logger.info("初始化 LLM 對話模組...")
        self.chatbot = ChatWith()
        await self.chatbot.wait_until_ready()
        
        logger.info("初始化面試官模組...")
        self.interviewer = Interviewer()
        
        logger.info("建立環境...")
        self._create_environment()
        
        logger.info("初始化完成!")

    def _create_environment(self):
        """建立火災現場環境"""
        self.rooms = {
            '客廳': Room('客廳'),
            '廚房': Room('廚房'),
            '臥室A': Room('臥室A'),
            '臥室B': Room('臥室B'),
            '臥室C': Room('臥室C'),
            '走廊': Room('走廊'),
            '浴室': Room('浴室'),
            '大門': Room('大門', is_exit=True)
        }
        
        # 房間連接圖（根據實際平面圖）
        # 上排房間（臥室A、B、C、浴室）都連接到走廊
        # 下排房間（客廳、廚房、大門）也連接到走廊
        # 客廳和廚房相鄰
        # 廚房和大門相鄰
        self.connections = {
            '臥室A': ['走廊'],
            '臥室B': ['走廊'],
            '臥室C': ['走廊'],
            '浴室': ['走廊'],
            '走廊': ['臥室A', '臥室B', '臥室C', '浴室', '客廳', '廚房', '大門'],
            '客廳': ['走廊', '廚房'],
            '廚房': ['走廊', '客廳', '大門'],
            '大門': ['走廊', '廚房']
        }

    async def create_agents_from_database(self, num_agents=5):
        """從人格資料庫隨機選取代理人"""
        all_personas = list(self.persona_loader.get_all_personas().values())
        if len(all_personas) < num_agents:
            logger.warning(f"資料庫只有 {len(all_personas)} 個人格，將使用全部")
            selected_personas = all_personas
        else:
            selected_personas = random.sample(all_personas, num_agents)
        
        # 隨機分配初始位置（不包括出口和走廊）
        possible_locations = ['客廳', '臥室A', '臥室B', '臥室C', '廚房', '浴室']
        
        self.agents = []
        for persona in selected_personas:
            initial_location = random.choice(possible_locations)
            agent = Agent(persona['基本資料']['id'], persona, initial_location)
            self.agents.append(agent)
            
            # 更新房間內代理人列表
            self.rooms[initial_location].agents.append(agent.id)
            
            logger.info(f"建立代理人: {agent.name} (位於 {initial_location})")
            
        return self.agents

    async def create_agents_from_config(self, agents_config: List[Dict]):
        """根據自訂配置建立代理人
        
        Args:
            agents_config: 代理人配置列表，每個配置包含:
                - persona_id: 人格ID
                - initial_room: 初始房間位置
        """
        all_personas = self.persona_loader.get_all_personas()
        
        # 有效的房間列表
        valid_rooms = list(self.rooms.keys())
        
        self.agents = []
        for config in agents_config:
            persona_id = config.get('persona_id')
            initial_room = config.get('initial_room', '客廳')
            
            # 驗證人格是否存在
            if persona_id not in all_personas:
                logger.warning(f"找不到人格 ID: {persona_id}，跳過")
                continue
            
            persona = all_personas[persona_id]
            
            # 驗證房間是否有效
            if initial_room not in valid_rooms:
                logger.warning(f"無效的房間: {initial_room}，使用預設房間 '客廳'")
                initial_room = '客廳'
            
            # 建立代理人
            agent = Agent(persona_id, persona, initial_room)
            self.agents.append(agent)
            
            # 更新房間內代理人列表
            self.rooms[initial_room].agents.append(agent.id)
            
            logger.info(f"建立代理人: {agent.name} (位於 {initial_room})")
        
        if not self.agents:
            logger.warning("沒有成功建立任何代理人，使用隨機選擇")
            await self.create_agents_from_database(5)
            
        return self.agents

    async def agent_decision_making(self, agent: Agent, situation_context: str) -> Dict:
        """使用 LLM 進行代理人決策"""
        try:
            # 建立決策 prompt
            current_room = self.rooms[agent.location]
            other_agents_here = [a for a in self.agents if a.location == agent.location and a.id != agent.id]
            available_exits = self.connections[agent.location]
            
            # 獲取 RAG 記憶
            try:
                index, docs, _ = await self.chatbot.interviewer_obj.load_rag_database(agent.id)
                retrieved_docs, _ = await self.chatbot.interviewer_obj.retrieve_similar_docs(
                    situation_context, index, docs, top_k=2
                )
            except:
                retrieved_docs = []
            
            # 內部工具：簡單字串裁切
            def _trim_text(text: str, max_len: int) -> str:
                if not text:
                    return ""
                text = str(text).replace('\n', ' ').strip()
                return (text[:max_len] + '…') if len(text) > max_len else text

            # 從檢索結果擷取記憶片段（10~20字，這裡選擇最長20字）
            memory_snippet = ""
            if retrieved_docs:
                try:
                    # 嘗試抓第一筆文件的主要內容
                    first_doc = retrieved_docs[0]
                    # 常見物件：有 page_content 屬性 或 字典 或 一般字串
                    if hasattr(first_doc, 'page_content'):
                        memory_snippet = _trim_text(getattr(first_doc, 'page_content', ''), 20)
                    elif isinstance(first_doc, dict):
                        memory_snippet = _trim_text(first_doc.get('page_content') or first_doc.get('content') or first_doc.get('text') or "", 20)
                    else:
                        memory_snippet = _trim_text(str(first_doc), 20)
                except Exception:
                    memory_snippet = _trim_text(str(retrieved_docs[0]), 20)

            context = f"""
當前情境: {situation_context}
你的位置: {agent.location} ({current_room})
可前往的房間: {', '.join(available_exits)}
同房間的人: {[a.name for a in other_agents_here]}
你的恐慌程度: {agent.panic_level}/100
你的健康狀況: {agent.health}/100
是否知道火災: {agent.known_fire}
最近收到的訊息: {agent.messages_received[-3:] if agent.messages_received else '無'}
            """

            from module.prompt_loader import get_system_prompt, get_user_prompt
            
            # 格式化 retrieved_docs 為字符串
            retrieved_docs_str = '無特別記憶'
            if retrieved_docs:
                docs_list = retrieved_docs[:2]
                if docs_list:
                    retrieved_docs_str = '\n'.join([str(doc) for doc in docs_list])
            
            # 格式化 recent_messages 為字符串
            recent_messages_str = '無'
            if agent.messages_received:
                recent_msgs = agent.messages_received[-3:]
                recent_messages_str = ', '.join([msg.get('message', str(msg)) if isinstance(msg, dict) else str(msg) for msg in recent_msgs])
            
            # 風險評估
            room_temp = getattr(current_room, 'temperature', 25)
            room_vis = getattr(current_room, 'visibility', 100)
            room_on_fire = getattr(current_room, 'on_fire', False)
            room_smoke = getattr(current_room, 'smoke', False)
            risk_score = 0
            if room_on_fire: risk_score += 2
            if room_temp > 150: risk_score += 2
            if room_vis < 20: risk_score += 2
            if agent.panic_level >= 80: risk_score += 1
            if agent.health <= 20: risk_score += 2
            risk_level = 'low'
            if risk_score >= 6:
                risk_level = 'critical'
            elif risk_score >= 3:
                risk_level = 'high'
            elif risk_score >= 1:
                risk_level = 'medium'

            system_content = get_system_prompt("agent_decision_making",
                                             agent_name=agent.name,
                                             traits=str(agent.traits) if isinstance(agent.traits, (list, dict)) else agent.traits,
                                             social_abilities=str(agent.social_abilities) if isinstance(agent.social_abilities, (list, dict)) else agent.social_abilities,
                                             ability_attributes=str(agent.ability_attributes) if isinstance(agent.ability_attributes, (list, dict)) else agent.ability_attributes,
                                             retrieved_docs=retrieved_docs_str)
            
            user_content = get_user_prompt("agent_decision_making",
                                         situation_context=situation_context,
                                         location=agent.location,
                                         current_room=str(current_room),
                                         available_exits=', '.join(available_exits),
                                         other_agents=', '.join([a.name for a in other_agents_here]) if other_agents_here else '無',
                                         panic_level=agent.panic_level,
                                         health=agent.health,
                                         known_fire=agent.known_fire,
                                         recent_messages=recent_messages_str,
                                         risk_level=risk_level,
                                         room_temperature=room_temp,
                                         visibility=room_vis,
                                         on_fire=room_on_fire,
                                         smoke=room_smoke)

            prompt = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            
            response = await self.chatbot.interviewer_obj.simulate_persona_answer(prompt)
            
            # 解析回應
            try:
                # 清理回應，提取JSON
                response = response.strip()
                if response.startswith('```'):
                    response = response.split('```')[1]
                if response.startswith('json'):
                    response = response[4:].strip()
                
                decision = json.loads(response)
                # 產生精簡 10 字內獨白（根據行為與環境）
                def _generate_short_monologue(a: Agent, d: Dict) -> str:
                    # 多樣化短語，依狀態／行為與隨機性
                    import random
                    if a.escaped:
                        # 已逃生仍可返回救援
                        return "已逃生"
                    if a.injured:
                        return "受傷" if a.health > 0 else "倒地"
                    action = d.get('action', '')
                    room_state = self.rooms.get(a.location, None)
                    local_hazard = False
                    if room_state:
                        # 只用本地可見危險（著火/煙霧/能見度低/溫度高）作為危險依據
                        local_hazard = (
                            getattr(room_state, 'on_fire', False) or
                            getattr(room_state, 'smoke', False) or
                            getattr(room_state, 'visibility', 100) < 20 or
                            getattr(room_state, 'temperature', 25) > 150
                        )
                    # 在危險時，加入隨機多樣化
                    if local_hazard:
                        if action == 'help_others':
                            return random.choice(["我要救人", "先救人"])
                        if action == 'shout':
                            return random.choice(["呼救", "求援"])
                        if action == 'search':
                            return random.choice(["找出口", "搜尋出口"])
                        if action == 'move':
                            return random.choice(["快逃", "逃離火場"])
                        if action == 'stay':
                            return random.choice(["危險", "等一下"])
                        # 未知行為 fallback
                        return random.choice(["危險", "快逃", "呼救"]) 
                    # 沒有本地危險，但如果僅僅是收到警告（known_fire）則保持提醒語氣
                    if getattr(a, 'known_fire', False):
                        if action == 'shout':
                            return "警告火災"
                        if action == 'move':
                            return "小心前進"
                        return random.choice(["注意", "小心"]) 
                    # 非危險情境
                    if action == 'search':
                        return "找出口"
                    if action == 'help_others':
                        return random.choice(["我要救人", "先救人"])
                    if action == 'shout':
                        return "呼救"
                    if action == 'move':
                        return "移動"
                    if action == 'stay':
                        return "目前安全"
                    return "等待"
                agent.internal_monologue = _trim_text(_generate_short_monologue(agent, decision), 10)
                # 保留原本記憶片段欄位，即使前端不顯示
                agent.memory_snippet = memory_snippet
                return decision
            except json.JSONDecodeError:
                # 如果解析失敗，返回預設行為
                logger.warning(f"{agent.name} 決策解析失敗，使用預設行為")
                # 簡化失敗提示
                agent.internal_monologue = _trim_text("等待", 10)
                agent.memory_snippet = memory_snippet
                return {
                    "action": "move" if agent.known_fire else "stay",
                    "target": random.choice(available_exits) if agent.known_fire else agent.location,
                    "message": "",
                    "reasoning": "解析失敗的預設行為"
                }
                
        except Exception as e:
            logger.error(f"代理人 {agent.name} 決策失敗: {e}")
            # 決策錯誤時也維持最新的記憶片段供可視化
            try:
                agent.memory_snippet = memory_snippet if 'memory_snippet' in locals() else ""
            except Exception:
                pass
            agent.internal_monologue = _trim_text("錯誤", 10)
            return {
                "action": "stay",
                "target": agent.location,
                "message": "",
                "reasoning": "決策系統錯誤"
            }

    async def execute_agent_action(self, agent: Agent, decision: Dict):
        """執行代理人行動"""
        action = decision.get("action", "stay")
        target = decision.get("target", "")
        message = decision.get("message", "")
        reasoning = decision.get("reasoning", "")
        # 在執行前套用安全校正
        decision = self.validate_and_adjust_decision(agent, decision)
        action = decision.get("action", action)
        target = decision.get("target", target)
        message = decision.get("message", message)
        reasoning = decision.get("reasoning", reasoning)
        
        if action == "move" and target in self.connections[agent.location]:
            # 移動
            old_location = agent.location
            self.rooms[old_location].agents.remove(agent.id)
            agent.location = target
            self.rooms[target].agents.append(agent.id)
            
            if target in ['大門']:
                agent.escaped = True
                agent.current_action = f"已逃生到{target}"
                agent.add_log(self.tick, "逃生成功", f"從 {old_location} 逃到 {target}")
            else:
                # 若原本已逃生，現在移動回室內，取消逃生狀態
                if agent.escaped and agent.allow_reentry:
                    agent.escaped = False
                    agent.reentered_times += 1
                    agent.current_action = f"返回{target}救援"
                    agent.add_log(self.tick, "返回救援", f"從 {old_location} 進入 {target}, 理由: {reasoning}")
                else:
                    agent.current_action = f"移動到{target}"
                    agent.add_log(self.tick, "移動", f"從 {old_location} 到 {target}, 理由: {reasoning}")
                
        elif action == "shout" and message:
            # 呼救或警告
            agent.current_action = f"呼救: {message}"
            agent.add_log(self.tick, "呼救", f"說: '{message}'")
            # 通知同房間的其他人
            same_room_agents = [a for a in self.agents if a.location == agent.location and a.id != agent.id]
            for other_agent in same_room_agents:
                other_agent.messages_received.append({
                    "from": agent.name,
                    "message": message,
                    "tick": self.tick
                })
                # 如果是火災警告，讓其他人知道火災
                if "火" in message or "危險" in message:
                    other_agent.known_fire = True
                    
        elif action == "help_others" and target:
            agent.current_action = f"幫助 {target}"
            agent.add_log(self.tick, "幫助他人", f"幫助 {target}, 理由: {reasoning}")
            
        elif action == "search":
            agent.current_action = "搜尋出口"
            agent.add_log(self.tick, "搜尋", f"尋找出口或安全路線, 理由: {reasoning}")
            
        elif action == "panic":
            agent.current_action = "恐慌中"
            agent.panic_level = min(100, agent.panic_level + 20)
            agent.add_log(self.tick, "恐慌", f"恐慌程度上升, 理由: {reasoning}")
            
        else:
            agent.current_action = "等待觀察"
            agent.add_log(self.tick, "等待", reasoning)
            
        agent.last_action = decision

    def validate_and_adjust_decision(self, agent: Agent, decision: Dict) -> Dict:
        """在高風險情境下對 LLM 決策做安全校正"""
        try:
            room = self.rooms.get(agent.location)
            if not room:
                return decision
            temp = getattr(room, 'temperature', 25)
            vis = getattr(room, 'visibility', 100)
            on_fire = getattr(room, 'on_fire', False)
            smoke = getattr(room, 'smoke', False)
            # 僅以本地環境危險觸發安全校正，避免僅憑訊息或恐慌就強制撤離
            danger = on_fire or smoke or temp > 150 or vis < 20
            if not danger:
                return decision
            # 危險時若選了高風險行為，強制轉為移動至較安全處
            action = decision.get('action', 'stay')
            if action in ['help_others', 'stay', 'search', 'shout', 'panic']:
                safe_target = self._choose_safer_target(agent)
                if safe_target:
                    return {
                        "action": "move",
                        "target": safe_target,
                        "message": decision.get('message', ''),
                        "reasoning": "高風險自動改為優先撤離"
                    }
                else:
                    # 若無路可走，至少呼救
                    return {
                        "action": "shout",
                        "target": decision.get('target', ''),
                        "message": "這裡著火！需要協助！",
                        "reasoning": "無安全路徑，改為呼救"
                    }
            return decision
        except Exception:
            return decision

    def _choose_safer_target(self, agent: Agent) -> Optional[str]:
        """從目前房間的連接中挑選較安全且更接近出口的目標"""
        neighbors = self.connections.get(agent.location, [])
        if not neighbors:
            return None
        # 排序偏好：大門 > 走廊 > 非著火無煙房 > 其他
        def room_score(name: str) -> int:
            if name == '大門':
                return 100
            if name == '走廊':
                return 80
            r = self.rooms.get(name)
            if not r:
                return 0
            score = 50
            if not r.on_fire:
                score += 20
            if not r.smoke:
                score += 10
            # 低溫高能見度更好
            score += max(0, 30 - max(0, getattr(r, 'temperature', 25) - 100) // 10)
            score += min(20, getattr(r, 'visibility', 100) // 10)
            return score
        best = sorted(neighbors, key=lambda n: room_score(n), reverse=True)
        return best[0] if best else None

    async def fire_scenario_tick(self):
        """執行一個時間刻度的事件"""
        self.tick += 1
        tick_log = []
        
        # 1. 環境事件
        if self.tick == 1:
            # 使用配置的起火房間
            fire_room = self.fire_start_room if self.fire_start_room in self.rooms else "廚房"
            self.rooms[fire_room].on_fire = True
            self.rooms[fire_room].smoke = True
            self.rooms[fire_room].temperature = 200  # 起火房間溫度升高
            tick_log.append(f"[環境] {fire_room}起火！產生濃煙")
        
        # 2. 動態火災蔓延機制（Tick 2+）
        if self.tick >= 2:
            # 2.1 煙霧蔓延：從所有著火房間和有煙霧的房間擴散
            smoke_sources = [
                name for name, room in self.rooms.items() 
                if room.on_fire or room.smoke
            ]
            
            for source_room in smoke_sources:
                connected_rooms = self.connections[source_room]
                for connected in connected_rooms:
                    # 如果相鄰房間還沒有煙霧，有一定機率產生煙霧
                    if not self.rooms[connected].smoke:
                        # 如果源房間著火，煙霧蔓延機率較高；如果只是有煙霧，機率較低
                        smoke_probability = 0.6 if self.rooms[source_room].on_fire else 0.3
                        if random.random() < smoke_probability:
                            self.rooms[connected].smoke = True
                            tick_log.append(f"[環境] 煙霧從{source_room}蔓延到{connected}")
            
            # 2.2 火勢蔓延：從所有著火房間蔓延
            fire_rooms = [name for name, room in self.rooms.items() if room.on_fire]
            
            for fire_room in fire_rooms:
                connected_rooms = self.connections[fire_room]
                for connected in connected_rooms:
                    # 大門是出口，不能著火（讓代理人有機會逃生）
                    if self.rooms[connected].is_exit:
                        continue
                    # 如果相鄰房間還沒著火，有一定機率蔓延
                    if not self.rooms[connected].on_fire:
                        # 如果相鄰房間已經有煙霧，火勢蔓延機率較高
                        fire_probability = 0.4 if self.rooms[connected].smoke else 0.2
                        if random.random() < fire_probability:
                            self.rooms[connected].on_fire = True
                            self.rooms[connected].smoke = True  # 著火時同時產生煙霧
                            self.rooms[connected].temperature = 200  # 溫度升高
                            tick_log.append(f"[環境] 火勢從{fire_room}蔓延到{connected}！")
                        
        # 3. 更新環境狀態
        for room in self.rooms.values():
            room.update_environment(self.tick)
            
        # 4. 代理人感知與決策
        for agent in self.agents:
            # 受傷者不再行動；已逃生者允許在 allow_reentry==True 的前提下繼續決策（可返回救援）
            if agent.injured:
                continue
                
            current_room = self.rooms[agent.location]
            
            # 感知火災
            if current_room.on_fire or current_room.smoke:
                agent.known_fire = True
                
            # 更新代理人狀態
            agent.update_status(current_room)
            
            # 建立情境描述
            situation = f"時間{self.tick}: 你在{agent.location}，"
            if current_room.on_fire:
                situation += "房間著火了！"
            elif current_room.smoke:
                situation += "房間有煙霧。"
            else:
                situation += "房間目前安全。"
                
            if agent.messages_received:
                situation += f" 剛收到訊息: {agent.messages_received[-1]['message']}"
                
            # LLM 決策
            if agent.can_move:
                decision = await self.agent_decision_making(agent, situation)
                await self.execute_agent_action(agent, decision)
        
        # 4. 記錄本輪日誌
        self.simulation_log.extend(tick_log)
        return tick_log

    async def run_simulation(self, max_rounds=15, num_agents=5):
        """執行完整模擬"""
        logger.info("=" * 50)
        logger.info("開始火災現場模擬")
        logger.info("=" * 50)
        
        # 建立代理人
        await self.create_agents_from_database(num_agents)
        
        # 顯示初始狀態
        logger.info(f"\n初始狀態:")
        for agent in self.agents:
            logger.info(f"  {agent}")
            
        # 開始模擬
        for round_num in range(max_rounds):
            logger.info(f"\n{'=' * 20} 時間 {round_num + 1} {'=' * 20}")
            
            tick_events = await self.fire_scenario_tick()
            
            # 輸出本輪事件
            for event in tick_events:
                logger.info(event)
                
            # 輸出房間狀態
            logger.info("\n房間狀態:")
            for room in self.rooms.values():
                logger.info(f"  {room}")
                
            # 輸出代理人狀態
            logger.info("\n代理人狀態:")
            for agent in self.agents:
                logger.info(f"  {agent}")
                
            # 檢查是否所有人都逃生或受傷
            active_agents = [a for a in self.agents if not a.escaped and not a.injured]
            if not active_agents:
                logger.info("\n所有代理人都已逃生或受傷，模擬結束")
                break
                
            # 延遲以便觀看
            await asyncio.sleep(0.1)
            
        # 輸出最終結果
        await self.output_final_results()

    async def output_final_results(self):
        """輸出最終模擬結果"""
        logger.info("\n" + "=" * 50)
        logger.info("模擬結束 - 最終結果")
        logger.info("=" * 50)
        
        escaped_count = sum(1 for a in self.agents if a.escaped)
        injured_count = sum(1 for a in self.agents if a.injured)
        
        logger.info(f"\n統計結果:")
        logger.info(f"  總代理人數: {len(self.agents)}")
        logger.info(f"  成功逃生: {escaped_count}")
        logger.info(f"  受傷人數: {injured_count}")
        logger.info(f"  逃生率: {escaped_count/len(self.agents)*100:.1f}%")
        
        logger.info(f"\n個別結果:")
        for agent in self.agents:
            status = "逃生" if agent.escaped else "受傷" if agent.injured else "未脫險"
            logger.info(f"  {agent.name}: {status} (最終健康度: {agent.health})")
            
        # 儲存詳細日誌
        await self.save_simulation_results()

    async def save_simulation_results(self):
        """儲存模擬結果到檔案"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fire_simulation_{timestamp}.json"
        
        results = {
            "timestamp": timestamp,
            "total_agents": len(self.agents),
            "escaped_count": sum(1 for a in self.agents if a.escaped),
            "injured_count": sum(1 for a in self.agents if a.injured),
            "total_ticks": self.tick,
            "agents": [
                {
                    "id": agent.id,
                    "name": agent.name,
                    "traits": agent.traits,
                    "escaped": agent.escaped,
                    "injured": agent.injured,
                    "final_health": agent.health,
                    "final_location": agent.location,
                    "action_log": agent.log
                }
                for agent in self.agents
            ],
            "simulation_log": self.simulation_log
        }
        
        os.makedirs("simulation_results", exist_ok=True)
        filepath = os.path.join("simulation_results", filename)
        
        async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(results, ensure_ascii=False, indent=2))
            
        logger.info(f"\n模擬結果已儲存至: {filepath}")

# =======================================================
# 批次模擬功能
# =======================================================
async def run_batch_simulation(num_simulations=5, num_agents=5):
    """執行批次模擬並統計結果"""
    logger.info(f"開始批次模擬 ({num_simulations} 次)")
    
    all_results = []
    
    for i in range(num_simulations):
        logger.info(f"\n{'#' * 60}")
        logger.info(f"第 {i+1}/{num_simulations} 次模擬")
        logger.info(f"{'#' * 60}")
        
        simulator = FireScenarioSimulator()
        await simulator.initialize()
        await simulator.run_simulation(max_rounds=15, num_agents=num_agents)
        
        # 收集結果
        result = {
            "simulation_id": i+1,
            "escaped_count": sum(1 for a in simulator.agents if a.escaped),
            "injured_count": sum(1 for a in simulator.agents if a.injured),
            "total_ticks": simulator.tick,
            "escape_rate": sum(1 for a in simulator.agents if a.escaped) / len(simulator.agents)
        }
        all_results.append(result)
        
    # 統計分析
    logger.info(f"\n{'=' * 60}")
    logger.info("批次模擬統計結果")
    logger.info(f"{'=' * 60}")
    
    avg_escape_rate = sum(r["escape_rate"] for r in all_results) / len(all_results)
    avg_injured = sum(r["injured_count"] for r in all_results) / len(all_results)
    avg_ticks = sum(r["total_ticks"] for r in all_results) / len(all_results)
    
    logger.info(f"平均逃生率: {avg_escape_rate*100:.1f}%")
    logger.info(f"平均受傷人數: {avg_injured:.1f}")
    logger.info(f"平均模擬時長: {avg_ticks:.1f} 回合")
    
    # 儲存批次結果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_filename = f"batch_simulation_{timestamp}.json"
    
    batch_results = {
        "timestamp": timestamp,
        "num_simulations": num_simulations,
        "num_agents": num_agents,
        "average_escape_rate": avg_escape_rate,
        "average_injured_count": avg_injured,
        "average_ticks": avg_ticks,
        "individual_results": all_results
    }
    
    os.makedirs("simulation_results", exist_ok=True)
    filepath = os.path.join("simulation_results", batch_filename)
    
    async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
        await f.write(json.dumps(batch_results, ensure_ascii=False, indent=2))
        
    logger.info(f"批次結果已儲存至: {filepath}")

# =======================================================
# 主要執行入口
# =======================================================
async def main():
    """主要執行函數"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        # 批次模擬
        num_sims = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        num_agents = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        await run_batch_simulation(num_sims, num_agents)
    else:
        # 單次模擬
        num_agents = int(sys.argv[1]) if len(sys.argv) > 1 else 5
        simulator = FireScenarioSimulator()
        await simulator.initialize()
        await simulator.run_simulation(num_agents=num_agents)

if __name__ == "__main__":
    asyncio.run(main()) 