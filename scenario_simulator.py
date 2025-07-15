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
            '大門': Room('大門', is_exit=True),
            '陽台': Room('陽台', is_exit=True)
        }
        
        # 房間連接圖
        self.connections = {
            '客廳': ['廚房', '走廊', '陽台'],
            '廚房': ['客廳'],
            '臥室A': ['走廊'],
            '臥室B': ['走廊'],
            '臥室C': ['走廊'],
            '走廊': ['客廳', '臥室A', '臥室B', '臥室C', '浴室', '大門'],
            '浴室': ['走廊'],
            '大門': ['走廊'],
            '陽台': ['客廳']
        }

    async def create_agents_from_database(self, num_agents=5):
        """從人格資料庫隨機選取代理人"""
        all_personas = list(self.persona_loader.get_all_personas().values())
        if len(all_personas) < num_agents:
            logger.warning(f"資料庫只有 {len(all_personas)} 個人格，將使用全部")
            selected_personas = all_personas
        else:
            selected_personas = random.sample(all_personas, num_agents)
        
        # 隨機分配初始位置
        possible_locations = ['客廳', '臥室A', '臥室B', '臥室C', '廚房']
        
        self.agents = []
        for persona in selected_personas:
            initial_location = random.choice(possible_locations)
            agent = Agent(persona['基本資料']['id'], persona, initial_location)
            self.agents.append(agent)
            
            # 更新房間內代理人列表
            self.rooms[initial_location].agents.append(agent.id)
            
            logger.info(f"建立代理人: {agent.name} (位於 {initial_location})")
            
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

            prompt = [
                {
                    "role": "system",
                    "content": f"""你是 {agent.name}，根據以下人格特質與記憶資料來做決策：
人格特質: {agent.traits}
社交能力: {agent.social_abilities}  
能力屬性: {agent.ability_attributes}
記憶資料: {retrieved_docs[:2] if retrieved_docs else '無特別記憶'}

請根據當前情境做出最符合你人格的決策。你的回應必須是以下JSON格式：
{{
    "action": "move|stay|shout|help_others|search|panic",
    "target": "房間名稱或人名（如果適用）",
    "message": "如果要說話或呼救的內容",
    "reasoning": "決策理由（50字內）"
}}

可能的行動：
- move: 移動到其他房間
- stay: 留在原地
- shout: 大聲呼救或警告
- help_others: 幫助他人
- search: 搜尋出口或安全路線
- panic: 恐慌行為

請只回傳JSON，不要其他文字。
                    """
                },
                {"role": "user", "content": context}
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
                return decision
            except json.JSONDecodeError:
                # 如果解析失敗，返回預設行為
                logger.warning(f"{agent.name} 決策解析失敗，使用預設行為")
                return {
                    "action": "move" if agent.known_fire else "stay",
                    "target": random.choice(available_exits) if agent.known_fire else agent.location,
                    "message": "",
                    "reasoning": "解析失敗的預設行為"
                }
                
        except Exception as e:
            logger.error(f"代理人 {agent.name} 決策失敗: {e}")
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
        
        if action == "move" and target in self.connections[agent.location]:
            # 移動
            old_location = agent.location
            self.rooms[old_location].agents.remove(agent.id)
            agent.location = target
            self.rooms[target].agents.append(agent.id)
            
            if target in ['大門', '陽台']:
                agent.escaped = True
                agent.current_action = f"已逃生到{target}"
                agent.add_log(self.tick, "逃生成功", f"從 {old_location} 逃到 {target}")
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

    async def fire_scenario_tick(self):
        """執行一個時間刻度的事件"""
        self.tick += 1
        tick_log = []
        
        # 1. 環境事件
        if self.tick == 1:
            self.rooms['廚房'].on_fire = True
            self.rooms['廚房'].smoke = True
            tick_log.append(f"[環境] 廚房起火！產生濃煙")
            
        elif self.tick == 2:
            self.rooms['客廳'].smoke = True
            tick_log.append(f"[環境] 煙霧蔓延到客廳")
            
        elif self.tick == 3:
            self.rooms['走廊'].smoke = True
            tick_log.append(f"[環境] 煙霧蔓延到走廊")
            
        elif self.tick == 4:
            self.rooms['廚房'].temperature = 200
            self.rooms['客廳'].on_fire = True
            tick_log.append(f"[環境] 火勢蔓延到客廳！")
            
        elif self.tick >= 5:
            # 火勢隨機蔓延
            fire_rooms = [name for name, room in self.rooms.items() if room.on_fire]
            for room_name in fire_rooms:
                connected_rooms = self.connections[room_name]
                for connected in connected_rooms:
                    if not self.rooms[connected].on_fire and random.random() < 0.3:
                        self.rooms[connected].on_fire = True
                        tick_log.append(f"[環境] 火勢蔓延到{connected}！")
                        
        # 2. 更新環境狀態
        for room in self.rooms.values():
            room.update_environment(self.tick)
            
        # 3. 代理人感知與決策
        for agent in self.agents:
            if agent.escaped or agent.injured:
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