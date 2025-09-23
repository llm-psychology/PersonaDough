# life_simulation_backend.py
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pydantic import BaseModel
import random

logger = logging.getLogger(__name__)

# 數據模型
@dataclass
class AgentStatus:
    agent_id: str
    persona_id: str
    name: str = ""
    position: Dict[str, float] = None
    current_location: str = ""
    current_activity: Dict[str, Any] = None
    health: float = 100.0
    energy: float = 100.0
    happiness: float = 50.0
    social_needs: float = 50.0
    hunger: float = 0.0
    is_moving: bool = False
    last_update: str = ""
    
    def __post_init__(self):
        if self.position is None:
            self.position = {"x": 0.0, "y": 0.0}
        if self.current_activity is None:
            self.current_activity = {
                "name": "待機中",
                "type": "Idle",
                "location": "",
                "progress": 0.0
            }
        if not self.last_update:
            self.last_update = datetime.now().isoformat()

class LifeSimulationRequest(BaseModel):
    agent_count: int = 3
    time_speed: float = 60.0  # 時間加速倍率
    simulation_mode: str = "autonomous"  # autonomous, guided, hybrid

class LifeSimulationResponse(BaseModel):
    success: bool
    message: str
    agents: List[Dict[str, Any]]
    simulation_id: str = ""

class LifeSimulationManager:
    def __init__(self):
        self.agents: Dict[str, AgentStatus] = {}
        self.simulation_active = False
        self.time_speed = 60.0
        self.last_tick = datetime.now()
        self.simulation_id = ""
        
        # 活動模板
        self.activity_templates = {
            "sleep": {"name": "睡覺", "type": "Sleep", "duration": 480, "energy_gain": 80},
            "work": {"name": "工作", "type": "Work", "duration": 480, "energy_cost": 30},
            "study": {"name": "學習", "type": "Study", "duration": 240, "energy_cost": 20},
            "eat": {"name": "用餐", "type": "Eat", "duration": 60, "hunger_restore": 80},
            "play_football": {"name": "踢足球", "type": "PlayFootball", "duration": 120, "happiness_gain": 30},
            "rest": {"name": "休息", "type": "Rest", "duration": 120, "energy_gain": 20},
            "socialize": {"name": "社交", "type": "TalkTo", "duration": 90, "social_gain": 40}
        }
    
    async def start_simulation(self, request: LifeSimulationRequest) -> LifeSimulationResponse:
        """啟動生活模擬"""
        try:
            self.time_speed = request.time_speed
            self.simulation_id = f"life_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 創建代理人
            await self.create_default_agents(request.agent_count)
            
            # 啟動模擬循環
            self.simulation_active = True
            asyncio.create_task(self.simulation_loop())
            
            agent_list = [asdict(agent) for agent in self.agents.values()]
            
            logger.info(f"生活模擬已啟動，ID: {self.simulation_id}, 代理人數量: {len(self.agents)}")
            
            return LifeSimulationResponse(
                success=True,
                message=f"生活模擬已啟動，共創建 {len(self.agents)} 個代理人",
                agents=agent_list,
                simulation_id=self.simulation_id
            )
            
        except Exception as e:
            logger.error(f"啟動生活模擬失敗: {e}", exc_info=True)
            return LifeSimulationResponse(
                success=False,
                message=f"啟動失敗: {str(e)}",
                agents=[]
            )
    
    async def create_default_agents(self, count: int = 3):
        """創建預設代理人"""
        default_personas = ["687a1b2d5b", "867b21d0fe", "eef4720230"]
        
        for i in range(min(count, len(default_personas))):
            persona_id = default_personas[i]
            agent_id = f"agent_{i+1}"
            
            # 嘗試從persona系統獲取名字
            name = await self.get_persona_name(persona_id)
            
            agent = AgentStatus(
                agent_id=agent_id,
                persona_id=persona_id,
                name=name,
                position={"x": random.uniform(-5, 5), "y": random.uniform(-5, 5)},
                current_location="home",
                health=random.uniform(80, 100),
                energy=random.uniform(60, 100),
                happiness=random.uniform(40, 80),
                social_needs=random.uniform(30, 70),
                hunger=random.uniform(0, 30)
            )
            
            self.agents[agent_id] = agent
            logger.info(f"創建代理人: {agent_id} ({name})")
    
    async def get_persona_name(self, persona_id: str) -> str:
        """從persona系統獲取角色名字"""
        try:
            # 嘗試從persona_loader獲取
            from module.PERSONA_loader import PersonaLoader
            loader = PersonaLoader()
            await loader.wait_until_ready()
            
            persona_data = loader.get_persona_by_id(persona_id)
            if persona_data:
                return persona_data.get('基本資料', {}).get('姓名', f'角色_{persona_id[:8]}')
            
        except Exception as e:
            logger.warning(f"獲取persona名字失敗: {e}")
        
        return f'角色_{persona_id[:8]}'
    
    async def create_default_agent_status(self, persona_id: str) -> Dict[str, Any]:
        """為指定persona創建預設代理人狀態"""
        try:
            # 查找現有代理人
            for agent in self.agents.values():
                if agent.persona_id == persona_id:
                    return asdict(agent)
            
            # 創建新代理人
            agent_id = f"agent_{len(self.agents) + 1}"
            name = await self.get_persona_name(persona_id)
            
            agent = AgentStatus(
                agent_id=agent_id,
                persona_id=persona_id,
                name=name,
                position={"x": random.uniform(-5, 5), "y": random.uniform(-5, 5)},
                current_location="home"
            )
            
            self.agents[agent_id] = agent
            logger.info(f"為persona {persona_id} 創建新代理人: {agent_id}")
            
            return asdict(agent)
            
        except Exception as e:
            logger.error(f"創建預設代理人狀態失敗: {e}")
            return {
                "agent_id": f"temp_{persona_id[:8]}",
                "persona_id": persona_id,
                "name": f"角色_{persona_id[:8]}",
                "position": {"x": 0.0, "y": 0.0},
                "current_location": "home",
                "current_activity": {
                    "name": "待機中",
                    "type": "Idle", 
                    "location": "",
                    "progress": 0.0
                },
                "health": 100.0,
                "energy": 80.0,
                "happiness": 50.0,
                "social_needs": 50.0,
                "hunger": 20.0,
                "is_moving": False,
                "last_update": datetime.now().isoformat()
            }
    
    def get_agent_status(self, agent_id: str) -> Optional[AgentStatus]:
        """獲取代理人狀態"""
        return self.agents.get(agent_id)
    
    def get_agent_by_persona_id(self, persona_id: str) -> Optional[AgentStatus]:
        """通過persona_id獲取代理人狀態"""
        for agent in self.agents.values():
            if agent.persona_id == persona_id:
                return agent
        return None
    
    def get_all_agents_status(self) -> List[Dict[str, Any]]:
        """獲取所有代理人狀態"""
        return [asdict(agent) for agent in self.agents.values()]
    
    async def simulation_loop(self):
        """主要模擬循環"""
        logger.info("生活模擬循環已啟動")
        
        while self.simulation_active:
            try:
                await self.update_agents()
                await asyncio.sleep(1.0 / self.time_speed)  # 控制模擬速度
                
            except Exception as e:
                logger.error(f"模擬循環錯誤: {e}")
                await asyncio.sleep(1.0)
    
    async def update_agents(self):
        """更新所有代理人狀態"""
        for agent in self.agents.values():
            await self.update_single_agent(agent)
    
    async def update_single_agent(self, agent: AgentStatus):
        """更新單個代理人"""
        try:
            # 更新基本需求
            self.update_basic_needs(agent)
            
            # 決定新活動
            if not agent.is_moving and agent.current_activity.get("name") == "待機中":
                await self.decide_next_activity(agent)
            
            # 更新時間戳
            agent.last_update = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"更新代理人 {agent.agent_id} 失敗: {e}")
    
    def update_basic_needs(self, agent: AgentStatus):
        """更新基本需求"""
        # 隨時間自然變化
        agent.hunger = min(100, agent.hunger + 0.5)
        agent.energy = max(0, agent.energy - 0.3)
        
        if agent.energy < 20:
            agent.happiness = max(0, agent.happiness - 0.5)
        
        if agent.hunger > 80:
            agent.happiness = max(0, agent.happiness - 0.3)
    
    async def decide_next_activity(self, agent: AgentStatus):
        """決定代理人的下一個活動"""
        try:
            # 基於需求決定活動
            if agent.hunger > 70:
                activity = self.activity_templates["eat"].copy()
            elif agent.energy < 30:
                activity = self.activity_templates["sleep"].copy()
            elif agent.happiness < 30:
                activity = self.activity_templates["play_football"].copy()
            elif agent.social_needs < 30:
                activity = self.activity_templates["socialize"].copy()
            else:
                # 隨機選擇活動
                activities = ["work", "study", "rest"]
                chosen = random.choice(activities)
                activity = self.activity_templates[chosen].copy()
            
            # 設定位置
            location = self.get_activity_location(activity["type"])
            activity["location"] = location
            
            agent.current_activity = activity
            agent.is_moving = True
            
            logger.debug(f"代理人 {agent.agent_id} 開始活動: {activity['name']}")
            
        except Exception as e:
            logger.error(f"決定活動失敗: {e}")
    
    def get_activity_location(self, activity_type: str) -> str:
        """根據活動類型獲取位置"""
        location_map = {
            "Sleep": "home",
            "Work": "workplace",
            "Study": "school",
            "Eat": "restaurant",
            "PlayFootball": "soccer_field",
            "Rest": "home",
            "TalkTo": "park"
        }
        return location_map.get(activity_type, "home")
    
    def get_location_coordinates(self, location_id: str) -> Dict[str, float]:
        """根據位置ID獲取座標"""
        location_coords = {
            "home": {"x": 0.0, "y": 0.0},
            "school": {"x": 10.0, "y": 10.0},
            "workplace": {"x": -10.0, "y": 5.0},
            "soccer_field": {"x": 20.0, "y": -10.0},
            "restaurant": {"x": -5.0, "y": -5.0},
            "park": {"x": 15.0, "y": 5.0},
        }
        return location_coords.get(location_id, {"x": 0.0, "y": 0.0})
    
    async def execute_agent_command(self, agent_id: str, command: str, params: Dict[str, Any] = None):
        """執行代理人指令"""
        # 嘗試通過不同方式找到代理人
        agent = self.agents.get(agent_id)
        if not agent:
            agent = self.get_agent_by_persona_id(agent_id)
        
        if not agent:
            return {"success": False, "message": f"代理人 {agent_id} 不存在"}
        
        if params is None:
            params = {}
        
        try:
            if command == "move_to":
                location = params.get("location", "home")
                speed = params.get("speed", 1.0)
                
                # 獲取位置座標
                location_coords = self.get_location_coordinates(location)
                
                agent.current_activity = {
                    "name": f"移動到 {location}",
                    "type": "MoveTo",
                    "location": f"{location_coords['x']},{location_coords['y']}",
                    "progress": speed
                }
                agent.is_moving = True
                agent.current_location = location
                
                return {
                    "success": True, 
                    "message": f"代理人開始移動到 {location}",
                    "estimated_time": 5.0
                }
                
            elif command == "move_to_coordinates":
                x = params.get("x", 0.0)
                y = params.get("y", 0.0)
                speed = params.get("speed", 1.0)
                
                agent.current_activity = {
                    "name": f"移動到座標 ({x},{y})",
                    "type": "MoveTo",
                    "location": f"{x},{y}",
                    "progress": speed
                }
                agent.is_moving = True
                agent.position = {"x": x, "y": y}
                
                return {
                    "success": True,
                    "message": f"代理人開始移動到座標 ({x},{y})",
                    "estimated_time": 5.0
                }
                
            elif command == "start_activity":
                activity_type = params.get("activity", "rest")
                if activity_type in self.activity_templates:
                    activity = self.activity_templates[activity_type].copy()
                    location = self.get_activity_location(activity["type"])
                    location_coords = self.get_location_coordinates(location)
                    activity["location"] = f"{location_coords['x']},{location_coords['y']}"
                    agent.current_activity = activity
                    return {"success": True, "message": f"代理人開始 {activity['name']}"}
                
            return {"success": False, "message": f"未知指令: {command}"}
            
        except Exception as e:
            logger.error(f"執行指令失敗: {e}")
            return {"success": False, "message": f"執行失敗: {str(e)}"}
    
    def stop_simulation(self):
        """停止模擬"""
        self.simulation_active = False
        logger.info("生活模擬已停止")

# 全域管理器實例
life_simulation_manager = LifeSimulationManager()
