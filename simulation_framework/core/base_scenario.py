"""
基礎情境類別和配置系統
Base Scenario Classes and Configuration System
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import json
import yaml
from pathlib import Path

class ScenarioType(Enum):
    """情境類型"""
    FIRE = "fire"                    # 火災
    EARTHQUAKE = "earthquake"        # 地震  
    FLOOD = "flood"                 # 水災
    BLACKOUT = "blackout"           # 停電
    TERRORIST = "terrorist"         # 恐攻
    EPIDEMIC = "epidemic"           # 疫情
    CUSTOM = "custom"               # 自訂

class EventType(Enum):
    """事件類型"""
    ENVIRONMENTAL = "environmental"  # 環境變化
    SOCIAL = "social"               # 社交互動
    SYSTEM = "system"               # 系統事件
    AGENT_ACTION = "agent_action"   # 代理人行動
    TRIGGER = "trigger"             # 觸發事件

@dataclass
class ScenarioConfig:
    """情境配置"""
    # 基本資訊
    name: str                           # 情境名稱
    description: str                    # 情境描述  
    scenario_type: ScenarioType         # 情境類型
    version: str = "1.0.0"             # 版本
    
    # 地圖設定
    map_config: Dict[str, Any] = field(default_factory=dict)
    
    # 代理人設定
    min_agents: int = 1                 # 最少代理人數
    max_agents: int = 20                # 最多代理人數
    default_agents: int = 5             # 預設代理人數
    agent_spawn_areas: List[str] = field(default_factory=list)  # 代理人生成區域
    
    # 時間設定
    max_ticks: int = 100                # 最大時間步數
    tick_duration: float = 1.0          # 每步時間長度（秒）
    real_time_factor: float = 1.0       # 實時比例
    
    # 情境參數
    scenario_params: Dict[str, Any] = field(default_factory=dict)
    
    # 觸發條件
    start_triggers: List[Dict[str, Any]] = field(default_factory=list)
    end_conditions: List[Dict[str, Any]] = field(default_factory=list)
    
    # 評估指標
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    
    def save_to_file(self, filepath: str):
        """儲存配置到檔案"""
        path = Path(filepath)
        data = {
            'name': self.name,
            'description': self.description,
            'scenario_type': self.scenario_type.value,
            'version': self.version,
            'map_config': self.map_config,
            'min_agents': self.min_agents,
            'max_agents': self.max_agents,
            'default_agents': self.default_agents,
            'agent_spawn_areas': self.agent_spawn_areas,
            'max_ticks': self.max_ticks,
            'tick_duration': self.tick_duration,
            'real_time_factor': self.real_time_factor,
            'scenario_params': self.scenario_params,
            'start_triggers': self.start_triggers,
            'end_conditions': self.end_conditions,
            'success_criteria': self.success_criteria
        }
        
        if path.suffix.lower() == '.yaml' or path.suffix.lower() == '.yml':
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True, indent=2)
        else:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'ScenarioConfig':
        """從檔案載入配置"""
        path = Path(filepath)
        
        if path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        else:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        # 轉換 scenario_type
        if isinstance(data.get('scenario_type'), str):
            data['scenario_type'] = ScenarioType(data['scenario_type'])
        
        return cls(**data)

class BaseScenario(ABC):
    """情境基礎類別"""
    
    def __init__(self, config: ScenarioConfig):
        self.config = config
        self.current_tick = 0
        self.is_running = False
        self.is_finished = False
        self.events = []
        self.agents = []
        self.environment_state = {}
        
    @abstractmethod
    async def initialize(self, map_graph, agents: List[Any]) -> bool:
        """
        初始化情境
        
        Args:
            map_graph: 地圖圖形結構
            agents: 代理人列表
            
        Returns:
            bool: 初始化是否成功
        """
        pass
    
    @abstractmethod
    async def process_tick(self) -> List[Dict[str, Any]]:
        """
        處理一個時間步
        
        Returns:
            List[Dict]: 本回合產生的事件列表
        """
        pass
    
    @abstractmethod
    def check_end_conditions(self) -> Tuple[bool, str]:
        """
        檢查結束條件
        
        Returns:
            Tuple[bool, str]: (是否結束, 結束原因)
        """
        pass
    
    @abstractmethod
    def get_situation_context(self, agent_id: str) -> str:
        """
        獲取代理人的情境脈絡
        
        Args:
            agent_id: 代理人ID
            
        Returns:
            str: 情境描述
        """
        pass
    
    @abstractmethod
    def get_available_actions(self, agent_id: str) -> List[str]:
        """
        獲取代理人可用行動
        
        Args:
            agent_id: 代理人ID
            
        Returns:
            List[str]: 可用行動列表
        """
        pass
    
    def add_event(self, event_type: EventType, description: str, data: Optional[Dict[str, Any]] = None):
        """添加事件"""
        event = {
            'tick': self.current_tick,
            'type': event_type.value,
            'description': description,
            'data': data or {},
            'timestamp': self.current_tick * self.config.tick_duration
        }
        self.events.append(event)
        return event
    
    def get_recent_events(self, count: int = 10) -> List[Dict[str, Any]]:
        """獲取最近的事件"""
        return self.events[-count:] if self.events else []
    
    def get_environment_status(self) -> Dict[str, Any]:
        """獲取環境狀態"""
        return {
            'tick': self.current_tick,
            'is_running': self.is_running,
            'is_finished': self.is_finished,
            'scenario_type': self.config.scenario_type.value,
            'environment_state': self.environment_state.copy()
        }
    
    def calculate_success_metrics(self) -> Dict[str, float]:
        """計算成功指標"""
        metrics = {}
        
        if not self.agents:
            return metrics
            
        total_agents = len(self.agents)
        
        # 基本統計
        escaped_count = sum(1 for agent in self.agents if getattr(agent, 'escaped', False))
        injured_count = sum(1 for agent in self.agents if getattr(agent, 'injured', False))
        
        metrics.update({
            'total_agents': total_agents,
            'escaped_count': escaped_count,
            'injured_count': injured_count,
            'escape_rate': escaped_count / total_agents if total_agents > 0 else 0,
            'injury_rate': injured_count / total_agents if total_agents > 0 else 0,
            'survival_rate': (total_agents - injured_count) / total_agents if total_agents > 0 else 0
        })
        
        # 情境特定指標
        scenario_metrics = self._calculate_scenario_specific_metrics()
        metrics.update(scenario_metrics)
        
        return metrics
    
    def _calculate_scenario_specific_metrics(self) -> Dict[str, float]:
        """計算情境特定的成功指標（子類別覆寫）"""
        return {}
    
    def export_simulation_data(self) -> Dict[str, Any]:
        """匯出模擬資料"""
        return {
            'config': {
                'name': self.config.name,
                'scenario_type': self.config.scenario_type.value,
                'version': self.config.version
            },
            'simulation_state': {
                'total_ticks': self.current_tick,
                'is_finished': self.is_finished,
                'duration': self.current_tick * self.config.tick_duration
            },
            'events': self.events,
            'final_metrics': self.calculate_success_metrics(),
            'environment_state': self.environment_state
        }

class ScenarioRegistry:
    """情境註冊器"""
    
    _scenarios: Dict[ScenarioType, type] = {}
    
    @classmethod
    def register(cls, scenario_type: ScenarioType, scenario_class: type):
        """註冊情境類別"""
        cls._scenarios[scenario_type] = scenario_class
    
    @classmethod
    def get_scenario_class(cls, scenario_type: ScenarioType) -> Optional[type]:
        """獲取情境類別"""
        return cls._scenarios.get(scenario_type)
    
    @classmethod
    def list_available_scenarios(cls) -> List[ScenarioType]:
        """列出可用情境"""
        return list(cls._scenarios.keys())
    
    @classmethod
    def create_scenario(cls, config: ScenarioConfig) -> BaseScenario:
        """創建情境實例"""
        scenario_class = cls.get_scenario_class(config.scenario_type)
        if not scenario_class:
            raise ValueError(f"未註冊的情境類型: {config.scenario_type}")
        return scenario_class(config)

# 裝飾器用於自動註冊情境
def register_scenario(scenario_type: ScenarioType):
    """情境註冊裝飾器"""
    def decorator(cls):
        ScenarioRegistry.register(scenario_type, cls)
        return cls
    return decorator 