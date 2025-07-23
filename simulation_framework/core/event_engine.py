"""
通用事件引擎和觸發器系統
Universal Event Engine and Trigger System
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Union
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import random
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class EventPriority(Enum):
    """事件優先級"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5

class TriggerConditionType(Enum):
    """觸發條件類型"""
    TIME_BASED = "time_based"           # 基於時間
    LOCATION_BASED = "location_based"   # 基於位置
    AGENT_BASED = "agent_based"         # 基於代理人
    STATE_BASED = "state_based"         # 基於狀態
    RANDOM = "random"                   # 隨機
    COMPOSITE = "composite"             # 複合條件

@dataclass
class Event:
    """事件"""
    id: str                                    # 事件ID
    name: str                                  # 事件名稱
    description: str                           # 事件描述
    event_type: str                           # 事件類型
    priority: EventPriority                   # 優先級
    tick: int                                 # 發生時間（tick）
    duration: int = 1                         # 持續時間
    affected_locations: List[str] = field(default_factory=list)  # 影響位置
    affected_agents: List[str] = field(default_factory=list)     # 影響代理人
    data: Dict[str, Any] = field(default_factory=dict)          # 事件資料
    is_active: bool = True                    # 是否活躍
    is_resolved: bool = False                 # 是否已解決
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'event_type': self.event_type,
            'priority': self.priority.value,
            'tick': self.tick,
            'duration': self.duration,
            'affected_locations': self.affected_locations,
            'affected_agents': self.affected_agents,
            'data': self.data,
            'is_active': self.is_active,
            'is_resolved': self.is_resolved
        }

class TriggerCondition(ABC):
    """觸發條件基礎類別"""
    
    def __init__(self, condition_type: TriggerConditionType):
        self.condition_type = condition_type
        
    @abstractmethod
    def check(self, simulation_state: Dict[str, Any]) -> bool:
        """檢查條件是否滿足"""
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        pass

class TimeBasedCondition(TriggerCondition):
    """基於時間的觸發條件"""
    
    def __init__(self, target_tick: int, operator: str = ">="):
        super().__init__(TriggerConditionType.TIME_BASED)
        self.target_tick = target_tick
        self.operator = operator  # ">=", "<=", "==", ">", "<"
    
    def check(self, simulation_state: Dict[str, Any]) -> bool:
        current_tick = simulation_state.get('current_tick', 0)
        
        if self.operator == ">=":
            return current_tick >= self.target_tick
        elif self.operator == "<=":
            return current_tick <= self.target_tick
        elif self.operator == "==":
            return current_tick == self.target_tick
        elif self.operator == ">":
            return current_tick > self.target_tick
        elif self.operator == "<":
            return current_tick < self.target_tick
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.condition_type.value,
            'target_tick': self.target_tick,
            'operator': self.operator
        }

class LocationBasedCondition(TriggerCondition):
    """基於位置的觸發條件"""
    
    def __init__(self, location_id: str, property_name: str, 
                 expected_value: Any, operator: str = "=="):
        super().__init__(TriggerConditionType.LOCATION_BASED)
        self.location_id = location_id
        self.property_name = property_name
        self.expected_value = expected_value
        self.operator = operator
    
    def check(self, simulation_state: Dict[str, Any]) -> bool:
        locations = simulation_state.get('locations', {})
        location = locations.get(self.location_id, {})
        current_value = location.get(self.property_name)
        
        if current_value is None:
            return False
        
        if self.operator == "==":
            return current_value == self.expected_value
        elif self.operator == "!=":
            return current_value != self.expected_value
        elif self.operator == ">":
            return current_value > self.expected_value
        elif self.operator == "<":
            return current_value < self.expected_value
        elif self.operator == ">=":
            return current_value >= self.expected_value
        elif self.operator == "<=":
            return current_value <= self.expected_value
        elif self.operator == "contains":
            return self.expected_value in current_value
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.condition_type.value,
            'location_id': self.location_id,
            'property_name': self.property_name,
            'expected_value': self.expected_value,
            'operator': self.operator
        }

class AgentBasedCondition(TriggerCondition):
    """基於代理人的觸發條件"""
    
    def __init__(self, agent_condition: str, threshold: Union[int, float] = 1):
        super().__init__(TriggerConditionType.AGENT_BASED)
        self.agent_condition = agent_condition  # "escaped", "injured", "panic_level_high"
        self.threshold = threshold
    
    def check(self, simulation_state: Dict[str, Any]) -> bool:
        agents = simulation_state.get('agents', [])
        
        if self.agent_condition == "escaped":
            escaped_count = sum(1 for agent in agents if agent.get('escaped', False))
            return escaped_count >= self.threshold
        elif self.agent_condition == "injured":
            injured_count = sum(1 for agent in agents if agent.get('injured', False))
            return injured_count >= self.threshold
        elif self.agent_condition == "panic_level_high":
            high_panic_count = sum(1 for agent in agents if agent.get('panic_level', 0) >= 70)
            return high_panic_count >= self.threshold
        elif self.agent_condition == "all_escaped":
            return all(agent.get('escaped', False) for agent in agents)
        elif self.agent_condition == "any_injured":
            return any(agent.get('injured', False) for agent in agents)
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.condition_type.value,
            'agent_condition': self.agent_condition,
            'threshold': self.threshold
        }

class RandomCondition(TriggerCondition):
    """隨機觸發條件"""
    
    def __init__(self, probability: float, cooldown: int = 0):
        super().__init__(TriggerConditionType.RANDOM)
        self.probability = probability  # 0.0 - 1.0
        self.cooldown = cooldown
        self.last_triggered = -999
    
    def check(self, simulation_state: Dict[str, Any]) -> bool:
        current_tick = simulation_state.get('current_tick', 0)
        
        # 檢查冷卻時間
        if current_tick - self.last_triggered < self.cooldown:
            return False
        
        if random.random() < self.probability:
            self.last_triggered = current_tick
            return True
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.condition_type.value,
            'probability': self.probability,
            'cooldown': self.cooldown
        }

class CompositeCondition(TriggerCondition):
    """複合觸發條件"""
    
    def __init__(self, conditions: List[TriggerCondition], logic: str = "AND"):
        super().__init__(TriggerConditionType.COMPOSITE)
        self.conditions = conditions
        self.logic = logic  # "AND", "OR"
    
    def check(self, simulation_state: Dict[str, Any]) -> bool:
        if not self.conditions:
            return False
        
        results = [condition.check(simulation_state) for condition in self.conditions]
        
        if self.logic == "AND":
            return all(results)
        elif self.logic == "OR":
            return any(results)
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.condition_type.value,
            'conditions': [condition.to_dict() for condition in self.conditions],
            'logic': self.logic
        }

@dataclass
class Trigger:
    """觸發器"""
    id: str                                    # 觸發器ID
    name: str                                  # 觸發器名稱
    condition: TriggerCondition               # 觸發條件
    event_template: Dict[str, Any]            # 事件模板
    is_repeatable: bool = False               # 是否可重複觸發
    is_active: bool = True                    # 是否活躍
    max_triggers: int = -1                    # 最大觸發次數 (-1 表示無限制)
    triggered_count: int = 0                  # 已觸發次數
    
    def can_trigger(self, simulation_state: Dict[str, Any]) -> bool:
        """檢查是否可以觸發"""
        if not self.is_active:
            return False
        
        if self.max_triggers > 0 and self.triggered_count >= self.max_triggers:
            return False
        
        if not self.is_repeatable and self.triggered_count > 0:
            return False
        
        return self.condition.check(simulation_state)
    
    def trigger(self, simulation_state: Dict[str, Any]) -> Event:
        """觸發並創建事件"""
        self.triggered_count += 1
        
        # 創建事件
        event_data = self.event_template.copy()
        event = Event(
            id=f"{self.id}_{self.triggered_count}",
            name=event_data.get('name', f"Event from {self.name}"),
            description=event_data.get('description', ''),
            event_type=event_data.get('event_type', 'trigger'),
            priority=EventPriority(event_data.get('priority', EventPriority.NORMAL.value)),
            tick=simulation_state.get('current_tick', 0),
            duration=event_data.get('duration', 1),
            affected_locations=event_data.get('affected_locations', []),
            affected_agents=event_data.get('affected_agents', []),
            data=event_data.get('data', {})
        )
        
        logger.info(f"觸發器 {self.name} 觸發了事件: {event.name}")
        return event

class EventQueue:
    """事件佇列"""
    
    def __init__(self):
        self.events: List[Event] = []
        self.processed_events: List[Event] = []
    
    def add_event(self, event: Event):
        """添加事件"""
        self.events.append(event)
        # 按優先級和時間排序
        self.events.sort(key=lambda e: (-e.priority.value, e.tick))
        logger.debug(f"添加事件到佇列: {event.name} (優先級: {event.priority.value})")
    
    def get_current_events(self, current_tick: int) -> List[Event]:
        """獲取當前tick的事件"""
        current_events = []
        remaining_events = []
        
        for event in self.events:
            if event.tick <= current_tick and event.is_active:
                current_events.append(event)
                # 檢查事件是否結束
                if current_tick >= event.tick + event.duration:
                    event.is_active = False
                    event.is_resolved = True
                    self.processed_events.append(event)
                else:
                    remaining_events.append(event)
            else:
                remaining_events.append(event)
        
        self.events = remaining_events
        return current_events
    
    def clear_resolved_events(self):
        """清理已解決的事件"""
        self.events = [e for e in self.events if e.is_active]

class EventManager:
    """事件管理器"""
    
    def __init__(self):
        self.triggers: Dict[str, Trigger] = {}
        self.event_queue = EventQueue()
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.global_handlers: List[Callable] = []
        
    def register_trigger(self, trigger: Trigger):
        """註冊觸發器"""
        self.triggers[trigger.id] = trigger
        logger.info(f"註冊觸發器: {trigger.name}")
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """註冊事件處理器"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def register_global_handler(self, handler: Callable):
        """註冊全域事件處理器"""
        self.global_handlers.append(handler)
    
    def add_event(self, event: Event):
        """手動添加事件"""
        self.event_queue.add_event(event)
    
    async def process_tick(self, simulation_state: Dict[str, Any]) -> List[Event]:
        """處理一個時間步"""
        current_tick = simulation_state.get('current_tick', 0)
        triggered_events = []
        
        # 檢查觸發器
        for trigger in self.triggers.values():
            if trigger.can_trigger(simulation_state):
                event = trigger.trigger(simulation_state)
                self.event_queue.add_event(event)
                triggered_events.append(event)
        
        # 獲取當前事件
        current_events = self.event_queue.get_current_events(current_tick)
        all_events = triggered_events + current_events
        
        # 處理事件
        for event in all_events:
            await self._handle_event(event, simulation_state)
        
        return all_events
    
    async def _handle_event(self, event: Event, simulation_state: Dict[str, Any]):
        """處理單個事件"""
        # 執行事件類型特定的處理器
        handlers = self.event_handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event, simulation_state)
                else:
                    handler(event, simulation_state)
            except Exception as e:
                logger.error(f"處理事件 {event.name} 時發生錯誤: {e}")
        
        # 執行全域處理器
        for handler in self.global_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event, simulation_state)
                else:
                    handler(event, simulation_state)
            except Exception as e:
                logger.error(f"全域處理器處理事件 {event.name} 時發生錯誤: {e}")
    
    def get_active_events(self) -> List[Event]:
        """獲取活躍事件"""
        return [e for e in self.event_queue.events if e.is_active]
    
    def get_processed_events(self) -> List[Event]:
        """獲取已處理事件"""
        return self.event_queue.processed_events
    
    def clear_all_events(self):
        """清空所有事件"""
        self.event_queue.events.clear()
        self.event_queue.processed_events.clear()
    
    def export_state(self) -> Dict[str, Any]:
        """匯出狀態"""
        return {
            'triggers': {
                trigger_id: {
                    'id': trigger.id,
                    'name': trigger.name,
                    'is_active': trigger.is_active,
                    'triggered_count': trigger.triggered_count,
                    'condition': trigger.condition.to_dict()
                }
                for trigger_id, trigger in self.triggers.items()
            },
            'active_events': [event.to_dict() for event in self.get_active_events()],
            'processed_events': [event.to_dict() for event in self.get_processed_events()]
        }

# 工廠函數用於創建觸發條件
class TriggerConditionFactory:
    """觸發條件工廠"""
    
    @staticmethod
    def create_time_condition(target_tick: int, operator: str = ">=") -> TimeBasedCondition:
        """創建時間條件"""
        return TimeBasedCondition(target_tick, operator)
    
    @staticmethod
    def create_location_condition(location_id: str, property_name: str, 
                                expected_value: Any, operator: str = "==") -> LocationBasedCondition:
        """創建位置條件"""
        return LocationBasedCondition(location_id, property_name, expected_value, operator)
    
    @staticmethod
    def create_agent_condition(agent_condition: str, threshold: Union[int, float] = 1) -> AgentBasedCondition:
        """創建代理人條件"""
        return AgentBasedCondition(agent_condition, threshold)
    
    @staticmethod
    def create_random_condition(probability: float, cooldown: int = 0) -> RandomCondition:
        """創建隨機條件"""
        return RandomCondition(probability, cooldown)
    
    @staticmethod
    def create_composite_condition(conditions: List[TriggerCondition], logic: str = "AND") -> CompositeCondition:
        """創建複合條件"""
        return CompositeCondition(conditions, logic) 