# Simulation Framework Core Module
# 模擬框架核心模組

from .base_scenario import BaseScenario, ScenarioConfig
from .map_system import MapBuilder, Route, Location, SpatialGraph
from .event_engine import EventManager, Event, Trigger, EventQueue
from .agent_manager import AgentManager, AgentState
from .simulation_engine import SimulationEngine, SimulationState

__version__ = "1.0.0"
__author__ = "Simulation Framework Team"

__all__ = [
    # 情境相關
    "BaseScenario",
    "ScenarioConfig", 
    
    # 地圖系統
    "MapBuilder",
    "Route",
    "Location",
    "SpatialGraph",
    
    # 事件引擎
    "EventManager",
    "Event",
    "Trigger", 
    "EventQueue",
    
    # 代理人管理
    "AgentManager",
    "AgentState",
    
    # 模擬引擎
    "SimulationEngine",
    "SimulationState"
] 