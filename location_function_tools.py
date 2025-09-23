# location_function_tools.py
import json
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 位置相關的數據模型
@dataclass
class LocationInfo:
    id: str
    name: str
    type: str
    position: Dict[str, float]  # {"x": 0.0, "y": 0.0}
    description: str = ""

@dataclass  
class CharacterLocationStatus:
    character_id: str
    character_name: str
    current_position: Dict[str, float]
    target_location: Optional[str] = None
    is_moving: bool = False
    movement_progress: float = 0.0

# 角色位置記憶
@dataclass
class CharacterLocationMemory:
    character_id: str
    preferred_locations: Dict[str, str] = None  # location_type -> location_id
    known_locations: List[str] = None          # 所有認識的位置ID
    
    def __post_init__(self):
        if self.preferred_locations is None:
            self.preferred_locations = {}
        if self.known_locations is None:
            self.known_locations = []

# 預設位置定義
DEFAULT_LOCATIONS = {
    "home_default": LocationInfo("home_default", "預設家", "Home", {"x": 0.0, "y": 0.0}, "預設家位置"),
    "school_default": LocationInfo("school_default", "預設學校", "School", {"x": 10.0, "y": 10.0}, "預設學校"),
    "workplace_default": LocationInfo("workplace_default", "預設工作場所", "Workplace", {"x": -10.0, "y": 5.0}, "預設工作地點"),
    "soccer_field_default": LocationInfo("soccer_field_default", "預設足球場", "SoccerField", {"x": 20.0, "y": -10.0}, "預設運動場所"),
    "restaurant_default": LocationInfo("restaurant_default", "預設餐廳", "Restaurant", {"x": -5.0, "y": -5.0}, "預設用餐地點"),
    
    # 可以動態添加更多位置
    "home_house1": LocationInfo("home_house1", "1號房子", "Home", {"x": 2.0, "y": 3.0}, "社區1號房"),
    "home_house2": LocationInfo("home_house2", "2號房子", "Home", {"x": -3.0, "y": 2.0}, "社區2號房"),
    "home_apartment_a": LocationInfo("home_apartment_a", "A棟公寓", "Home", {"x": 5.0, "y": -2.0}, "公寓A棟"),
    "home_apartment_b": LocationInfo("home_apartment_b", "B棟公寓", "Home", {"x": 7.0, "y": -1.0}, "公寓B棟"),
    
    "school_primary": LocationInfo("school_primary", "小學", "School", {"x": 12.0, "y": 8.0}, "小學部"),
    "school_secondary": LocationInfo("school_secondary", "中學", "School", {"x": 15.0, "y": 12.0}, "中學部"),
    
    "workplace_office_a": LocationInfo("workplace_office_a", "A公司", "Workplace", {"x": -12.0, "y": 7.0}, "A公司辦公室"),
    "workplace_office_b": LocationInfo("workplace_office_b", "B公司", "Workplace", {"x": -8.0, "y": 3.0}, "B公司辦公室"),
}

# 智慧位置管理器
class SmartLocationManager:
    def __init__(self):
        self.locations = DEFAULT_LOCATIONS.copy()
        self.character_status = {}  # character_id -> CharacterLocationStatus
        self.character_memories = {}  # character_id -> CharacterLocationMemory
        
        # 初始化預設角色記憶
        self._init_default_character_memories()
        
    def _init_default_character_memories(self):
        """為預設角色設定記憶"""
        default_characters = ["687a1b2d5b", "867b21d0fe", "eef4720230"]
        
        for i, char_id in enumerate(default_characters):
            memory = CharacterLocationMemory(character_id=char_id)
            
            # 為每個角色分配不同的偏好位置
            if i == 0:  # 第一個角色
                memory.preferred_locations = {
                    "Home": "home_house1",
                    "School": "school_primary", 
                    "Workplace": "workplace_office_a",
                    "SoccerField": "soccer_field_default",
                    "Restaurant": "restaurant_default"
                }
            elif i == 1:  # 第二個角色
                memory.preferred_locations = {
                    "Home": "home_house2",
                    "School": "school_secondary",
                    "Workplace": "workplace_office_b", 
                    "SoccerField": "soccer_field_default",
                    "Restaurant": "restaurant_default"
                }
            else:  # 第三個角色
                memory.preferred_locations = {
                    "Home": "home_apartment_a",
                    "School": "school_primary",
                    "Workplace": "workplace_office_a",
                    "SoccerField": "soccer_field_default", 
                    "Restaurant": "restaurant_default"
                }
            
            # 記錄所有認識的位置
            memory.known_locations = list(memory.preferred_locations.values())
            
            self.character_memories[char_id] = memory
            logger.info(f"初始化角色 {char_id} 的位置記憶")
    
    def get_location(self, location_id: str) -> Optional[LocationInfo]:
        """獲取位置資訊"""
        return self.locations.get(location_id)
    
    def add_location(self, location: LocationInfo):
        """添加新位置"""
        self.locations[location.id] = location
        logger.info(f"添加新位置: {location.id} - {location.name}")
    
    def set_character_preferred_location(self, character_id: str, location_type: str, location_id: str):
        """設定角色對某類型位置的偏好"""
        if character_id not in self.character_memories:
            self.character_memories[character_id] = CharacterLocationMemory(character_id=character_id)
        
        memory = self.character_memories[character_id]
        memory.preferred_locations[location_type] = location_id
        
        if location_id not in memory.known_locations:
            memory.known_locations.append(location_id)
        
        logger.info(f"設定角色 {character_id} 的 {location_type} 偏好為: {location_id}")
    
    def get_character_preferred_location(self, character_id: str, location_type: str) -> Optional[LocationInfo]:
        """獲取角色偏好的位置"""
        if character_id in self.character_memories:
            memory = self.character_memories[character_id]
            location_id = memory.preferred_locations.get(location_type)
            if location_id:
                return self.locations.get(location_id)
        
        # 如果沒有偏好設定，返回預設位置
        default_id = f"{location_type.lower()}_default"
        return self.locations.get(default_id)
    
    def set_shared_location(self, character_ids: List[str], location_type: str, location_id: str):
        """設定多個角色共享同一位置（如室友）"""
        for char_id in character_ids:
            self.set_character_preferred_location(char_id, location_type, location_id)
        
        logger.info(f"設定 {len(character_ids)} 個角色共享 {location_type}: {location_id}")
    
    def get_all_locations_by_type(self, location_type: str) -> List[LocationInfo]:
        """獲取指定類型的所有位置"""
        return [loc for loc in self.locations.values() if loc.type == location_type]
    
    def get_character_memory(self, character_id: str) -> Optional[CharacterLocationMemory]:
        """獲取角色記憶"""
        return self.character_memories.get(character_id)
    
    def add_character_known_location(self, character_id: str, location_id: str):
        """讓角色認識新位置"""
        if character_id not in self.character_memories:
            self.character_memories[character_id] = CharacterLocationMemory(character_id=character_id)
        
        memory = self.character_memories[character_id]
        if location_id not in memory.known_locations:
            memory.known_locations.append(location_id)
            logger.info(f"角色 {character_id} 認識了新位置: {location_id}")

# 全域位置管理器實例
location_manager = SmartLocationManager()

# =======================================================
# Function Calling 工具函數
# =======================================================

def move_character_to_location(character_id: str, location_id: str, movement_speed: float = 1.0) -> Dict[str, Any]:
    """
    智慧角色移動到位置 - 使用角色記憶系統
    
    Args:
        character_id: 角色ID
        location_id: 目標位置ID 或 位置類型 (如 "home", "school", "workplace")
        movement_speed: 移動速度倍數 (0.5-3.0)
    
    Returns:
        Dict: 包含 success, message, estimated_time, location_details 的結果
    """
    try:
        location = None
        movement_strategy = "direct"  # direct, smart, fallback
        
        # 1. 智慧位置解析 - 優先使用角色記憶
        if location_id.lower() in ["home", "家", "住所"]:
            # 特殊處理：回家
            home_location = location_manager.get_character_home(character_id)
            if home_location:
                location = location_manager.get_location(home_location)
                movement_strategy = "smart_home"
                logger.info(f"角色 {character_id} 使用智慧記憶回家: {home_location}")
            else:
                # 沒有記憶的家，創建一個
                location_manager.set_character_home(character_id, "home_01")
                location = location_manager.get_location("home_01")
                movement_strategy = "auto_assign_home"
                logger.info(f"角色 {character_id} 自動分配住所: home_01")
        
        elif location_id.lower() in ["school", "學校", "workplace", "工作", "soccerfield", "足球場", "restaurant", "餐廳"]:
            # 其他位置類型的智慧選擇
            location_type_map = {
                "school": "School", "學校": "School",
                "workplace": "Workplace", "工作": "Workplace", 
                "soccerfield": "SoccerField", "足球場": "SoccerField",
                "restaurant": "Restaurant", "餐廳": "Restaurant"
            }
            
            location_type = location_type_map.get(location_id.lower())
            if location_type:
                # 使用角色偏好或智慧推薦
                preferred_location = location_manager.get_character_preferred_location(character_id, location_type)
                if preferred_location:
                    location = preferred_location
                    movement_strategy = "smart_preference"
                else:
                    # 智慧推薦最佳位置
                    all_locations = location_manager.get_all_locations_by_type(location_type)
                    if all_locations:
                        location = all_locations[0]  # 選第一個可用的
                        movement_strategy = "smart_recommendation"
                        # 記錄這個位置給角色
                        location_manager.add_character_known_location(character_id, location.id)
        
        else:
            # 2. 直接位置ID查找
            location = location_manager.get_location(location_id)
            movement_strategy = "direct"
        
        if not location:
            return {
                "success": False,
                "message": f"找不到位置: {location_id}",
                "estimated_time": 0.0
            }
        
        # 計算預計到達時間
        estimated_time = 5.0 / movement_speed
        
        # 更新角色狀態 (這裡需要與life_simulation_manager整合)
        try:
            from life_simulation_backend import life_simulation_manager
            agent = life_simulation_manager.get_agent_by_persona_id(character_id)
            if agent:
                agent.current_activity = {
                    "name": f"移動到 {location.name}",
                    "type": "MoveTo",
                    "location": f"{location.position['x']},{location.position['y']}",
                    "progress": movement_speed
                }
                agent.is_moving = True
                agent.current_location = location.id
        except ImportError:
            logger.warning("life_simulation_manager 不可用，無法更新角色狀態")
        
        logger.info(f"角色 {character_id} 開始移動到 {location.name} ({location.position['x']}, {location.position['y']})")
        
        # 記錄造訪到角色記憶中
        location_manager.add_character_known_location(character_id, location.id)
        
        return {
            "success": True,
            "message": f"✅ 角色 {character_id} 使用{movement_strategy}策略移動到 {location.name}",
            "estimated_time": estimated_time,
            "target_position": location.position,
            "location_details": {
                "id": location.id,
                "name": location.name,
                "type": location.type,
                "strategy": movement_strategy
            }
        }
        
    except Exception as e:
        logger.error(f"移動角色到位置時發生錯誤: {e}")
        return {
            "success": False,
            "message": f"移動失敗: {str(e)}",
            "estimated_time": 0.0
        }

def move_character_to_coordinates(character_id: str, x: float, y: float, movement_speed: float = 1.0) -> Dict[str, Any]:
    """讓角色移動到指定座標"""
    try:
        estimated_time = 5.0 / movement_speed
        
        # 更新角色狀態
        try:
            from life_simulation_backend import life_simulation_manager
            agent = life_simulation_manager.get_agent_by_persona_id(character_id)
            if agent:
                agent.current_activity = {
                    "name": f"移動到座標 ({x},{y})",
                    "type": "MoveTo",
                    "location": f"{x},{y}",
                    "progress": movement_speed
                }
                agent.is_moving = True
        except ImportError:
            logger.warning("life_simulation_manager 不可用")
        
        logger.info(f"角色 {character_id} 移動到座標 ({x}, {y})")
        
        return {
            "success": True,
            "message": f"角色 {character_id} 正在移動到座標 ({x}, {y})",
            "estimated_time": estimated_time,
            "target_position": {"x": x, "y": y}
        }
        
    except Exception as e:
        logger.error(f"移動角色到座標時發生錯誤: {e}")
        return {
            "success": False,
            "message": f"移動失敗: {str(e)}",
            "estimated_time": 0.0
        }

def get_character_location(character_id: str) -> Dict[str, Any]:
    """獲取角色當前位置"""
    try:
        from life_simulation_backend import life_simulation_manager
        agent = life_simulation_manager.get_agent_by_persona_id(character_id)
        if agent:
            return {
                "success": True,
                "character_id": character_id,
                "current_location": agent.current_location,
                "position": agent.position,
                "is_moving": agent.is_moving
            }
    except ImportError:
        pass
    
    return {
        "success": False,
        "message": f"找不到角色: {character_id}"
    }

def set_character_home(character_id: str, location_id: str) -> Dict[str, Any]:
    """設定角色的家位置"""
    try:
        location = location_manager.get_location(location_id)
        if not location:
            return {
                "success": False,
                "message": f"位置不存在: {location_id}"
            }
        
        location_manager.set_character_preferred_location(character_id, "Home", location_id)
        
        return {
            "success": True,
            "message": f"已設定角色 {character_id} 的家為: {location.name}",
            "location": {
                "id": location.id,
                "name": location.name,
                "position": location.position
            }
        }
        
    except Exception as e:
        logger.error(f"設定角色家位置時發生錯誤: {e}")
        return {
            "success": False,
            "message": f"設定失敗: {str(e)}"
        }

def create_shared_house(character_ids: List[str], location_id: str, house_name: str, position: Dict[str, float]) -> Dict[str, Any]:
    """創建共享房屋"""
    try:
        # 創建新位置
        new_location = LocationInfo(
            id=location_id,
            name=house_name,
            type="Home",
            position=position,
            description=f"共享房屋，住戶: {', '.join(character_ids)}"
        )
        
        location_manager.add_location(new_location)
        location_manager.set_shared_location(character_ids, "Home", location_id)
        
        return {
            "success": True,
            "message": f"創建共享房屋 {house_name}，住戶: {', '.join(character_ids)}",
            "location": {
                "id": location_id,
                "name": house_name,
                "position": position,
                "residents": character_ids
            }
        }
        
    except Exception as e:
        logger.error(f"創建共享房屋時發生錯誤: {e}")
        return {
            "success": False,
            "message": f"創建失敗: {str(e)}"
        }

def get_all_locations_info() -> Dict[str, Any]:
    """獲取所有位置資訊"""
    try:
        locations = []
        for loc in location_manager.locations.values():
            locations.append({
                "id": loc.id,
                "name": loc.name,
                "type": loc.type,
                "position": loc.position,
                "description": loc.description
            })
        
        return {
            "success": True,
            "total_count": len(locations),
            "locations": locations
        }
        
    except Exception as e:
        logger.error(f"獲取位置資訊時發生錯誤: {e}")
        return {
            "success": False,
            "message": f"獲取失敗: {str(e)}",
            "total_count": 0,
            "locations": []
        }

def get_character_memory_info(character_id: str) -> Dict[str, Any]:
    """獲取角色位置記憶資訊"""
    try:
        memory = location_manager.get_character_memory(character_id)
        if not memory:
            return {
                "success": False,
                "message": f"角色 {character_id} 沒有位置記憶"
            }
        
        # 詳細記憶資訊
        memory_details = {}
        for loc_type, loc_id in memory.preferred_locations.items():
            location = location_manager.get_location(loc_id)
            if location:
                memory_details[loc_type] = {
                    "location_id": loc_id,
                    "location_name": location.name,
                    "position": location.position
                }
        
        return {
            "success": True,
            "character_id": character_id,
            "preferred_locations": memory_details,
            "known_locations_count": len(memory.known_locations)
        }
        
    except Exception as e:
        logger.error(f"獲取角色記憶時發生錯誤: {e}")
        return {
            "success": False,
            "message": f"獲取失敗: {str(e)}"
        }

# =======================================================
# Function Calling 定義
# =======================================================

LOCATION_FUNCTION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "move_character_to_location",
            "description": "讓指定的角色移動到一個位置。支援使用位置類型（如 'home', 'school'）或具體位置ID。角色會使用自己記憶中的偏好位置。",
            "parameters": {
                "type": "object",
                "properties": {
                    "character_id": {
                        "type": "string",
                        "description": "要移動的角色的唯一ID (persona_id)。"
                    },
                    "location_id": {
                        "type": "string",
                        "description": "目標位置。可以是位置類型（'home', 'school', 'workplace', 'soccerfield', 'restaurant'）或具體位置ID。"
                    },
                    "movement_speed": {
                        "type": "number",
                        "description": "角色移動的速度倍數 (預設為 1.0)。",
                        "default": 1.0
                    }
                },
                "required": ["character_id", "location_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_character_home",
            "description": "設定指定角色的家位置。這會更新角色的位置記憶。",
            "parameters": {
                "type": "object",
                "properties": {
                    "character_id": {
                        "type": "string",
                        "description": "要設定家位置的角色的唯一ID (persona_id)。"
                    },
                    "location_id": {
                        "type": "string",
                        "description": "要設為家的位置ID。"
                    }
                },
                "required": ["character_id", "location_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_shared_house",
            "description": "創建一個共享房屋，讓多個角色住在同一個地方（如室友、家人）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "character_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要共享房屋的角色ID列表。"
                    },
                    "location_id": {
                        "type": "string",
                        "description": "新房屋的位置ID。"
                    },
                    "house_name": {
                        "type": "string",
                        "description": "房屋的名稱。"
                    },
                    "position": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"}
                        },
                        "description": "房屋在地圖上的位置座標。"
                    }
                },
                "required": ["character_ids", "location_id", "house_name", "position"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "move_character_to_coordinates",
            "description": "讓指定的角色移動到地圖上的特定座標。",
            "parameters": {
                "type": "object",
                "properties": {
                    "character_id": {
                        "type": "string",
                        "description": "要移動的角色的唯一ID (persona_id)。"
                    },
                    "x": {
                        "type": "number",
                        "description": "目標位置的X座標。"
                    },
                    "y": {
                        "type": "number",
                        "description": "目標位置的Y座標。"
                    },
                    "movement_speed": {
                        "type": "number",
                        "description": "角色移動的速度倍數 (預設為 1.0)。",
                        "default": 1.0
                    }
                },
                "required": ["character_id", "x", "y"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_character_location",
            "description": "獲取指定角色的當前位置資訊。",
            "parameters": {
                "type": "object",
                "properties": {
                    "character_id": {
                        "type": "string",
                        "description": "要查詢的角色的唯一ID (persona_id)。"
                    }
                },
                "required": ["character_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_all_locations_info",
            "description": "獲取所有可用位置的詳細資訊，包括ID、名稱、類型和座標。",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_character_memory_info",
            "description": "獲取指定角色的位置記憶資訊，包括偏好位置和認識的所有位置。",
            "parameters": {
                "type": "object",
                "properties": {
                    "character_id": {
                        "type": "string",
                        "description": "要查詢記憶的角色的唯一ID (persona_id)。"
                    }
                },
                "required": ["character_id"]
            }
        }
    }
]

LOCATION_FUNCTION_MAP = {
    "move_character_to_location": move_character_to_location,
    "move_character_to_coordinates": move_character_to_coordinates,
    "get_character_location": get_character_location,
    "set_character_home": set_character_home,
    "create_shared_house": create_shared_house,
    "get_all_locations_info": get_all_locations_info,
    "get_character_memory_info": get_character_memory_info,
}


