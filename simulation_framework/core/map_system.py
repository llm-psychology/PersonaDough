"""
地圖系統和空間關係管理
Map System and Spatial Relationship Management
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from enum import Enum
import json
import networkx as nx
import math
from pathlib import Path

class LocationType(Enum):
    """位置類型"""
    ROOM = "room"                   # 房間
    CORRIDOR = "corridor"           # 走廊
    STAIRWAY = "stairway"          # 樓梯
    ELEVATOR = "elevator"          # 電梯
    EXIT = "exit"                  # 出口
    SAFE_ZONE = "safe_zone"        # 安全區域
    HAZARD_ZONE = "hazard_zone"    # 危險區域
    UTILITY = "utility"            # 公用設施
    OUTDOOR = "outdoor"            # 戶外

class ConnectionType(Enum):
    """連接類型"""
    DOOR = "door"                  # 門
    WINDOW = "window"              # 窗戶
    CORRIDOR = "corridor"          # 走廊
    STAIRWAY = "stairway"          # 樓梯
    ELEVATOR = "elevator"          # 電梯
    EMERGENCY_EXIT = "emergency_exit"  # 緊急出口

@dataclass
class Location:
    """位置"""
    id: str                                 # 位置ID
    name: str                              # 位置名稱
    location_type: LocationType            # 位置類型
    coordinates: Tuple[float, float]       # 座標 (x, y)
    floor: int = 0                         # 樓層
    capacity: int = -1                     # 容量 (-1 表示無限制)
    properties: Dict[str, Any] = field(default_factory=dict)  # 自訂屬性
    
    def distance_to(self, other: 'Location') -> float:
        """計算到另一個位置的距離"""
        return math.sqrt(
            (self.coordinates[0] - other.coordinates[0])**2 + 
            (self.coordinates[1] - other.coordinates[1])**2
        )

@dataclass 
class Connection:
    """連接"""
    from_location: str                     # 起始位置ID
    to_location: str                       # 目標位置ID
    connection_type: ConnectionType        # 連接類型
    is_bidirectional: bool = True          # 是否雙向
    travel_time: float = 1.0               # 移動時間
    capacity: int = -1                     # 通道容量
    properties: Dict[str, Any] = field(default_factory=dict)  # 自訂屬性

@dataclass
class Route:
    """路線"""
    path: List[str]                        # 路徑位置ID列表
    total_distance: float                  # 總距離
    total_time: float                      # 總時間
    is_safe: bool = True                   # 是否安全
    difficulty: float = 1.0                # 難度係數

class SpatialGraph:
    """空間圖形"""
    
    def __init__(self):
        self.locations: Dict[str, Location] = {}
        self.connections: Dict[str, Connection] = {}
        self.graph = nx.Graph()
        
    def add_location(self, location: Location):
        """添加位置"""
        self.locations[location.id] = location
        self.graph.add_node(
            location.id,
            location_type=location.location_type.value,
            coordinates=location.coordinates,
            floor=location.floor,
            capacity=location.capacity,
            **location.properties
        )
    
    def add_connection(self, connection: Connection):
        """添加連接"""
        conn_id = f"{connection.from_location}-{connection.to_location}"
        self.connections[conn_id] = connection
        
        # 添加邊
        self.graph.add_edge(
            connection.from_location,
            connection.to_location,
            connection_type=connection.connection_type.value,
            travel_time=connection.travel_time,
            capacity=connection.capacity,
            weight=connection.travel_time,
            **connection.properties
        )
        
        # 如果是雙向連接，添加反向邊
        if connection.is_bidirectional:
            reverse_conn = Connection(
                from_location=connection.to_location,
                to_location=connection.from_location,
                connection_type=connection.connection_type,
                is_bidirectional=False,  # 避免重複
                travel_time=connection.travel_time,
                capacity=connection.capacity,
                properties=connection.properties.copy()
            )
            reverse_id = f"{reverse_conn.from_location}-{reverse_conn.to_location}"
            self.connections[reverse_id] = reverse_conn
    
    def find_shortest_path(self, start: str, end: str, 
                          weight_attr: str = 'travel_time') -> Optional[Route]:
        """尋找最短路徑"""
        try:
            path = nx.shortest_path(self.graph, start, end, weight=weight_attr)
            total_time = nx.shortest_path_length(self.graph, start, end, weight=weight_attr)
            
            # 計算總距離
            total_distance = 0.0
            for i in range(len(path) - 1):
                loc1 = self.locations[path[i]]
                loc2 = self.locations[path[i + 1]]
                total_distance += loc1.distance_to(loc2)
            
            return Route(
                path=path,
                total_distance=total_distance,
                total_time=total_time
            )
        except nx.NetworkXNoPath:
            return None
    
    def find_all_paths(self, start: str, end: str, 
                      max_length: int = 10) -> List[Route]:
        """尋找所有可能路徑"""
        routes = []
        try:
            for path in nx.all_simple_paths(self.graph, start, end, cutoff=max_length):
                # 計算路徑總時間和距離
                total_time = 0.0
                total_distance = 0.0
                
                for i in range(len(path) - 1):
                    edge_data = self.graph[path[i]][path[i + 1]]
                    total_time += edge_data.get('travel_time', 1.0)
                    
                    loc1 = self.locations[path[i]]
                    loc2 = self.locations[path[i + 1]]
                    total_distance += loc1.distance_to(loc2)
                
                routes.append(Route(
                    path=path,
                    total_distance=total_distance,
                    total_time=total_time
                ))
        except nx.NetworkXNoPath:
            pass
        
        return sorted(routes, key=lambda r: r.total_time)
    
    def get_neighbors(self, location_id: str) -> List[str]:
        """獲取相鄰位置"""
        return list(self.graph.neighbors(location_id))
    
    def get_locations_by_type(self, location_type: LocationType) -> List[Location]:
        """根據類型獲取位置"""
        return [
            loc for loc in self.locations.values() 
            if loc.location_type == location_type
        ]
    
    def get_exits(self) -> List[Location]:
        """獲取所有出口"""
        return self.get_locations_by_type(LocationType.EXIT)
    
    def is_connected(self, start: str, end: str) -> bool:
        """檢查兩個位置是否連通"""
        return nx.has_path(self.graph, start, end)
    
    def export_to_dict(self) -> Dict[str, Any]:
        """匯出為字典"""
        return {
            'locations': {
                loc_id: {
                    'id': loc.id,
                    'name': loc.name,
                    'location_type': loc.location_type.value,
                    'coordinates': loc.coordinates,
                    'floor': loc.floor,
                    'capacity': loc.capacity,
                    'properties': loc.properties
                }
                for loc_id, loc in self.locations.items()
            },
            'connections': {
                conn_id: {
                    'from_location': conn.from_location,
                    'to_location': conn.to_location,
                    'connection_type': conn.connection_type.value,
                    'is_bidirectional': conn.is_bidirectional,
                    'travel_time': conn.travel_time,
                    'capacity': conn.capacity,
                    'properties': conn.properties
                }
                for conn_id, conn in self.connections.items()
                if conn.is_bidirectional or not any(
                    c.from_location == conn.to_location and 
                    c.to_location == conn.from_location and 
                    c.is_bidirectional 
                    for c in self.connections.values()
                )
            }
        }

class MapBuilder:
    """地圖建構器"""
    
    def __init__(self):
        self.spatial_graph = SpatialGraph()
    
    def create_grid_map(self, rows: int, cols: int, 
                       cell_size: float = 1.0) -> SpatialGraph:
        """創建網格地圖"""
        # 創建位置
        for row in range(rows):
            for col in range(cols):
                location_id = f"R{row}C{col}"
                location = Location(
                    id=location_id,
                    name=f"房間 {row}-{col}",
                    location_type=LocationType.ROOM,
                    coordinates=(col * cell_size, row * cell_size)
                )
                self.spatial_graph.add_location(location)
        
        # 創建連接
        for row in range(rows):
            for col in range(cols):
                current_id = f"R{row}C{col}"
                
                # 向右連接
                if col < cols - 1:
                    right_id = f"R{row}C{col + 1}"
                    connection = Connection(
                        from_location=current_id,
                        to_location=right_id,
                        connection_type=ConnectionType.DOOR
                    )
                    self.spatial_graph.add_connection(connection)
                
                # 向下連接
                if row < rows - 1:
                    down_id = f"R{row + 1}C{col}"
                    connection = Connection(
                        from_location=current_id,
                        to_location=down_id,
                        connection_type=ConnectionType.DOOR
                    )
                    self.spatial_graph.add_connection(connection)
        
        return self.spatial_graph
    
    def load_from_file(self, filepath: str) -> SpatialGraph:
        """從檔案載入地圖"""
        path = Path(filepath)
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 載入位置
        for loc_data in data.get('locations', {}).values():
            location = Location(
                id=loc_data['id'],
                name=loc_data['name'],
                location_type=LocationType(loc_data['location_type']),
                coordinates=tuple(loc_data['coordinates']),
                floor=loc_data.get('floor', 0),
                capacity=loc_data.get('capacity', -1),
                properties=loc_data.get('properties', {})
            )
            self.spatial_graph.add_location(location)
        
        # 載入連接
        for conn_data in data.get('connections', {}).values():
            connection = Connection(
                from_location=conn_data['from_location'],
                to_location=conn_data['to_location'],
                connection_type=ConnectionType(conn_data['connection_type']),
                is_bidirectional=conn_data.get('is_bidirectional', True),
                travel_time=conn_data.get('travel_time', 1.0),
                capacity=conn_data.get('capacity', -1),
                properties=conn_data.get('properties', {})
            )
            self.spatial_graph.add_connection(connection)
        
        return self.spatial_graph
    
    def save_to_file(self, filepath: str):
        """儲存地圖到檔案"""
        data = self.spatial_graph.export_to_dict()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def add_exits(self, exit_locations: List[Tuple[str, str]]):
        """添加出口"""
        for exit_id, exit_name in exit_locations:
            if exit_id in self.spatial_graph.locations:
                location = self.spatial_graph.locations[exit_id]
                location.location_type = LocationType.EXIT
                location.name = exit_name
    
    def set_hazard_zones(self, hazard_locations: List[str]):
        """設定危險區域"""
        for location_id in hazard_locations:
            if location_id in self.spatial_graph.locations:
                location = self.spatial_graph.locations[location_id]
                location.location_type = LocationType.HAZARD_ZONE

# 預設地圖模板
class MapTemplates:
    """地圖模板"""
    
    @staticmethod
    def create_simple_building() -> SpatialGraph:
        """創建簡單建築物地圖"""
        builder = MapBuilder()
        
        # 創建房間
        rooms = [
            ("A1", "會議室A", (0, 0)),
            ("A2", "辦公室A", (1, 0)),
            ("A3", "茶水間", (2, 0)),
            ("B1", "辦公室B", (0, 1)),
            ("B2", "會議室B", (1, 1)),
            ("B3", "儲藏室", (2, 1)),
            ("CORRIDOR", "走廊", (1, 0.5)),
            ("EXIT1", "主要出口", (3, 0.5)),
            ("EXIT2", "緊急出口", (-1, 0.5))
        ]
        
        for room_id, room_name, coordinates in rooms:
            location_type = LocationType.EXIT if "EXIT" in room_id else \
                           LocationType.CORRIDOR if room_id == "CORRIDOR" else \
                           LocationType.ROOM
            
            location = Location(
                id=room_id,
                name=room_name,
                location_type=location_type,
                coordinates=coordinates
            )
            builder.spatial_graph.add_location(location)
        
        # 創建連接
        connections = [
            ("A1", "CORRIDOR"), ("A2", "CORRIDOR"), ("A3", "CORRIDOR"),
            ("B1", "CORRIDOR"), ("B2", "CORRIDOR"), ("B3", "CORRIDOR"),
            ("CORRIDOR", "EXIT1"), ("CORRIDOR", "EXIT2")
        ]
        
        for from_loc, to_loc in connections:
            connection = Connection(
                from_location=from_loc,
                to_location=to_loc,
                connection_type=ConnectionType.DOOR
            )
            builder.spatial_graph.add_connection(connection)
        
        return builder.spatial_graph 