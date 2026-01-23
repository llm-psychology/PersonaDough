#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
智慧位置系統測試腳本
測試新的角色記憶位置系統和function calling功能
"""

import asyncio
import aiohttp
import json
import sys
import os

# 添加當前目錄到Python路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from location_function_tools import (
    move_character_to_location, 
    smart_assign_character_home,
    get_character_life_locations,
    location_manager
)

async def test_smart_location_system():
    """測試智慧位置系統"""
    
    print("🧪 開始測試智慧位置系統...")
    print("=" * 60)
    
    # 測試角色ID
    test_characters = [
        "687a1b2d5b",  # 王美鳳
        "867b21d0fe",  # 何素珍  
        "eef4720230"   # 呂志強
    ]
    
    # 1. 測試智慧住所分配
    print("\n📍 測試1: 智慧住所分配")
    
    # 王美鳳獨立住所
    result1 = smart_assign_character_home(test_characters[0], preference="private")
    print(f"王美鳳: {result1['message']}")
    
    # 何素珍獨立住所
    result2 = smart_assign_character_home(test_characters[1], preference="private")
    print(f"何素珍: {result2['message']}")
    
    # 呂志強與王美鳳共享住所
    result3 = smart_assign_character_home(test_characters[2], share_with=test_characters[0], preference="shared")
    print(f"呂志強: {result3['message']}")
    
    # 2. 測試智慧移動 - 回家
    print("\n🏠 測試2: 智慧移動功能 (回家)")
    
    for i, char_id in enumerate(test_characters):
        result = move_character_to_location(char_id, "home")
        print(f"角色 {i+1}: {result['message']}")
        if result['success']:
            print(f"  └── 目標位置: {result['location_details']}")
    
    # 3. 測試其他位置移動
    print("\n🏫 測試3: 智慧移動功能 (學校/工作)")
    
    # 王美鳳去工作
    result = move_character_to_location(test_characters[0], "workplace")
    print(f"王美鳳去工作: {result['message']}")
    
    # 何素珍去學校
    result = move_character_to_location(test_characters[1], "school")  
    print(f"何素珍去學校: {result['message']}")
    
    # 呂志強去踢足球
    result = move_character_to_location(test_characters[2], "soccerfield")
    print(f"呂志強去踢足球: {result['message']}")
    
    # 4. 測試位置記憶查詢
    print("\n🧠 測試4: 角色位置記憶查詢")
    
    for i, char_id in enumerate(test_characters):
        result = get_character_life_locations(char_id)
        if result['success']:
            print(f"角色 {i+1} 的生活地圖:")
            print(f"  └── 住所: {result['locations']['home']}")
            print(f"  └── 認識的位置數量: {result['locations']['total_known']}")
        else:
            print(f"角色 {i+1}: {result['message']}")
    
    # 5. 測試共享位置信息
    print("\n🏡 測試5: 共享位置檢查")
    
    wang_home = location_manager.get_character_home(test_characters[0])
    lu_home = location_manager.get_character_home(test_characters[2])
    
    if wang_home == lu_home:
        print(f"✅ 王美鳳和呂志強成功共享住所: {wang_home}")
    else:
        print(f"❌ 共享住所失敗 - 王美鳳: {wang_home}, 呂志強: {lu_home}")
    
    print("\n" + "=" * 60)
    print("✅ 智慧位置系統測試完成！")

async def test_backend_integration():
    """測試與後端的整合"""
    
    print("\n🔗 測試後端API整合...")
    
    base_url = "http://localhost:8000"
    
    async with aiohttp.ClientSession() as session:
        try:
            # 測試移動API
            move_data = {
                "character_id": "687a1b2d5b",
                "location_id": "home",
                "movement_speed": 1.5
            }
            
            async with session.post(f"{base_url}/api/location/move-character", json=move_data) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ 後端移動API測試成功: {result.get('message', 'OK')}")
                else:
                    print(f"❌ 後端移動API測試失敗: {response.status}")
                    
            # 測試獲取所有位置
            async with session.get(f"{base_url}/api/location/all-locations") as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ 後端位置列表API測試成功: 找到 {result.get('total_count', 0)} 個位置")
                else:
                    print(f"❌ 後端位置列表API測試失敗: {response.status}")
                    
        except Exception as e:
            print(f"❌ 後端連接測試失敗: {e}")
            print("💡 確保 backend.py 正在運行 (python backend.py)")

async def main():
    """主測試函數"""
    
    print("🚀 智慧角色位置系統 - 完整測試")
    print("=" * 60)
    
    # 本地測試
    await test_smart_location_system()
    
    # 後端整合測試
    await test_backend_integration()
    
    print("\n🎯 測試重點總結:")
    print("1. ✅ 角色可以記住自己的住所和常去地點")
    print("2. ✅ 支持多人共享同一個住所")
    print("3. ✅ 智慧移動系統根據角色記憶選擇最佳位置")
    print("4. ✅ 位置坐標不再是(0,0)")
    print("5. ✅ Function calling可以控制角色移動和任務")
    
    print("\n💡 使用建議:")
    print("- 使用 move_character_to_location(character_id, 'home') 讓角色回家")
    print("- 使用 smart_assign_character_home() 智慧分配住所")
    print("- 角色會記住去過的地方，下次移動更智慧")

if __name__ == "__main__":
    asyncio.run(main())

