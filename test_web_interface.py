#!/usr/bin/env python3
"""
測試網頁行程管理界面
"""

import asyncio
import aiohttp
import json

async def test_web_interface():
    """測試網頁界面功能"""
    
    base_url = "http://localhost:20000"
    
    print("=== 測試網頁行程管理界面 ===\n")
    
    try:
        async with aiohttp.ClientSession() as session:
            
            # 1. 測試網頁是否可訪問
            print("1. 測試網頁界面...")
            try:
                async with session.get(f"{base_url}/schedule-control") as response:
                    if response.status == 200:
                        print("   ✓ 網頁界面可正常訪問")
                    else:
                        print(f"   ✗ 網頁界面訪問失敗: {response.status}")
                        return
            except Exception as e:
                print(f"   ✗ 網頁界面訪問失敗: {e}")
                return
            
            # 2. 啟動生活模擬
            print("\n2. 啟動生活模擬...")
            async with session.get(f"{base_url}/api/life-simulation/auto-start") as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"   ✓ 生活模擬啟動成功: {result['message']}")
                else:
                    print(f"   ✗ 生活模擬啟動失敗: {response.status}")
                    return
            
            # 3. 測試獲取角色狀態
            print("\n3. 測試獲取角色狀態...")
            async with session.get(f"{base_url}/api/life-simulation/all-agents-status") as response:
                if response.status == 200:
                    result = await response.json()
                    agents = result.get('agents', [])
                    print(f"   ✓ 成功獲取 {len(agents)} 個角色狀態")
                    for agent in agents:
                        print(f"     - {agent.get('name', 'Unknown')} (ID: {agent.get('agent_id', 'Unknown')})")
                else:
                    print(f"   ✗ 獲取角色狀態失敗: {response.status}")
                    return
            
            # 4. 測試獲取行程資訊
            print("\n4. 測試獲取行程資訊...")
            async with session.get(f"{base_url}/api/schedule/all") as response:
                if response.status == 200:
                    result = await response.json()
                    schedules = result.get('schedules', {})
                    print(f"   ✓ 成功獲取行程資訊: {len(schedules)} 個角色有行程")
                else:
                    print(f"   ✗ 獲取行程資訊失敗: {response.status}")
                    return
            
            # 5. 測試安排行程
            print("\n5. 測試安排學生行程...")
            async with session.post(f"{base_url}/api/schedule/apply-to-all?schedule_type=student") as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"   ✓ 成功安排行程: {result['message']}")
                    print(f"     成功安排: {result['success_count']}/{result['total_agents']} 個角色")
                else:
                    print(f"   ✗ 安排行程失敗: {response.status}")
                    error_text = await response.text()
                    print(f"     錯誤: {error_text}")
            
            # 6. 測試獲取行程模板
            print("\n6. 測試獲取行程模板...")
            async with session.get(f"{base_url}/api/schedule/templates") as response:
                if response.status == 200:
                    result = await response.json()
                    templates = result.get('templates', {})
                    print(f"   ✓ 成功獲取 {len(templates)} 個行程模板")
                    for template_type, tasks in templates.items():
                        print(f"     - {template_type}: {len(tasks)} 個任務")
                else:
                    print(f"   ✗ 獲取行程模板失敗: {response.status}")
            
            # 7. 測試切換行程狀態
            if agents:
                first_agent_id = agents[0].get('agent_id')
                if first_agent_id:
                    print(f"\n7. 測試切換角色 {first_agent_id} 的行程狀態...")
                    async with session.post(f"{base_url}/api/schedule/toggle/{first_agent_id}") as response:
                        if response.status == 200:
                            result = await response.json()
                            print(f"   ✓ 成功切換行程狀態: {result['message']}")
                        else:
                            print(f"   ✗ 切換行程狀態失敗: {response.status}")
            
            # 8. 測試清除所有行程
            print("\n8. 測試清除所有行程...")
            async with session.delete(f"{base_url}/api/schedule/clear-all") as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"   ✓ 成功清除所有行程: {result['message']}")
                else:
                    print(f"   ✗ 清除行程失敗: {response.status}")
            
            print("\n=== 網頁界面測試完成 ===")
            print("\n🌐 現在可以在瀏覽器中訪問以下網址:")
            print(f"   {base_url}/schedule-control")
            print("\n📋 網頁功能:")
            print("   - 一鍵安排所有角色行程")
            print("   - 個別管理每個角色")
            print("   - 即時查看角色狀態")
            print("   - 控制模擬系統")
            
    except Exception as e:
        print(f"測試過程中發生錯誤: {e}")

if __name__ == "__main__":
    asyncio.run(test_web_interface())


