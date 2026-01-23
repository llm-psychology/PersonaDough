#!/usr/bin/env python3
"""
測試每日行程安排 API 的腳本
"""

import asyncio
import aiohttp
import json

async def test_schedule_api():
    """測試行程安排 API"""
    
    base_url = "http://localhost:20000"
    
    print("=== 測試每日行程安排 API ===\n")
    
    try:
        async with aiohttp.ClientSession() as session:
            
            # 1. 啟動生活模擬
            print("1. 啟動生活模擬...")
            async with session.get(f"{base_url}/api/life-simulation/auto-start") as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"   ✓ 生活模擬啟動成功: {result['message']}")
                else:
                    print(f"   ✗ 生活模擬啟動失敗: {response.status}")
                    return
            
            # 2. 獲取所有代理人狀態
            print("\n2. 獲取代理人狀態...")
            async with session.get(f"{base_url}/api/life-simulation/all-agents-status") as response:
                if response.status == 200:
                    result = await response.json()
                    agents = result.get('agents', [])
                    print(f"   ✓ 找到 {len(agents)} 個代理人")
                    for agent in agents:
                        print(f"     - {agent.get('name', 'Unknown')} (ID: {agent.get('agent_id', 'Unknown')})")
                else:
                    print(f"   ✗ 獲取代理人狀態失敗: {response.status}")
                    return
            
            # 3. 獲取行程模板
            print("\n3. 獲取行程模板...")
            async with session.get(f"{base_url}/api/schedule/templates") as response:
                if response.status == 200:
                    result = await response.json()
                    templates = result.get('templates', {})
                    print(f"   ✓ 可用行程模板: {list(templates.keys())}")
                    for template_type, tasks in templates.items():
                        print(f"     - {template_type}: {len(tasks)} 個任務")
                else:
                    print(f"   ✗ 獲取行程模板失敗: {response.status}")
            
            # 4. 為所有代理人安排學生行程
            print("\n4. 為所有代理人安排學生行程...")
            async with session.post(f"{base_url}/api/schedule/apply-to-all?schedule_type=student") as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"   ✓ {result['message']}")
                    print(f"     成功安排: {result['success_count']}/{result['total_agents']} 個代理人")
                else:
                    print(f"   ✗ 安排學生行程失敗: {response.status}")
                    error_text = await response.text()
                    print(f"     錯誤: {error_text}")
            
            # 5. 獲取所有行程安排
            print("\n5. 獲取所有行程安排...")
            async with session.get(f"{base_url}/api/schedule/all") as response:
                if response.status == 200:
                    result = await response.json()
                    schedules = result.get('schedules', {})
                    print(f"   ✓ 總代理人: {result['total_agents']}, 有行程: {result['scheduled_agents']}")
                    
                    for agent_id, schedule_info in schedules.items():
                        print(f"     - 代理人 {agent_id}:")
                        print(f"       行程ID: {schedule_info['schedule_id']}")
                        print(f"       狀態: {'啟用' if schedule_info['is_active'] else '停用'}")
                        print(f"       任務數: {len(schedule_info['tasks'])}")
                        
                        if schedule_info.get('current_task'):
                            current_task = schedule_info['current_task']
                            print(f"       當前任務: {current_task['name']} ({current_task['start_time']})")
                        else:
                            print(f"       當前任務: 無")
                else:
                    print(f"   ✗ 獲取行程安排失敗: {response.status}")
            
            # 6. 等待一段時間讓模擬運行
            print("\n6. 等待模擬運行 5 秒...")
            await asyncio.sleep(5)
            
            # 7. 再次檢查代理人狀態
            print("\n7. 檢查代理人狀態變化...")
            async with session.get(f"{base_url}/api/life-simulation/all-agents-status") as response:
                if response.status == 200:
                    result = await response.json()
                    agents = result.get('agents', [])
                    print(f"   ✓ 代理人狀態更新:")
                    for agent in agents:
                        current_activity = agent.get('current_activity', {})
                        location = agent.get('current_location', 'Unknown')
                        print(f"     - {agent.get('name', 'Unknown')}: {current_activity.get('name', 'Unknown')} @ {location}")
                else:
                    print(f"   ✗ 獲取代理人狀態失敗: {response.status}")
            
            # 8. 測試切換行程狀態
            if agents:
                first_agent_id = agents[0].get('agent_id')
                if first_agent_id:
                    print(f"\n8. 切換代理人 {first_agent_id} 的行程狀態...")
                    async with session.post(f"{base_url}/api/schedule/toggle/{first_agent_id}") as response:
                        if response.status == 200:
                            result = await response.json()
                            print(f"   ✓ {result['message']}")
                        else:
                            print(f"   ✗ 切換行程狀態失敗: {response.status}")
            
            # 9. 清除所有行程
            print("\n9. 清除所有行程安排...")
            async with session.delete(f"{base_url}/api/schedule/clear-all") as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"   ✓ {result['message']}")
                else:
                    print(f"   ✗ 清除行程失敗: {response.status}")
            
            print("\n=== 測試完成 ===")
            
    except Exception as e:
        print(f"測試過程中發生錯誤: {e}")

if __name__ == "__main__":
    asyncio.run(test_schedule_api())


