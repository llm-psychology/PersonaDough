#!/usr/bin/env python3
"""
Unity記憶管理API測試腳本

測試後端記憶管理功能是否正常運作
"""

import requests
import json
import sys
import time

# 配置
BASE_URL = "http://localhost:20000"

def test_add_memory():
    """測試添加記憶功能"""
    print("=== 測試添加記憶功能 ===")
    
    url = f"{BASE_URL}/unity-api/memory/add"
    data = {
        "persona1_id": "test_persona_1",
        "persona1_name": "測試角色一",
        "persona2_id": "test_persona_2", 
        "persona2_name": "測試角色二",
        "speaker": "測試角色一",
        "content": "你好，很高興認識你！",
        "round": "1"
    }
    
    try:
        response = requests.post(url, json=data)
        print(f"狀態碼: {response.status_code}")
        print(f"回應: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"錯誤: {e}")
        return False

def test_get_memory_context():
    """測試獲取記憶上下文功能"""
    print("\n=== 測試獲取記憶上下文功能 ===")
    
    url = f"{BASE_URL}/unity-api/memory/context"
    data = {
        "persona1_id": "test_persona_1",
        "persona2_id": "test_persona_2"
    }
    
    try:
        response = requests.post(url, json=data)
        print(f"狀態碼: {response.status_code}")
        result = response.json()
        print(f"回應: {result}")
        print(f"記憶上下文: {result.get('context', '無')}")
        return response.status_code == 200
    except Exception as e:
        print(f"錯誤: {e}")
        return False

def test_memory_stats():
    """測試記憶統計功能"""
    print("\n=== 測試記憶統計功能 ===")
    
    url = f"{BASE_URL}/unity-api/memory/stats"
    
    try:
        response = requests.get(url)
        print(f"狀態碼: {response.status_code}")
        result = response.json()
        print(f"統計結果: {result}")
        return response.status_code == 200
    except Exception as e:
        print(f"錯誤: {e}")
        return False

def test_clear_memory():
    """測試清除記憶功能"""
    print("\n=== 測試清除記憶功能 ===")
    
    url = f"{BASE_URL}/unity-api/memory/clear/test_persona_1/test_persona_2"
    
    try:
        response = requests.delete(url)
        print(f"狀態碼: {response.status_code}")
        print(f"回應: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"錯誤: {e}")
        return False

def main():
    """主測試函數"""
    print("Unity記憶管理API測試開始")
    print("=" * 50)
    
    # 檢查後端是否運行
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"後端狀態: 正常運行 (狀態碼: {response.status_code})")
    except Exception as e:
        print(f"後端連接失敗: {e}")
        print("請確保後端服務正在運行 (python backend.py)")
        sys.exit(1)
    
    print()
    
    # 執行測試
    tests = [
        ("添加記憶", test_add_memory),
        ("添加第二條記憶", lambda: test_add_second_memory()),
        ("獲取記憶上下文", test_get_memory_context),
        ("記憶統計", test_memory_stats),
        ("清除記憶", test_clear_memory),
        ("驗證清除後的狀態", test_get_memory_context)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n執行測試: {test_name}")
        success = test_func()
        results.append((test_name, success))
        time.sleep(0.5)  # 短暫延遲
    
    # 測試結果總結
    print("\n" + "=" * 50)
    print("測試結果總結:")
    for test_name, success in results:
        status = "✓ 通過" if success else "✗ 失敗"
        print(f"{test_name}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(1 for _, success in results if success)
    print(f"\n總計: {passed_tests}/{total_tests} 個測試通過")

def test_add_second_memory():
    """添加第二條記憶用於測試"""
    url = f"{BASE_URL}/unity-api/memory/add"
    data = {
        "persona1_id": "test_persona_1",
        "persona1_name": "測試角色一",
        "persona2_id": "test_persona_2", 
        "persona2_name": "測試角色二",
        "speaker": "測試角色二",
        "content": "我也很高興認識你，你看起來很友善！",
        "round": "2"
    }
    
    try:
        response = requests.post(url, json=data)
        print(f"狀態碼: {response.status_code}")
        print(f"回應: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"錯誤: {e}")
        return False

if __name__ == "__main__":
    main()
