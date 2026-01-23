# Backend API 完整解釋文件

## 系統概述

這是一個基於 FastAPI 的人格代理人模擬系統，整合了多個核心功能模組，提供完整的人格生成、對話模擬、生活模擬和場景模擬功能。

### 核心架構

```
persona-dough/
├── backend.py                    # 主要 API 服務器
├── life_simulation_backend.py    # 生活模擬後端
├── llm_scheduler.py              # LLM 智能排程系統
├── location_function_tools.py    # 位置管理工具
├── scenario_simulator.py         # 場景模擬器
├── module/                       # 核心功能模組
│   ├── LLM_responder.py         # LLM 回應處理
│   ├── PERSONA_loader.py        # 人格資料載入
│   ├── QA_loader.py             # 問答資料載入
│   └── prompt_loader.py         # 提示詞載入
├── interviewer/                  # 面試官系統
├── humanoid/                     # 人形代理人生成
└── simulation_framework/         # 模擬框架
    └── core/                     # 核心模擬組件
```

## API 端點分類

### 1. 前端頁面服務

| 端點 | 方法 | 描述 |
|------|------|------|
| `/` | GET | 主要前端頁面 |
| `/unity-app` | GET | Unity WebGL 應用重定向 |
| `/fire_simulation` | GET | 火災模擬頁面 |
| `/fire_simulation_sse` | GET | 火災模擬 SSE 頁面 |
| `/schedule-control` | GET | 排程控制頁面 |
| `/llm-schedule-control` | GET | LLM 排程控制頁面 |

### 2. LLM 智能排程 API

#### 2.1 活動生成

**POST** `/api/llm-schedule/generate-activity`

使用 LLM 為指定人格生成個性化活動。

**請求參數:**
```json
{
  "persona_id": "string",           // 必需：人格ID
  "time_slot": "string",            // 時間段，預設："現在"
  "duration_minutes": 60,           // 活動持續時間（分鐘）
  "constraints": ["string"],        // 限制條件列表
  "weather": "string",              // 天氣狀況，預設："晴朗"
  "season": "string",               // 季節，預設："春季"
  "other_characters": ["string"],   // 其他角色ID列表
  "special_events": ["string"]      // 特殊事件列表
}
```

**回應格式:**
```json
{
  "success": true,
  "activity": {
    "name": "string",                    // 活動名稱
    "activity_type": "string",           // 活動類型代碼
    "location": "string",                // 地點代碼
    "duration": 60,                      // 持續時間
    "priority": 0.8,                     // 優先級 (0.0-1.0)
    "reasoning": "string",               // 選擇理由
    "social_participants": ["string"],   // 社交參與者
    "required_items": ["string"]         // 需要物品
  }
}
```

#### 2.2 日程生成

**POST** `/api/llm-schedule/generate-daily`

為指定人格生成一整天的個性化排程。

**請求參數:**
```json
{
  "persona_id": "string",    // 必需：人格ID
  "date": "YYYY-MM-DD"       // 日期，可選
}
```

**回應格式:**
```json
{
  "success": true,
  "persona_id": "string",
  "date": "YYYY-MM-DD",
  "total_activities": 9,
  "schedule": [
    {
      "name": "string",
      "activity_type": "string",
      "location": "string",
      "start_time": "HH:MM",
      "end_time": "HH:MM",
      "duration": 60,
      "priority": 0.8,
      "reasoning": "string",
      "social_participants": ["string"],
      "required_items": ["string"]
    }
  ]
}
```

#### 2.3 排程應用

**POST** `/api/llm-schedule/apply-to-agent`

將 LLM 生成的排程應用到生活模擬系統中的代理人。

**請求參數:**
```json
{
  "persona_id": "string",    // 必需：人格ID
  "date": "YYYY-MM-DD"       // 日期，可選
}
```

#### 2.4 人格檔案

**GET** `/api/llm-schedule/personality/{persona_id}`

獲取指定人格的詳細個性檔案。

**回應格式:**
```json
{
  "success": true,
  "profile": {
    "persona_id": "string",
    "name": "string",
    "age": 25,
    "occupation": "string",
    "personality_traits": ["string"],
    "interests": ["string"],
    "daily_preferences": {
      "wake_time": "07:00",
      "sleep_time": "23:00",
      "meal_times": ["07:30", "12:00", "18:30"]
    },
    "energy_pattern": "morning|evening|flexible",
    "social_level": "introvert|extrovert|ambivert"
  }
}
```

### 3. 人格代理人管理 API

#### 3.1 人格生成

**POST** `/api/generate_humanoid`

生成新的人格代理人。

**請求參數:**
```json
{
  "count": 1,                    // 生成數量，預設：1
  "generate_photo": true,        // 是否生成照片，預設：true
  "custom_requirements": "string" // 自訂需求
}
```

#### 3.2 人格查詢

**GET** `/api/humanoids`

獲取所有人格代理人列表。

**查詢參數:**
- `age_min`: 最小年齡
- `age_max`: 最大年齡  
- `gender`: 性別
- `birthplace`: 出生地
- `traits`: 人格特質（逗號分隔）

#### 3.3 人格詳情

**GET** `/api/humanoids/{humanoid_id}`

獲取指定人格代理人的詳細資訊。

#### 3.4 人格照片

**GET** `/api/humanoids/{humanoid_id}/photo`

獲取指定人格代理人的照片。

#### 3.5 人格刪除

**DELETE** `/api/humanoids/{humanoid_id}`

刪除指定的人格代理人。

### 4. 對話模擬 API

#### 4.1 單人面試

**POST** `/api/interview`

與指定人格進行面試對話。

**請求參數:**
```json
{
  "persona_id": "string",              // 人格ID，可選
  "num_rounds": 10,                    // 對話輪數
  "static_interview_data": "string",   // 用戶問題
  "session_id": "string"               // 會話ID，可選
}
```

#### 4.2 雙人對話

**POST** `/api/persona_dialogue`

兩個人格之間的對話模擬。

**請求參數:**
```json
{
  "persona1_id": "string",       // 第一個人格ID
  "persona2_id": "string",       // 第二個人格ID
  "num_rounds": 5,               // 對話輪數
  "initial_topic": "string",     // 初始話題
  "session_id": "string",        // 會話ID，可選
  "memory_context": "string",    // 記憶上下文
  "is_unity_dialogue": false     // 是否為Unity對話
}
```

#### 4.3 群組對話

**POST** `/api/group_dialogue`

多個人格的群組對話模擬。

**請求參數:**
```json
{
  "persona_ids": ["string"],     // 參與者人格ID列表
  "num_rounds": 5,               // 對話輪數
  "initial_topic": "string",     // 初始話題
  "session_id": "string"         // 會話ID，可選
}
```

### 5. Unity 記憶管理 API

#### 5.1 記憶儲存

**POST** `/api/unity/memory`

儲存 Unity 對話記憶。

**請求參數:**
```json
{
  "persona1_id": "string",
  "persona1_name": "string", 
  "persona2_id": "string",
  "persona2_name": "string",
  "speaker": "string",
  "content": "string",
  "round": "string"
}
```

#### 5.2 記憶查詢

**POST** `/api/unity/memory/context`

查詢兩個人格之間的對話記憶。

**請求參數:**
```json
{
  "persona1_id": "string",
  "persona2_id": "string"
}
```

### 6. 生活模擬 API

#### 6.1 模擬啟動

**POST** `/api/life-simulation/start`

啟動生活模擬。

**請求參數:**
```json
{
  "agent_count": 3,                    // 代理人數量
  "time_speed": 60.0,                  // 時間加速倍率
  "simulation_mode": "autonomous"       // 模擬模式
}
```

#### 6.2 代理人狀態

**GET** `/api/life-simulation/agents`

獲取所有代理人狀態。

**GET** `/api/life-simulation/agents/{agent_id}`

獲取指定代理人狀態。

#### 6.3 代理人控制

**POST** `/api/life-simulation/agents/{agent_id}/command`

向代理人發送指令。

**請求參數:**
```json
{
  "command": "move_to|start_activity",
  "params": {
    "location": "string",
    "speed": 1.0
  }
}
```

#### 6.4 排程管理

**POST** `/api/life-simulation/schedule/create`

為代理人創建排程。

**POST** `/api/life-simulation/schedule/apply`

應用排程到代理人。

**GET** `/api/life-simulation/schedule/{agent_id}`

獲取代理人排程。

### 7. 位置管理 API

#### 7.1 位置查詢

**GET** `/api/locations`

獲取所有位置資訊。

**GET** `/api/locations/{location_id}`

獲取指定位置詳情。

#### 7.2 角色位置

**GET** `/api/characters/{character_id}/location`

獲取角色當前位置。

**POST** `/api/characters/{character_id}/move`

移動角色到指定位置。

**請求參數:**
```json
{
  "location_id": "string",       // 目標位置ID或類型
  "movement_speed": 1.0          // 移動速度
}
```

#### 7.3 住所管理

**POST** `/api/characters/{character_id}/home`

設定角色住所。

**POST** `/api/locations/shared-house`

創建共享房屋。

### 8. 火災模擬 API

#### 8.1 模擬控制

**POST** `/api/fire-simulation/start`

啟動火災模擬。

**POST** `/api/fire-simulation/stop/{simulation_id}`

停止火災模擬。

#### 8.2 模擬狀態

**GET** `/api/fire-simulation/status/{simulation_id}`

獲取模擬狀態。

**GET** `/api/fire-simulation/events/{simulation_id}`

獲取模擬事件流（SSE）。

#### 8.3 批次模擬

**POST** `/api/fire-simulation/batch`

執行批次火災模擬。

### 9. 系統管理 API

#### 9.1 系統狀態

**GET** `/api/system/status`

獲取系統狀態。

**GET** `/api/system/health`

系統健康檢查。

#### 9.2 資料管理

**GET** `/api/system/backup`

備份系統資料。

**POST** `/api/system/restore`

恢復系統資料。

## 核心模組說明

### 1. LLM_responder 模組

負責與大型語言模型的交互，支援 OpenAI 和 Ollama 兩種後端。

**主要功能:**
- 文字生成和對話
- 圖片生成
- 文字嵌入生成
- 多模型支援

**關鍵方法:**
- `full_chat_gpt_4o()`: GPT-4o 對話
- `full_chat_gpt_41()`: GPT-4.1 對話  
- `photo_generate()`: 生成人格照片
- `generate_embedding()`: 生成文字嵌入

### 2. PERSONA_loader 模組

管理人格資料的載入、儲存和查詢。

**主要功能:**
- 異步載入人格資料
- 人格資料 CRUD 操作
- 人格搜尋和統計
- 資料備份和恢復

**關鍵方法:**
- `get_all_personas()`: 獲取所有人格
- `get_persona_by_id()`: 根據ID獲取人格
- `add_persona()`: 添加新人格
- `search_personas()`: 搜尋人格

### 3. 生活模擬系統

提供完整的代理人生活模擬功能。

**核心組件:**
- `LifeSimulationManager`: 主要管理器
- `AgentStatus`: 代理人狀態
- `DailySchedule`: 日程安排
- `ScheduledTask`: 排程任務

**模擬功能:**
- 自主決策系統
- 需求管理（飢餓、能量、快樂度）
- 活動執行和位置移動
- 排程管理和時間系統

### 4. LLM 智能排程系統

基於人格特徵的智能活動安排系統。

**核心組件:**
- `LLMScheduler`: 主要排程器
- `PersonalityProfile`: 人格檔案
- `ActivityRequest`: 活動請求
- `GeneratedActivity`: 生成的活動

**智能功能:**
- 基於人格的活動推薦
- 時間和環境感知
- 社交需求考量
- 個性化理由生成

### 5. 位置管理系統

智能位置管理和角色移動系統。

**核心組件:**
- `SmartLocationManager`: 智能位置管理器
- `LocationInfo`: 位置資訊
- `CharacterLocationMemory`: 角色位置記憶

**智能功能:**
- 角色偏好位置記憶
- 智能路徑規劃
- 共享位置管理
- 位置類型推薦

### 6. 場景模擬框架

通用的場景模擬框架，支援多種緊急情況模擬。

**核心組件:**
- `BaseScenario`: 場景基礎類
- `EventManager`: 事件管理器
- `SpatialGraph`: 空間圖形系統
- `TriggerCondition`: 觸發條件系統

**模擬類型:**
- 火災模擬
- 地震模擬
- 水災模擬
- 自訂場景

## 資料模型

### 人格資料結構

```json
{
  "基本資料": {
    "id": "string",
    "姓名": "string", 
    "性別": "string",
    "年紀": 25,
    "出生地": "string"
  },
  "人格屬性": {
    "人格特質": ["string"],
    "社交能力": ["string"],
    "能力屬性": ["string"]
  },
  "背景故事": "string",
  "說話風格": "string",
  "行為模式": "string"
}
```

### 代理人狀態結構

```json
{
  "agent_id": "string",
  "persona_id": "string", 
  "name": "string",
  "position": {"x": 0.0, "y": 0.0},
  "current_location": "string",
  "current_activity": {
    "name": "string",
    "type": "string",
    "location": "string", 
    "progress": 0.0
  },
  "health": 100.0,
  "energy": 80.0,
  "happiness": 50.0,
  "social_needs": 50.0,
  "hunger": 20.0,
  "is_moving": false,
  "last_update": "ISO_timestamp"
}
```

### 排程任務結構

```json
{
  "name": "string",
  "activity_type": "string",
  "location": "string",
  "start_time": "HH:MM",
  "duration": 60,
  "priority": 1
}
```

## 錯誤處理

所有 API 端點都遵循統一的錯誤回應格式：

```json
{
  "success": false,
  "error": "error_code",
  "message": "詳細錯誤訊息",
  "details": {}
}
```

**常見錯誤碼:**
- `400`: 請求參數錯誤
- `404`: 資源不存在
- `500`: 內部服務器錯誤
- `503`: 服務暫時不可用

## 配置說明

### 環境變數

```bash
OPENAI_API_KEY=your_openai_api_key
OLLAMA_BASE_URL=http://localhost:11434
DATABASE_PATH=humanoid/humanoid_database
CONVERSATION_HISTORY_PATH=conversation_history
UNITY_MEMORY_PATH=unity_memory
```

### 系統設定

- **預設代理人數量**: 3
- **時間加速倍率**: 60.0
- **最大對話輪數**: 10
- **照片生成尺寸**: 1024x1024
- **模擬最大時間步**: 100

## 部署說明

### 啟動服務

```bash
# 安裝依賴
pip install -r requirements.txt

# 啟動服務器
python backend.py

# 或使用 uvicorn
uvicorn backend:app --host 0.0.0.0 --port 8000
```

### Docker 部署

```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 效能考量

### 並發處理

- 使用 FastAPI 的異步處理能力
- LLM 請求使用信號量控制並發數
- 資料庫操作採用異步 I/O

### 記憶體管理

- 人格資料採用延遲載入
- 對話歷史定期清理
- 模擬狀態增量更新

### 快取策略

- 人格資料記憶體快取
- LLM 回應結果快取
- 位置資訊靜態快取

## 安全考量

### API 安全

- CORS 中間件配置
- 請求參數驗證
- 檔案上傳限制

### 資料安全

- 敏感資料加密儲存
- 定期資料備份
- 存取日誌記錄

## 監控和日誌

### 日誌級別

- `DEBUG`: 詳細除錯資訊
- `INFO`: 一般操作資訊
- `WARNING`: 警告訊息
- `ERROR`: 錯誤訊息

### 監控指標

- API 回應時間
- 記憶體使用量
- 並發連接數
- 錯誤率統計

## 擴展指南

### 添加新的模擬場景

1. 繼承 `BaseScenario` 類
2. 實現必要的抽象方法
3. 註冊到 `ScenarioRegistry`
4. 添加對應的 API 端點

### 整合新的 LLM 模型

1. 在 `responder_wrapper.py` 中添加新的包裝器
2. 更新 `LLM_responder.py` 的模型選擇邏輯
3. 測試模型兼容性
4. 更新配置文件

### 擴展人格屬性

1. 更新人格資料結構
2. 修改 `PersonaLoader` 的驗證邏輯
3. 更新相關的 API 回應格式
4. 遷移現有資料

---

*本文件最後更新時間: 2025-01-21*
*版本: 1.0.0*


