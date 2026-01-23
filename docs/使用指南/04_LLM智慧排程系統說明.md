# LLM 智能排程系統說明

## 系統概述

LLM 智能排程系統是一個基於大型語言模型的自動排程生成器，能夠根據角色的個性、興趣、職業等特徵，自動生成個性化的日常活動安排。

## 核心功能

### 1. 個性分析模組
- **自動提取個性特徵**：從 persona 資料中分析個性關鍵詞
- **興趣愛好識別**：識別角色的興趣和偏好
- **生活模式判斷**：分析能量模式（晨型/夜型）和社交程度
- **職業適配**：根據職業調整活動類型

### 2. LLM 活動生成器
- **上下文感知**：考慮時間、天氣、季節等環境因素
- **個性化推理**：基於角色特徵生成合適的活動
- **約束條件處理**：支援自定義限制條件
- **多樣性保證**：避免重複單調的活動安排

### 3. 智能排程引擎
- **單個活動生成**：為特定時間段生成單個活動
- **一日排程生成**：生成完整的一天活動安排
- **動態調整**：根據實時情況調整排程
- **優先級管理**：為活動分配合理的優先級

## 技術架構

### 核心類別

#### PersonalityProfile
```python
@dataclass
class PersonalityProfile:
    persona_id: str
    name: str
    age: int
    occupation: str
    personality_traits: List[str]  # 個性特徵
    interests: List[str]           # 興趣愛好
    daily_preferences: Dict        # 日常偏好
    energy_pattern: str           # 能量模式
    social_level: str             # 社交程度
```

#### ActivityRequest
```python
@dataclass
class ActivityRequest:
    persona_id: str
    time_slot: str
    duration_minutes: int
    context: ContextInfo          # 環境上下文
    constraints: List[str]        # 約束條件
```

#### GeneratedActivity
```python
@dataclass
class GeneratedActivity:
    name: str                     # 活動名稱
    activity_type: str           # 活動類型
    location: str                # 地點
    duration: int                # 持續時間
    priority: float              # 優先級
    reasoning: str               # AI 推理過程
    social_participants: List[str] # 社交參與者
    required_items: List[str]    # 需要物品
```

### LLM 提示工程

系統使用精心設計的提示模板，包含：
- **角色資訊**：個性、興趣、職業等
- **環境上下文**：時間、天氣、可用地點
- **約束條件**：用戶自定義的限制
- **輸出格式**：結構化的 JSON 回應

## API 端點

### 1. 個性檔案管理
```
GET /api/llm-schedule/personality/{persona_id}
```
獲取角色的個性分析結果。

### 2. 單個活動生成
```
POST /api/llm-schedule/generate-activity
```
為指定角色和時間段生成單個活動。

**請求參數**：
```json
{
    "persona_id": "687a1b2d5b",
    "time_slot": "上午時光",
    "duration_minutes": 90,
    "weather": "晴朗",
    "constraints": ["適合戶外活動"]
}
```

### 3. 一日排程生成
```
POST /api/llm-schedule/generate-daily
```
生成完整的一天活動安排。

**請求參數**：
```json
{
    "persona_id": "687a1b2d5b",
    "date": "2025-10-02"
}
```

### 4. 排程應用
```
POST /api/llm-schedule/apply-to-agent
```
將 LLM 生成的排程應用到生活模擬系統中。

## 使用方式

### 1. 網頁介面
訪問 `http://localhost:20000/llm-schedule-control` 使用圖形化介面：
- 選擇角色並查看個性檔案
- 生成單個活動或一日排程
- 直接應用到生活模擬系統
- 監控角色狀態

### 2. API 調用
```python
import httpx
import asyncio

async def generate_activity():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:20000/api/llm-schedule/generate-activity",
            json={
                "persona_id": "687a1b2d5b",
                "time_slot": "下午時光",
                "duration_minutes": 120,
                "weather": "晴朗"
            }
        )
        return response.json()
```

### 3. 命令行測試
```bash
python test_llm_scheduler.py
```

## 個性化特徵

### 個性特徵識別
系統能識別以下個性特徵：
- **友善型**：安排更多社交活動
- **外向型**：偏好戶外和群體活動
- **內向型**：安排更多獨處和室內活動
- **創意型**：包含藝術和創作活動
- **分析型**：安排學習和思考時間
- **活力型**：包含更多運動和活動
- **冷靜型**：安排放鬆和冥想時間

### 興趣愛好適配
- **運動愛好者**：更多體育活動
- **閱讀愛好者**：安排閱讀時間
- **音樂愛好者**：包含音樂相關活動
- **烹飪愛好者**：安排料理時間
- **科技愛好者**：包含技術學習
- **旅行愛好者**：安排探索活動

### 職業適配
- **學生**：以學習為主，適當娛樂
- **上班族**：平衡工作與休閒
- **自由職業**：彈性安排工作時間
- **退休人員**：重視健康和社交

## 環境感知

### 時間感知
- **晨型人**：早起安排重要活動
- **夜型人**：晚上安排主要活動
- **彈性型**：根據具體情況調整

### 天氣適配
- **晴天**：優先戶外活動
- **雨天**：安排室內活動
- **雪天**：考慮交通和安全

### 季節調整
- **春季**：踏青、運動
- **夏季**：游泳、避暑
- **秋季**：賞楓、收穫
- **冬季**：室內、溫暖活動

## 活動類型

系統支援以下活動類型：
- `rest`: 休息放鬆
- `sleep`: 睡眠休息
- `work`: 工作學習
- `study`: 學習進修
- `eat`: 用餐飲食
- `play_football`: 運動娛樂
- `socialize`: 社交互動
- `exercise`: 體能訓練
- `hobby`: 興趣愛好
- `shopping`: 購物採買
- `travel`: 外出旅行

## 地點類型

支援的地點類型：
- `home`: 家中
- `school`: 學校
- `workplace`: 工作場所
- `restaurant`: 餐廳
- `park`: 公園
- `soccer_field`: 運動場
- `library`: 圖書館
- `shopping_mall`: 購物中心
- `cafe`: 咖啡廳

## 優勢特點

### 1. 高度個性化
- 基於真實的 persona 資料
- 考慮個性、興趣、職業等多維度特徵
- 生成符合角色特質的活動

### 2. 智能推理
- 使用 LLM 進行複雜推理
- 提供活動選擇的理由說明
- 考慮多種約束條件

### 3. 動態適應
- 根據環境變化調整活動
- 支援實時約束條件
- 可處理特殊事件

### 4. 無縫整合
- 與現有生活模擬系統完全兼容
- 自動轉換為 Unity 可執行的任務
- 支援實時狀態監控

## 使用場景

### 1. 遊戲開發
- 為 NPC 生成真實的日常行為
- 創造更有趣的角色互動
- 提升遊戲的沉浸感

### 2. 模擬研究
- 研究人類行為模式
- 測試不同個性的行為差異
- 驗證社會學理論

### 3. 教育應用
- 展示個性對行為的影響
- 教授時間管理概念
- 培養規劃能力

### 4. 娛樂體驗
- 創造個性化的虛擬角色
- 觀察不同角色的生活方式
- 探索人工智能的創造力

## 未來擴展

### 1. 學習能力
- 從用戶反饋中學習
- 優化活動生成算法
- 提升個性化程度

### 2. 社交互動
- 考慮角色間的關係
- 生成協作活動
- 處理衝突和協調

### 3. 長期規劃
- 支援週期性活動
- 考慮長期目標
- 生成發展軌跡

### 4. 多模態輸入
- 支援語音指令
- 圖像環境識別
- 實時情感分析

## 總結

LLM 智能排程系統結合了人工智能的推理能力和個性化分析，為虛擬角色創造了真實、多樣、有趣的日常生活。系統不僅技術先進，更重要的是能夠理解和模擬人類的複雜行為模式，為各種應用場景提供了強大的支持。


