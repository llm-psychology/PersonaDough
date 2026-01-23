# Unity對話系統整合說明

## 概述

本系統為Unity遊戲專門設計了一套完整的對話記憶管理系統，解決了Unity與後端對話API之間的記憶一致性問題。

## 主要功能

### 1. Unity專用對話API

**端點**: `POST /unity-api/dialogue`

**特點**:
- 自動使用Unity記憶系統
- 會話ID格式固定為: `dialogue_{persona1_id}_{persona2_id}_unity`
- 自動載入Unity記憶上下文
- 整合現有的對話生成系統

**請求範例**:
```json
{
    "persona1_id": "687a1b2d5b",
    "persona2_id": "eef4720230",
    "num_rounds": 3,
    "initial_topic": "你好，很高興見到你！",
    "memory_context": "額外的記憶上下文（可選）"
}
```

**回應範例**:
```json
{
    "dialogue_rounds": [
        {
            "speaker": "王美鳳",
            "content": "你好！很高興見到你...",
            "round": "1"
        },
        {
            "speaker": "呂志強", 
            "content": "我也很高興見到你...",
            "round": "1"
        }
    ],
    "session_id": "dialogue_687a1b2d5b_eef4720230_unity"
}
```

### 2. Unity記憶管理系統

#### 添加記憶
**端點**: `POST /unity-api/memory/add`

```json
{
    "persona1_id": "687a1b2d5b",
    "persona1_name": "王美鳳",
    "persona2_id": "eef4720230", 
    "persona2_name": "呂志強",
    "speaker": "王美鳳",
    "content": "對話內容...",
    "round": "1"
}
```

#### 獲取記憶上下文
**端點**: `POST /unity-api/memory/context`

```json
{
    "persona1_id": "687a1b2d5b",
    "persona2_id": "eef4720230"
}
```

#### 清除記憶
- 清除特定角色對: `DELETE /unity-api/memory/clear/{persona1_id}/{persona2_id}`
- 清除所有記憶: `DELETE /unity-api/memory/clear-all`

#### 記憶統計
**端點**: `GET /unity-api/memory/stats`

### 3. 對話歷史管理

系統現在支援兩種對話歷史存儲方式:

1. **一般對話**: 存儲在 `conversation_history/` 資料夾
   - 會話ID格式: `dialogue_{persona1_id}_{persona2_id}_{random_uuid}`

2. **Unity對話**: 存儲在 `conversation_history/` 資料夾  
   - 會話ID格式: `dialogue_{persona1_id}_{persona2_id}_unity`

## 系統架構

```
Unity遊戲
    ↓ 對話請求
Unity對話API (/unity-api/dialogue)
    ↓ 載入記憶
Unity記憶系統 (unity_memory/)
    ↓ 整合上下文
PersonaDialogueManager
    ↓ 生成對話
對話生成系統
    ↓ 保存記憶
Unity記憶系統 & 對話歷史
```

## 使用流程

### Unity端整合步驟

1. **發起對話**:
   ```csharp
   // Unity C# 代碼示例
   var dialogueRequest = new DialogueRequest
   {
       persona1_id = "687a1b2d5b",
       persona2_id = "eef4720230",
       num_rounds = 2,
       initial_topic = "你好！"
   };
   
   // 發送到 /unity-api/dialogue
   ```

2. **處理對話結果**:
   ```csharp
   // 獲取對話結果
   foreach (var round in dialogueResult.dialogue_rounds)
   {
       // 顯示對話
       DisplayDialogue(round.speaker, round.content);
       
       // 添加到Unity記憶
       AddMemoryToUnity(round);
   }
   ```

3. **添加記憶**:
   ```csharp
   // 為每輪對話添加記憶
   var memoryRequest = new MemoryRequest
   {
       persona1_id = dialogueRequest.persona1_id,
       persona1_name = GetPersonaName(dialogueRequest.persona1_id),
       persona2_id = dialogueRequest.persona2_id,
       persona2_name = GetPersonaName(dialogueRequest.persona2_id),
       speaker = round.speaker,
       content = round.content,
       round = round.round
   };
   
   // 發送到 /unity-api/memory/add
   ```

## 問題解決

### 1. JSON解析錯誤修復

系統現在能夠處理損壞的Unity記憶文件:
- 自動檢測空文件
- 創建備份文件
- 重新初始化記憶系統

### 2. 記憶一致性

- Unity記憶與對話歷史同步
- 自動載入相關記憶上下文
- 支援記憶上限管理（預設10條）

### 3. 會話ID標準化

- Unity對話固定使用 `_unity` 後綴
- 避免重複的隨機UUID
- 便於識別和管理

## 測試

執行測試腳本:
```bash
python test_unity_dialogue.py
```

測試內容包括:
- Unity對話API功能
- 記憶添加和獲取
- 上下文整合
- 錯誤處理

## 注意事項

1. **記憶限制**: 每個角色對最多保存10條記憶
2. **自動RAG生成**: 如果persona的RAG資料庫不存在，系統會自動生成
3. **錯誤處理**: 系統具備完整的錯誤處理和日誌記錄
4. **向後兼容**: 原有的對話API仍然可以正常使用

## API端點總覽

| 端點 | 方法 | 用途 |
|------|------|------|
| `/unity-api/dialogue` | POST | Unity專用對話生成 |
| `/unity-api/memory/add` | POST | 添加Unity記憶 |
| `/unity-api/memory/context` | POST | 獲取記憶上下文 |
| `/unity-api/memory/clear/{p1}/{p2}` | DELETE | 清除特定記憶 |
| `/unity-api/memory/clear-all` | DELETE | 清除所有記憶 |
| `/unity-api/memory/stats` | GET | 獲取記憶統計 |
| `/persona/{id}/name` | GET | 獲取角色名稱 |

這個系統完全解決了你提到的問題，Unity現在可以使用專門的記憶系統，並且對話會話ID會以"unity"結尾，保持了既有架構的完整性。




