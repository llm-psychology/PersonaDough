#  interviewer agent 面試官代理

[回主說明](../README.md)

# 流程圖

```
問卷 ➔ QA pairs ➔ 格式化 ➔ Embedding ➔ FAISS Index
                     ↓
                 使用者提問
                     ↓
               ➔ 查相似片段
                     ↓
               ➔ 組 Prompt
                     ↓
               ➔ LLM模擬回答
```

## 1. 問卷收集 (collect_user_answers)

### 功能  
問使用者一系列標準化問題，收集回答。

### Input  
- `question_list: List[str]`  
  （預先定義好的標準問題列表）

### Output  
- `qa_pairs: List[Dict[str, str]]`  
  每個元素是 `{q: 問題, a: 使用者回答}` 的字典。

### 實作細節
- 使用 CLI 介面進行問答。
- 確保使用者回答不能是空字串。

```python
def collect_user_answers(self, question_list: List[str]) -> List[Dict[str, str]]:
    ...
```

---

## 2. 文件格式化 (format_qa_pairs)

### 功能  
把問答對組成單一文本格式（後續送去embedding）。

### Input  
- `qa_pairs: List[Dict[str, str]]`

### Output  
- `docs: List[str]`  
  每條是 `"Q：{問題} A：{回答}"`

### 實作細節
- 保持固定格式方便後續 retrival 和模仿。

```python
def format_qa_pairs(self, qa_pairs: List[Dict[str, str]]) -> List[str]:
    ...
```

---

## 3. Embedding 建立 (generate_embedding)

### 功能  
對所有記憶片段做 embedding。

### Input  
- `docs: List[str]`

### Output  
- `embeddings: np.ndarray` (shape: (n_docs, embedding_dim))

### 實作細節
- 使用 OpenAI Embedding API。
- 繼承自 LLM_responder 類別。

```python
def generate_embedding(self, docs: List[str]) -> np.ndarray:
    ...
```

---

## 4. 向量資料庫建置 (build_vector_index)

### 功能  
把 embedding 加到向量資料庫（用 FAISS）。

### Input  
- `embeddings: np.ndarray`

### Output  
- `index: faiss.IndexFlatL2`

### 實作細節
- 使用 `IndexFlatL2` 做基本L2距離搜尋。

```python
def build_vector_index(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
    ...
```

---

## 5. 查詢檢索 (retrieve_similar_docs)

### 功能  
新提問時，找出最接近的 k 筆記憶片段。

### Input  
- `query: str`
- `index: faiss.IndexFlatL2`
- `docs: List[str]`
- `top_k: int = 3`
- `similarity_threshold: float = 0.7`

### Output  
- `tuple: (similar_docs: List[str], distances: np.ndarray)`
  - similar_docs: 相似的文件列表
  - distances: 對應的距離值

### 實作細節
- 返回所有 top_k 個結果，不管距離多遠。
- 距離值範圍在 0 到 2 之間，0 表示完全相同，2 表示完全相反。

```python
def retrieve_similar_docs(self, query: str, index: faiss.IndexFlatL2, docs: List[str], top_k: int = 3, similarity_threshold: float = 0.7) -> tuple:
    ...
```

---

## 6. Prompt 組合 (build_simulation_prompt)

### 功能  
組合 System Prompt，準備送進 LLM。

### Input  
- `retrieved_docs: List[str]`
- `user_query: str`

### Output  
- `chat_messages: List[Dict[str, str]]`  
  （符合 OpenAI Chat API 格式）

### 實作細節
- 如果 `retrieved_docs` 為空，在 prompt 中說明「無可參考的記憶資料」。

```python
def build_simulation_prompt(self, retrieved_docs: List[str], user_query: str) -> List[Dict[str, str]]:
    ...
```

---

## 7. 呼叫 LLM 回答 (simulate_persona_answer)

### 功能  
用組好的 Prompt，向 LLM發出請求，取得回答。

### Input  
- `chat_messages: List[Dict[str, str]]`

### Output  
- `response_text: str`

### 實作細節
- 繼承自 LLM_responder 類別。
- 使用 GPT-4 模型。

```python
def simulate_persona_answer(self, chat_messages: List[Dict[str, str]]) -> str:
    ...
```

---

## 8. RAG 資料庫管理

### 功能  
儲存和載入 RAG 資料庫。

### 方法
1. `save_rag_database(self, name: str, embeddings: np.ndarray, index: faiss.IndexFlatL2, docs: List[str], qa_pairs: List[Dict[str, str]])`
   - 儲存 embeddings、FAISS index、文件和問答對

2. `load_rag_database(self, name: str) -> tuple`
   - 載入並返回 embeddings、index、docs 和 qa_pairs

### 實作細節
- 資料庫儲存在 `interviewer/rag_database` 目錄下
- 每個資料庫有自己的子目錄
- 支援多個資料庫的管理

---

## 9. 批次處理功能

### 功能  
批次處理問卷問題。

### 方法
`process_questions_in_batches(self, loader: DataLoader, persona_loader: PersonaLoader = None, persona_id: str = None) -> List[Dict[str, str]]`

### 實作細節
- 支援使用 PersonaLoader 自動回答問題
- 預設每批次處理 10 個問題
- 支援 JSON 格式的問答輸出
- 提供相似度分析和距離顯示

# 注意事項補充

- **問卷問題設計得越多樣，模仿效果越好。**
- **Embedding模型、檢索向量DB和LLM需要相互搭配調整。**
- **相似度距離值說明：**
  - 0 表示完全相同
  - 2 表示完全相反
  - 一般來說，距離小於 0.5 的資料被認為是比較相似的
- **支援角色自動回答功能，可以讓指定角色自動回答問卷問題。**
- **提供完整的 RAG 資料庫管理功能，方便保存和載入不同的人格資料。**