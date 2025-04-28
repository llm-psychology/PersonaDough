# 🛠 人格模擬系統 - 技術規格書 (Spec)

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
- 可以是 CLI 問答、表單、或網站表單。
- 確保使用者回答不能是空字串。

```python
def collect_user_answers(question_list: List[str]) -> List[Dict[str, str]]:
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
def format_qa_pairs(qa_pairs: List[Dict[str, str]]) -> List[str]:
    ...
```

---

## 3. Embedding 建立 (generate_embeddings)

### 功能  
對所有記憶片段做 embedding。

### Input  
- `docs: List[str]`
- `embedding_model: str` （例如 `"text-embedding-ada-002"`）

### Output  
- `embeddings: np.ndarray` (shape: (n_docs, embedding_dim))

### 實作細節
- 可以用 OpenAI Embedding API，或其他本地模型。

```python
def generate_embeddings(docs: List[str], embedding_model: str) -> np.ndarray:
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
- 同時儲存 docs (記憶片段) 外部 list，方便檢索對應。

```python
def build_vector_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
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
- `embedding_model: str`
- `top_k: int = 3`
- `similarity_threshold: float = 0.7`

### Output  
- `retrieved_docs: List[str]`

### 實作細節
- 如果相似度小於設定的門檻（例如0.7），直接回傳空列表，避免人格走樣。

```python
def retrieve_similar_docs(query: str, index: faiss.IndexFlatL2, docs: List[str], embedding_model: str, top_k: int = 3, similarity_threshold: float = 0.7) -> List[str]:
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
- `system_prompt: str`
- `chat_messages: List[Dict[str, str]]`  
  （符合 OpenAI Chat API 格式）

### 實作細節
- 如果 `retrieved_docs` 為空，可以在 prompt 特別說明「無記憶片段可用，請謹慎推測」。

```python
def build_simulation_prompt(retrieved_docs: List[str], user_query: str) -> List[Dict[str, str]]:
    ...
```

---

## 7. 呼叫 LLM 回答 (simulate_persona_answer)

### 功能  
用組好的 Prompt，向 LLM發出請求，取得回答。

### Input  
- `chat_messages: List[Dict[str, str]]`
- `model: str` （例如 `"gpt-3.5-turbo"`）

### Output  
- `response_text: str`

### 實作細節
- 可設定 temperature 控制回答隨機程度。

```python
def simulate_persona_answer(chat_messages: List[Dict[str, str]], model: str) -> str:
    ...
```

---

# 流程總結圖

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

---

# 注意事項補充

- **問卷問題設計得越多樣，模仿效果越好。**
- **Embedding模型、檢索向量DB和LLM需要相互搭配調整。**
- **強烈建議加上`similarity threshold`，否則人格容易亂掉。**