# PersonaDough

把人格像麵糰／黏土一樣捏出來，輸出一份可供其他系統使用的虛構人格 JSON。

**專案邊界：人格生成到 JSON 匯出為止。** 不負責訪談記憶、回答一致性、行為錨定、HTTP API 或生活模擬。

## 流程

`基本資料 → 人格／社交／能力特質 → 背景故事 → 語言風格 → 簡化行為 → JSON`

原有生成流程已移至 `src/personadough/generator.py`。預設不自動生成圖片，也不以生成後的 temperature／top_p 建議當作心理測量結果。可選的參數分析保留為 Python 方法，尚未驗證有效性。

## 使用

需要 Python 3.10+，先建立虛擬環境並安裝：

```powershell
python -m pip install -e .
Copy-Item .env.example .env
```

在 `.env` 設定 `LLM_PROVIDER=ollama`、`OLLAMA_HOST`、`OLLAMA_MODEL`，或改為 `LLM_PROVIDER=openai` 並設定 `OPENAI_API_KEY`。

```powershell
python -m personadough --provider ollama --count 1
python -m personadough --provider openai --output-dir ./humanoid/humanoid_database
```

預設產物：`humanoid/humanoid_database/{id}_{姓名}.json`。可用 `PERSONA_DATA_DIR` 指定共享資料夾；跨 repo 使用時請設定相同的絕對路徑。

`examples/5c771e5f3c_李靜宜.json` 是這次新增的虛構示例，並非從原資料庫取回的原始角色。生成器不會改寫這份範例。

## 資料格式

保留原有中文欄位：`基本資料`、`人格屬性`、`生平故事`、`語言行為`、`簡化行為`，另可有 `AI參數`。`基本資料.id` 是十位十六進位字串。

`personadough.contract.validate_persona()` 驗證跨 repo 的必要欄位。PsychAnchor 讀取 JSON，不修改這份初始人格；後續經歷與訪談另外儲存。

## 程式結構

- `generator.py`：人格生成與匯出。
- `prompts.json`：只含生成領域提示詞。
- `contract.py`：既有 JSON 契約驗證。
- `runtime/`：共用 OpenAI／Ollama adapter 與提示詞載入工具，供 PsychAnchor 重用，無 API 或模擬依賴。

本地聊天使用 Ollama；共用 adapter 的 embedding／可選圖片功能目前仍使用 OpenAI。單純捏人格的預設流程只需所選的聊天後端。

## 測試與狀態

```powershell
python -m pip install -e '.[dev]'
python -m pytest tests
```

此版本是畢專半成品的第一次拆分。離線測試使用固定模型回應驗證資料流，不代表真實模型品質或論文結果。

原始版本為 `94f086f`；完整舊檔案另存主導覽 repo 的 `archive/PersonaDough-94f086f/`。拆分設計見 [persona-platform](https://github.com/llm-psychology/persona-platform)。

## 授權

GPL-3.0，沿用原始 `LICENSE`。舊 README 的 MIT 標示與實際授權檔不一致，這次依授權檔修正。
