# responder_wrapper.py
"""
LLM API 封裝模組
提供本地 Ollama 和遠端 OpenAI API 的統一介面
"""

import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from typing import Optional, Dict, Any, List
import asyncio

load_dotenv()

# ======= CONFIG =======
# OpenAI Models
EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL_4O = "gpt-4o"
CHAT_MODEL_41 = "gpt-4.1"
MINI_MODEL = "gpt-4.1-mini"
NANO_MODEL = "gpt-4.1-nano"
PHOTO_MODEL = "dall-e-3"
OLD_PHOTO_MODEL = "dall-e-2"

# Ollama Config
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")

class ResponderWrapper:
    """
    統一的 LLM API 封裝類別
    支援 OpenAI 和 Ollama 兩種後端
    """
    
    def __init__(self, api_key: Optional[str] = None, use_ollama: Optional[bool] = None):
        """
        初始化 ResponderWrapper
        
        Args:
            api_key: OpenAI API 金鑰
            use_ollama: 是否使用 Ollama 本地模型（預設 True）
        """
        provider = os.getenv("LLM_PROVIDER", "ollama" if os.getenv("USE_OLLAMA", "").lower() == "true" else "openai").lower()
        if use_ollama is None and provider not in {"openai", "ollama"}:
            raise ValueError("LLM_PROVIDER must be openai or ollama")
        self.use_ollama = (provider == "ollama") if use_ollama is None else use_ollama
        self.api_key = api_key
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            self._init_client()
        return self._client

    def _init_client(self):
        """初始化客戶端"""
        if self.use_ollama:
            # 使用 Ollama 本地模型
            self._client = AsyncOpenAI(
                api_key="ollama",  # Ollama 不需要真實 API key
                base_url=f"{OLLAMA_HOST}/v1"
            )

        else:
            # 使用 OpenAI API
            self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("需要設定 OPENAI_API_KEY 環境變數或直接傳入 API 金鑰")
            self._client = AsyncOpenAI(api_key=self.api_key)

    
    def _log_ollama_setup(self):
        """記錄 Ollama 設定資訊"""
        print(f"✅ 使用 Ollama 本地模型")
        print(f"📍 主機位址: {OLLAMA_HOST}")
        print(f"🤖 模型: {OLLAMA_MODEL}")
    
    def _log_openai_setup(self):
        """記錄 OpenAI 設定資訊"""
        print("✅ 使用 OpenAI API")

    
    def _get_model_for_request(self, requested_model: str) -> str:
        """
        根據當前設定決定實際使用的模型
        
        Args:
            requested_model: 請求的模型名稱
            
        Returns:
            實際使用的模型名稱
        """
        if self.use_ollama:
            # 當使用 Ollama 時，將 OpenAI 模型映射到本地模型
            openai_models = [CHAT_MODEL_4O, CHAT_MODEL_41, MINI_MODEL, NANO_MODEL]
            if requested_model in openai_models:
                return OLLAMA_MODEL
        return requested_model
    
    async def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str = CHAT_MODEL_4O,
        temperature: float = 0.7,
        top_p: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        執行聊天完成請求
        
        Args:
            messages: 對話訊息列表
            model: 使用的模型
            temperature: 溫度參數
            top_p: top_p 參數
            **kwargs: 其他參數
            
        Returns:
            API 回應結果
        """
        try:
            actual_model = self._get_model_for_request(model)
            
            response = await self.client.chat.completions.create(
                model=actual_model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )
            
            return response
        except Exception as e:
            raise Exception(f"Chat completion 請求失敗: {str(e)}")
    
    async def generate_embeddings(
        self,
        texts: List[str],
        model: str = EMBEDDING_MODEL
    ) -> Dict[str, Any]:
        """
        生成文字嵌入向量
        
        注意：Embedding 功能目前只支援 OpenAI API
        
        Args:
            texts: 要生成嵌入的文字列表
            model: 嵌入模型
            
        Returns:
            嵌入向量結果
        """
        if self.use_ollama:
            # 如果使用 Ollama，需要臨時建立 OpenAI 客戶端
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise Exception("Embedding 生成需要 OpenAI API Key，但未設定 OPENAI_API_KEY 環境變數")
            
            temp_client = AsyncOpenAI(api_key=openai_api_key)
            response = await temp_client.embeddings.create(
                input=texts,
                model=model
            )
        else:
            response = await self.client.embeddings.create(
                input=texts,
                model=model
            )
        
        return response
    
    async def generate_image(
        self,
        prompt: str,
        model: str = PHOTO_MODEL,
        size: str = "1024x1024",
        quality: str = "standard",
        n: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成圖片
        
        注意：圖片生成功能目前只支援 OpenAI API
        
        Args:
            prompt: 圖片生成提示詞
            model: 圖片生成模型
            size: 圖片尺寸
            quality: 圖片品質
            n: 生成圖片數量
            **kwargs: 其他參數
            
        Returns:
            圖片生成結果
        """
        if self.use_ollama:
            # 如果使用 Ollama，需要臨時建立 OpenAI 客戶端
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise Exception("圖片生成需要 OpenAI API Key，但未設定 OPENAI_API_KEY 環境變數")
            
            temp_client = AsyncOpenAI(api_key=openai_api_key)
            response = await temp_client.images.generate(
                prompt=prompt,
                model=model,
                size=size,
                quality=quality,
                n=n,
                **kwargs
            )
        else:
            response = await self.client.images.generate(
                prompt=prompt,
                model=model,
                size=size,
                quality=quality,
                n=n,
                **kwargs
            )
        
        return response
    
    def get_client_info(self) -> Dict[str, Any]:
        """
        取得客戶端資訊
        
        Returns:
            客戶端設定資訊
        """
        return {
            "use_ollama": self.use_ollama,
            "host": OLLAMA_HOST if self.use_ollama else "api.openai.com",
            "default_model": OLLAMA_MODEL if self.use_ollama else CHAT_MODEL_4O,
            "has_api_key": bool(self.api_key)
        }
    

    async def test_connection(self) -> bool:
        """
        測試連線是否正常
        
        Returns:
            連線是否成功
        """
        try:
            test_messages = [
                {"role": "system", "content": "你是一個測試助手。"},
                {"role": "user", "content": "請回答：測試成功"}
            ]
            
            response = await self.chat_completion(
                messages=test_messages,
                temperature=0.1
            )
            
            return bool(response.choices[0].message.content)
        except Exception as e:
            print(f"連線測試失敗: {e}")
            return False

        









# ======= 工廠方法 =======
def create_openai_wrapper(api_key: Optional[str] = None) -> ResponderWrapper:
    """
    建立 OpenAI API 封裝器
    
    Args:
        api_key: OpenAI API 金鑰
        
    Returns:
        配置為使用 OpenAI 的 ResponderWrapper 實例
    """
    return ResponderWrapper(api_key=api_key, use_ollama=False)

def create_ollama_wrapper(api_key: Optional[str] = None) -> ResponderWrapper:
    """
    建立 Ollama 本地模型封裝器
    
    Args:
        api_key: OpenAI API 金鑰（用於圖片生成和嵌入功能）
        
    Returns:
        配置為使用 Ollama 的 ResponderWrapper 實例
    """
    return ResponderWrapper(api_key=api_key, use_ollama=True)

def create_auto_wrapper(api_key: Optional[str] = None) -> ResponderWrapper:
    """
    自動選擇最佳的封裝器
    優先使用 Ollama，如果不可用則使用 OpenAI
    
    Args:
        api_key: OpenAI API 金鑰
        
    Returns:
        自動選擇的 ResponderWrapper 實例
    """
    # 檢查 Ollama 是否可用的邏輯可以在這裡實現
    # 目前預設使用 Ollama
    return create_ollama_wrapper(api_key)

