import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict
import numpy as np

load_dotenv()

# ======= CONFIG =======
EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-4o"

class LLM_responder:
    
    def __init__(self, api_key=None):
        """LLM api建構子"""   
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("需要設定 OPENAI_API_KEY 環境變數或直接傳入 API 金鑰")
        
        # 初始化 OpenAI 客戶端
        self.client = OpenAI(api_key=self.api_key)


    def _call_chat_openai_api(self, model, messages, temperature):
        """使用 OpenAI 套件呼叫 chat.completions API"""
        try:
            # 使用 OpenAI 客戶端發送請求
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            
            # 從回應中提取內容
            return response
            
        except Exception as e:
            raise Exception(f"API 請求失敗: {str(e)}")
    

    def chat_gpt_4o(self, prompt: str, temperature: int)->str:
        """使用 chat.completions API透過gpt-4o回答問題"""
        model = CHAT_MODEL
        prompt = [{"role": "user", "content": prompt}]
        res = self._call_chat_openai_api(model, prompt, temperature)
        return res.choices[0].message.content
            
    
    def simulate_persona_answer(self, chat_messages: List[Dict[str, str]], model: str = CHAT_MODEL) -> str:
        """使用 chat.completions API透過gpt-4o回答問題，相似的function: chat_gpt_4o()"""
        model = CHAT_MODEL
        prompt = chat_messages
        res = self._call_chat_openai_api(model, prompt, 0.7)
        return res.choices[0].message.content.strip()
    

    def _call_embeddings_openai_api(self, docs, model: str = EMBEDDING_MODEL):
        """使用 OpenAI 套件呼叫 embeddings.create API"""
        res = self.client.embeddings.create(
            input=docs,
            model=model
        )
        return res


    def generate_embedding(self, docs: List[str], embedding_model: str = EMBEDDING_MODEL):
        """使用 embeddings.create 得到 embeddings"""
        res = self._call_embeddings_openai_api(docs, embedding_model)
        return np.array([d.embedding for d in res.data])