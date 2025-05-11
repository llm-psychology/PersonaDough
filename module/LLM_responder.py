import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from typing import List, Dict
import numpy as np

load_dotenv()

# ======= CONFIG =======
EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL_4O = "gpt-4o"
CHAT_MODEL_41 = "gpt-4.1"
MINI_MODEL = "gpt-4.1-mini"
NANO_MODEL = "gpt-4.1-nano"

class LLM_responder:
    """LLM api 物件"""  

    def __init__(self, api_key=None):
        """LLM api建構子"""   
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("需要設定 OPENAI_API_KEY 環境變數或直接傳入 API 金鑰")
        
        # 初始化 OpenAI 客戶端
        self.client = AsyncOpenAI(api_key=self.api_key)

    async def _call_chat_openai_api(self, model, messages, temperature = 0.7, top_p = 1):
        """使用 OpenAI 套件呼叫 chat.completions API"""
        try:
            # 使用 OpenAI 客戶端發送請求
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p = top_p
            )
            
            # 從回應中提取內容
            return response
            
        except Exception as e:
            raise Exception(f"API 請求失敗: {str(e)}")
    
    async def chat_gpt_4o(self, prompt: str, temperature: int)->str:
        """簡單對話-使用 chat.completions API透過gpt-4o的user回答單輪對話"""
        model = CHAT_MODEL_4O
        prompt = [{"role": "user", "content": prompt}]
        res = await self._call_chat_openai_api(model, prompt, temperature)
        return res.choices[0].message.content
    #================================================================================================================
    async def full_chat_gpt_4o(self, sys_prompt: str="", usr_prompt: str="", temperature: int=0.7, top_p: int=1)->str:
        """使用 chat.completions API透過gpt-4o的自定義回答"單輪"對話

        usage: full_chat_gpt_4o("你是數學教學小助手，不可以誤人子弟亂回答，要教給五歲小孩聽的", "回答一個題目是: 1+1=多少?", 0.3)"""
        model = CHAT_MODEL_4O
        prompt=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": usr_prompt},
        ]
        res = await self._call_chat_openai_api(model, prompt, temperature, top_p)
        return res.choices[0].message.content
        
    async def full_chat_gpt_41(self, sys_prompt: str="", usr_prompt: str="", temperature: int=0.7, top_p: int=1)->str:
        """gpt4.1比較會遵守格式
        使用 chat.completions API透過gpt-4.1的自定義回答"單輪"對話

        usage: full_chat_gpt_41("你是數學教學小助手，不可以誤人子弟亂回答，要教給五歲小孩聽的", "回答一個題目是: 1+1=多少?", 0.3)"""
        model = CHAT_MODEL_41
        prompt=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": usr_prompt},
        ]
        top_p = 1
        res = await self._call_chat_openai_api(model, prompt, temperature, top_p)
        return res.choices[0].message.content
    
    async def full_chat_gpt_41_mini(self, sys_prompt: str="", usr_prompt: str="", temperature: int=0.9, top_p: int=1)->str:
        """gpt4.1mini 
        使用 chat.completions API透過gpt-4.1-mini的自定義回答"單輪"對話

        usage: full_chat_gpt_41_mini("總結剛剛的描述內容", "太陽是什麼? 太陽是一個恆星", 0.3)"""
        model = MINI_MODEL
        prompt=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": usr_prompt},
        ]
        top_p = 1
        res = await self._call_chat_openai_api(model, prompt, temperature, top_p)
        return res.choices[0].message.content
    
    async def full_chat_gpt_41_nano(self, sys_prompt: str="", usr_prompt: str="", temperature: int=0.9, top_p: int=1)->str:
        """gpt4.1nano 比較快比較不需要智商的
        使用 chat.completions API透過gpt-4.1-nano的自定義回答"單輪"對話

        usage: full_chat_gpt_41_nano("eval剛剛的內容", "太陽是什麼? 太陽是一個恆星", 0.3)"""
        model = NANO_MODEL
        prompt=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": usr_prompt},
        ]
        top_p = 1
        res = await self._call_chat_openai_api(model, prompt, temperature, top_p)
        return res.choices[0].message.content
    #================================================================================================================
    async def simulate_persona_answer(self, chat_messages: List[Dict[str, str]], model: str = CHAT_MODEL_4O) -> str:
        """使用 chat.completions API透過gpt-4o回答問題，相似的function: chat_gpt_4o()"""
        model = CHAT_MODEL_4O
        prompt = chat_messages
        res = await self._call_chat_openai_api(model, prompt, 0.7)
        return res.choices[0].message.content.strip()
    
    async def _call_embeddings_openai_api(self, docs, model: str = EMBEDDING_MODEL):
        """使用 OpenAI 套件呼叫 embeddings.create API"""
        res = await self.client.embeddings.create(
            input=docs,
            model=model
        )
        return res

    async def generate_embedding(self, docs: List[str], embedding_model: str = EMBEDDING_MODEL):
        """使用 embeddings.create 得到 embeddings"""
        res = await self._call_embeddings_openai_api(docs, embedding_model)
        return np.array([d.embedding for d in res.data])