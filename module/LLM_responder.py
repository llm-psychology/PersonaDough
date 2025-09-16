# LLM_responder.py
import os
import json
from dotenv import load_dotenv
from typing import List, Dict
import numpy as np
from deprecated import deprecated
import asyncio
import time
import aiohttp
from PIL import Image
from io import BytesIO

# 導入新的封裝模組
from .responder_wrapper import (
    ResponderWrapper, 
    create_openai_wrapper, 
    create_ollama_wrapper,
    EMBEDDING_MODEL, CHAT_MODEL_4O, CHAT_MODEL_41, 
    MINI_MODEL, NANO_MODEL, PHOTO_MODEL, OLD_PHOTO_MODEL
)

load_dotenv()

class LLM_responder:
    """LLM api 物件"""  

    def __init__(self, api_key=None, use_ollama=False):
        """LLM api建構子
        
        Args:
            api_key: OpenAI API 金鑰
            use_ollama: 是否使用 Ollama 本地模型（預設 True）
        """   
        # 使用新的封裝模組
        if use_ollama:
            self.wrapper = create_ollama_wrapper(api_key)
        else:
            self.wrapper = create_openai_wrapper(api_key)
        
        # 保持向後兼容
        self.use_ollama = use_ollama
        self.client = self.wrapper.client

    async def __call_completions_openai_api(self, model, messages, temperature = 0.7, top_p = 1, max_tokens = None):
        """使用封裝模組呼叫 chat.completions API"""
        try:
            # 準備參數
            params = {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "top_p": top_p
            }
            
            # 如果指定了 max_tokens，加入參數
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
            
            # 使用新的封裝模組進行請求
            response = await self.wrapper.chat_completion(**params)
            
            return response
        except Exception as e:
            raise Exception(f"API 請求失敗: {str(e)}")

    async def __call_images_openai_api(self, model, prompt: str, output_path: str, photo_size) -> None:
        """reification過的prompt轉圖片並下載
        https://platform.openai.com/docs/api-reference/images/create
        
        注意：圖片生成永遠使用 OpenAI，因為 Ollama 不支援圖片生成
        """
        try:
            # 使用新的封裝模組進行圖片生成
            response = await self.wrapper.generate_image(
                prompt=prompt,
                model=model,
                size=photo_size,
                quality="standard",
                n=1
            )
            
            image_url = response.data[0].url
            if not image_url:
                raise Exception("圖片 URL 為空，無法下載圖片")
            
            await self.download_and_convert_to_jpg_async(image_url, output_path)
        except Exception as e:
            print(f"下載並轉換圖片時發生錯誤:{e}")

    async def __call_embeddings_openai_api(self, docs, model: str = EMBEDDING_MODEL):
        """使用封裝模組呼叫 embeddings.create API
        
        注意：Embedding 永遠使用 OpenAI，因為 Ollama 的 embedding 格式可能不同
        """
        try:
            # 使用新的封裝模組進行嵌入生成
            response = await self.wrapper.generate_embeddings(
                texts=docs,
                model=model
            )
            return response
        except Exception as e:
            raise Exception(f"Embedding 生成失敗: {str(e)}")
    
    #================================================================================================================

    async def full_chat_gpt_4o(self, sys_prompt: str="", usr_prompt: str="", temperature: float=0.7, top_p: int=1)->str:
        """使用 chat.completions API透過gpt-4o的自定義回答"單輪"對話
        
        注意：當 use_ollama=True 時，會自動切換到本地 Ollama 模型

        usage: full_chat_gpt_4o("你是數學教學小助手，不可以誤人子弟亂回答，要教給五歲小孩聽的", "回答一個題目是: 1+1=多少?", 0.3)"""
        model = CHAT_MODEL_4O
        prompt=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": usr_prompt},
        ]
        respond = await self.__call_completions_openai_api(model, prompt, temperature, top_p)
        return respond.choices[0].message.content
        
    async def full_chat_gpt_41(self, sys_prompt: str="", usr_prompt: str="", temperature: float=0.7, top_p: int=1)->str:
        """gpt4.1比較會遵守格式
        使用 chat.completions API透過gpt-4.1的自定義回答"單輪"對話
        
        注意：當 use_ollama=True 時，會自動切換到本地 Ollama 模型

        usage: full_chat_gpt_41("你是數學教學小助手，不可以誤人子弟亂回答，要教給五歲小孩聽的", "回答一個題目是: 1+1=多少?", 0.3)"""
        model = CHAT_MODEL_41
        prompt=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": usr_prompt},
        ]
        top_p = 1
        respond = await self.__call_completions_openai_api(model, prompt, temperature, top_p)
        return respond.choices[0].message.content
    
    async def full_chat_gpt_41_mini(self, sys_prompt: str="", usr_prompt: str="", temperature: float=0.9, top_p: int=1)->str:
        """gpt4.1mini 
        使用 chat.completions API透過gpt-4.1-mini的自定義回答"單輪"對話
        
        注意：當 use_ollama=True 時，會自動切換到本地 Ollama 模型

        usage: full_chat_gpt_41_mini("總結剛剛的描述內容", "太陽是什麼? 太陽是一個恆星", 0.3)"""
        model = MINI_MODEL
        prompt=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": usr_prompt},
        ]
        top_p = 1
        respond = await self.__call_completions_openai_api(model, prompt, temperature, top_p)
        output = respond.choices[0].message.content
        return output if output is not None else ""
    
    async def full_chat_gpt_41_nano(self, sys_prompt: str="", usr_prompt: str="", temperature: float=0.9, top_p: int=1)->str:
        """gpt4.1nano 比較快比較不需要智商的
        使用 chat.completions API透過gpt-4.1-nano的自定義回答"單輪"對話
        
        注意：當 use_ollama=True 時，會自動切換到本地 Ollama 模型

        usage: full_chat_gpt_41_nano("eval剛剛的內容", "太陽是什麼? 太陽是一個恆星", 0.3)"""
        model = NANO_MODEL
        prompt=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": usr_prompt},
        ]
        top_p = 1
        respond = await self.__call_completions_openai_api(model, prompt, temperature, top_p)
        output = respond.choices[0].message.content
        return output if output is not None else ""

    #=================================================================================

    async def _persona_to_prompt_reification(self, persona_base_info: str)->str:
        # 產生一張0.04usd
        # style: 這邊放prompt嗎? 感覺不整齊
        # head in the middle prompt要補上去
        from module.prompt_loader import get_system_prompt, get_user_prompt
        
        sys_prompt = get_system_prompt("persona_to_prompt_reification")
        usr_prompt = get_user_prompt("persona_to_prompt_reification", persona_base_info=persona_base_info)
        response = await self.full_chat_gpt_41_mini(sys_prompt, usr_prompt, 0.7)
        return response

    async def photo_generate(self, persona_base_info:dict)-> None:
        """把基本資料用4.1mini生成具體化的reification_prompt後，再生成生成大頭照"""
        model = PHOTO_MODEL
        # 將字典轉換為JSON字串傳遞給提示詞
        persona_info_str = json.dumps(persona_base_info, ensure_ascii=False)
        reification_prompt = await self._persona_to_prompt_reification(persona_info_str)
        await self.__call_images_openai_api(model, reification_prompt, ("humanoid/humanoid_database/photo/" + persona_base_info['基本資料']['id'] + ".png"), '1024x1024')

    async def simulate_persona_answer(self, chat_messages: List[Dict[str, str]], model: str = CHAT_MODEL_4O, max_tokens: int = 150) -> str:
        """使用 chat.completions API透過gpt-4o回答問題，相似的function: chat_gpt_4o()
        
        注意：當 use_ollama=True 時，會自動切換到本地 Ollama 模型
        
        Args:
            chat_messages: 對話訊息列表
            model: 使用的模型
            max_tokens: 最大回應長度（預設150，約30-50中文字）
        """
        model = CHAT_MODEL_4O
        prompt = chat_messages
        respond = await self.__call_completions_openai_api(model, prompt, 0.7, 1, max_tokens)
        output = respond.choices[0].message.content.strip()
        return output if output is not None else ""


    async def generate_embedding(self, docs: List[str], embedding_model: str = EMBEDDING_MODEL):
        """使用 embeddings.create 得到 embeddings"""
        respond = await self.__call_embeddings_openai_api(docs, embedding_model)
        return np.array([d.embedding for d in respond.data])

    async def get_embedding_for_text(self, text: str) -> np.ndarray:
        """產生單一文字的embedding"""
        embeddings = await self.__call_embeddings_openai_api([text])
        return embeddings

    async def download_and_convert_to_jpg_async(self, url: str, output_path: str, quality: int = 85):
        # 確保資料夾存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 使用 aiohttp 非同步下載
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    raise Exception(f"圖片下載失敗，狀態碼：{resp.status}")
                content = await resp.read()

        # 使用 PIL 開啟並轉為 RGB，再轉 JPEG
        image = Image.open(BytesIO(content))
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")

        # 儲存為 JPEG，含壓縮
        image.save(output_path, format="JPEG", quality=quality)
        print(f"已儲存壓縮 JPEG 至 {output_path}")
    #=================================================================================

async def unit_test():
    # 23秒
    # 生成PERSONA的圖片
    from module.PERSONA_loader import PersonaLoader
    test_persona = PersonaLoader()
    await test_persona.wait_until_ready()
    test_llm = LLM_responder()
    test_personas = test_persona.get_all_personas()

    sem = asyncio.Semaphore(5) #5個一組

    async def limited_process(persona):
        async with sem:
            await test_llm.photo_generate(persona)

    await asyncio.gather(*(limited_process(test_personas[p]) for p in test_personas)) #p是id list

# ======= 工廠方法 =======
def create_openai_responder(api_key=None):
    """建立 OpenAI LLM responder"""
    return LLM_responder(api_key=api_key, use_ollama=False)

def create_ollama_responder(api_key=None):
    """建立 Ollama LLM responder
    
    Args:
        api_key: OpenAI API 金鑰（用於圖片生成和 embedding，可選）
    """
    return LLM_responder(api_key=api_key, use_ollama=True)

async def test_ollama_integration():
    """測試 Ollama 整合功能"""
    print("🚀 測試 Ollama 整合...")
    
    # 建立 Ollama responder
    llm = create_ollama_responder()
    
    # 測試簡單對話
    try:
        print("🗣️  測試對話功能...")
        response = await llm.full_chat_gpt_4o(
            sys_prompt="你是一個有幫助的AI助手。",
            usr_prompt="請用繁體中文回答：什麼是機器學習？請簡短回答。",
            temperature=0.7
        )
        print(f"✅ Ollama 回應測試成功：{response[:100]}...")
        return True
    except Exception as e:
        print(f"❌ Ollama 對話測試失敗：{e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test-ollama":
        # 測試 Ollama 整合
        asyncio.run(test_ollama_integration())
    else:
        # 原本的測試（生成 persona 照片）
        start = time.time()
        asyncio.run(unit_test())
        end = time.time()
        print(f"\n✅ 全部任務完成，共花費 {end - start:.2f} 秒")