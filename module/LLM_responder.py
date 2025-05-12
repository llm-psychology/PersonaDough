# LLM_responder.py
import os
from tkinter import NO
from dotenv import load_dotenv
from openai import AsyncOpenAI
from typing import List, Dict
import numpy as np
from deprecated import deprecated
import asyncio
import time
import requests
from module.PHOTO_compress import download_and_convert_to_jpg_async
load_dotenv()

# ======= CONFIG =======
EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL_4O = "gpt-4o"
CHAT_MODEL_41 = "gpt-4.1"
MINI_MODEL = "gpt-4.1-mini"
NANO_MODEL = "gpt-4.1-nano"
PHOTO_MODEL = "dall-e-3" # 只提供url要自己再request
OLD_PHOTO_MODEL = "dall-e-2" # 會自己回傳 但是產生的細節會遺漏

class LLM_responder:
    """LLM api 物件"""  

    def __init__(self, api_key=None):
        """LLM api建構子"""   
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("需要設定 OPENAI_API_KEY 環境變數或直接傳入 API 金鑰")
        
        # 初始化 OpenAI 客戶端
        self.client = AsyncOpenAI(api_key=self.api_key)

    async def __call_completions_openai_api(self, model, messages, temperature = 0.7, top_p = 1):
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

    async def __call_images_openai_api(self, model, prompt: str, output_path: str, photo_size: str) -> None:
        """reification過的prompt轉圖片並下載
        https://platform.openai.com/docs/api-reference/images/create
        """
        response = await self.client.images.generate(
            prompt=prompt,
            n=1,
            size=photo_size,
            model=model,
            quality = "standard",
        )

        image_url = response.data[0].url
        if not image_url:
            raise Exception("圖片 URL 為空，無法下載圖片")
        try:
            await download_and_convert_to_jpg_async(image_url, output_path)
        except Exception as e:
            print(f"下載並轉換圖片時發生錯誤:{e}")

    async def __call_embeddings_openai_api(self, docs, model: str = EMBEDDING_MODEL):
        """使用 OpenAI 套件呼叫 embeddings.create API"""
        respond = await self.client.embeddings.create(
            input=docs,
            model=model
        )
        return respond
    
    #================================================================================================================

    @deprecated(reason="準備移除")
    async def chat_gpt_4o(self, prompt: str, temperature: int)->str:
        """簡單對話-使用 chat.completions API透過gpt-4o的user回答單輪對話"""
        model = CHAT_MODEL_4O
        prompt = [{"role": "user", "content": prompt}]
        respond = await self.__call_completions_openai_api(model, prompt, temperature)
        return respond.choices[0].message.content
    
    #================================================================================================================

    async def full_chat_gpt_4o(self, sys_prompt: str="", usr_prompt: str="", temperature: int=0.7, top_p: int=1)->str:
        """使用 chat.completions API透過gpt-4o的自定義回答"單輪"對話

        usage: full_chat_gpt_4o("你是數學教學小助手，不可以誤人子弟亂回答，要教給五歲小孩聽的", "回答一個題目是: 1+1=多少?", 0.3)"""
        model = CHAT_MODEL_4O
        prompt=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": usr_prompt},
        ]
        respond = await self.__call_completions_openai_api(model, prompt, temperature, top_p)
        return respond.choices[0].message.content
        
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
        respond = await self.__call_completions_openai_api(model, prompt, temperature, top_p)
        return respond.choices[0].message.content
    
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
        respond = await self.__call_completions_openai_api(model, prompt, temperature, top_p)
        return respond.choices[0].message.content
    
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
        respond = await self.__call_completions_openai_api(model, prompt, temperature, top_p)
        return respond.choices[0].message.content

    #=================================================================================

    async def _persona_to_prompt_reification(self, persona_base_info: str)->str:
        # 產生一張0.04usd
        # style: 這邊放prompt嗎? 感覺不整齊
        # head in the middle prompt要補上去
        sys_prompt="""
            根據以下人物資料，生成一段用於圖像 AI 模型（如 DALL·E）的英文 prompt，描述他的證件照風格大頭照，背景為白色，鏡頭直視，專業且清晰，強調是亞洲人。根據年齡、性別、個性特質、語言風格、出生地等資訊，推理出這個人可能的外貌特徵（如微笑與否、眼神神情、髮型風格、服裝偏好、氣質感），並用英文精簡描述。
            輸出格式為：
            "Centered portrait with full head visible, not cropped, ample space above the head. Upper body in frame, symmetrical composition. A professional medium close up headshot of [描述], white background, neutral lighting, studio style, passport photo, high resolution"

            範例輸入：
            {
            "姓名": "林書豪",
            "年紀": 35,
            "性別": "男",
            "出生地": "台北",
            "人格特質": ["溫柔", "熱情"],
            "語言行為": "說話幽默、善於激勵人心"
            }
            範例輸出：
            "a professional headshot of a mid-aged Asian man with a warm smile and confident posture, neat hairstyle, clean shave, wearing a light blue shirt and blazer, looking directly at the camera, white background, neutral lighting, studio style, passport photo, high resolution"
            """
        usr_prompt = f"""這是要具體化的人像數據: {persona_base_info}"""
        response = await self.full_chat_gpt_41_mini(sys_prompt, usr_prompt, 0.7)
        return response

    async def photo_generate(self, persona_base_info:dict)-> None:
        """把基本資料用4.1mini生成具體化的reification_prompt後，再生成生成大頭照"""
        model = PHOTO_MODEL
        reification_prompt = await self._persona_to_prompt_reification(persona_base_info)
        await self.__call_images_openai_api(model, reification_prompt, ("humanoid/humanoid_database/photo/" + persona_base_info['基本資料']['id'] + ".png"), '1024x1024')

    async def simulate_persona_answer(self, chat_messages: List[Dict[str, str]], model: str = CHAT_MODEL_4O) -> str:
        """使用 chat.completions API透過gpt-4o回答問題，相似的function: chat_gpt_4o()"""
        model = CHAT_MODEL_4O
        prompt = chat_messages
        respond = await self.__call_completions_openai_api(model, prompt, 0.7)
        return respond.choices[0].message.content.strip()


    async def generate_embedding(self, docs: List[str], embedding_model: str = EMBEDDING_MODEL):
        """使用 embeddings.create 得到 embeddings"""
        respond = await self.__call_embeddings_openai_api(docs, embedding_model)
        return np.array([d.embedding for d in respond.data])

    async def get_embedding_for_text(self, text: str) -> np.ndarray:
        """產生單一文字的embedding"""
        embeddings = await self.__call_embeddings_openai_api([text])
        return embeddings

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

if __name__ == "__main__":
    # 生成10個persona的照片一共花費
    start = time.time()
    asyncio.run(unit_test())
    end = time.time()
    print(f"\n✅ 全部任務完成，共花費 {end - start:.2f} 秒")