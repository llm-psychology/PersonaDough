# humanoid_generator.py
import random
import json
from faker import Faker # update with more accurate zh-TW data
from module.LLM_responder import LLM_responder
import uuid
import asyncio
import aiofiles
import time

# ========================================================================================
NUM_GENERATE = 10
# ========================================================================================

class BaseInfoGenerator:
    """基礎資料生成器"""
    def __init__(self):
        self.fake = Faker('zh_TW')

    def _generate_uuid(self) -> str:
        """生成10位16進位UUID"""
        # 生成完整UUID並轉換為16進位
        full_uuid = uuid.uuid4().hex
        # 取前10位
        return full_uuid[:10]

    def generate(self):
        """生成基本個人資料"""
        gender = self.fake.random_element(["男", "女"])
        if gender == "男":
            name = self.fake.name_male()
        else:
            name = self.fake.name_female()

        return {
            "id": self._generate_uuid(),  # 添加UUID
            "姓名": name,
            "年紀": self.fake.random_int(18, 65),
            "性別": gender,
            "出生地": self.fake.city()
        }
    
# ========================================================================================

class AttributeInjector:
    """人格屬性注入器"""
    def __init__(self):
        self.attributes = {
            "人格特質": [
                "情感豐富", "適應力強", "善於調適", "冒險精神", "利他主義", "野心勃勃", "中向人格", "善於分析", "平易近人", "傲慢自大", "口齒伶俐", "藝術氣質", "果斷自信", "真實誠懇", "威嚴穩重", "平衡穩健", "大膽無畏", "勇敢堅毅", "沉著冷靜", "能力出眾", "無憂無慮", "關懷他人", "魅力非凡", "樂善好施", "風度翩翩", "騎士精神", "擅長合作", "全心投入", "善於溝通", "富有同情心", "競爭意識", "性格複雜", "鎮定自若", "充滿自信", "衝突調解", "盡責可靠", "體貼周到", "深思熟慮", "樂於配合", "勇氣可嘉", "彬彬有禮", "創意無限", "好奇心強", "狡詐欺騙", "決斷果敢", "奉獻精神", "謹慎行事", "值得信賴", "意志堅定", "勤奮努力", "外交手腕", "洞察力強", "自律嚴謹", "謹言慎行", "目標明確", "情緒化", "情商高超", "同理心強", "賦能他人", "鼓舞人心", "精力充沛", "進取精神", "熱情洋溢", "道德高尚", "善於表達", "活力四射", "無所畏懼", "靈活變通", "專注力強", "寬宏大量", "坦率直言", "自由奔放", "友善親切", "慷慨大方", "溫柔體貼", "真誠可靠", "善於傾聽", "性情溫和", "親切有禮", "交際廣泛", "心懷感恩", "腳踏實地", "勤勞刻苦", "誠實守信", "謙遜有禮", "幽默風趣", "理想主義", "愚昧無知", "想像力豐富", "公正無私", "獨立自主", "勤奮刻苦", "創新思維", "求知若渴", "見解獨到", "鼓舞人心", "激勵他人", "正直誠信", "聰明睿智", "內向性格", "直覺敏銳", "發明創造", "歡欣喜悅", "公正嚴明", "善良仁慈", "知識淵博", "領導才能", "傾聽者", "邏輯清晰", "忠誠可靠", "慾望強烈", "成熟穩重", "指導他人", "有條不紊", "心思縝密", "謙虛低調", "激勵人心", "負面消極", "談判專家", "培育養育", "客觀公正", "觀察入微", "思想開明", "樂觀向上", "有條有理", "原創精神", "外向活潑", "心直口快", "熱情洋溢", "耐心十足", "洞察敏銳", "堅持不懈", "說服力強", "哲學思考", "玩心十足", "彬彬有禮", "實際務實", "原則性強", "主動積極", "善解難題", "進步思想", "保護慾強", "理性思考", "現實主義", "反思自省", "可靠穩重", "含蓄內斂", "適應力強", "足智多謀", "尊重他人", "敢於冒險", "浪漫情懷", "自信滿滿", "自我認知", "自律嚴格", "自力更生", "無私奉獻", "敏感細膩", "真摯誠懇", "善於交際", "精神抖擻", "隨性而為", "堅定不移", "堅忍克己", "策略思考", "意志堅強", "支持鼓勵", "同情理解", "圓滑得體", "團隊精神", "堅韌不拔", "體貼入微", "寬容大度", "有害人格", "透明公開", "信任他人", "容易信任", "值得信賴", "謙遜質樸", "不循常規", "善解人意", "獨特個性", "不裝腔作勢", "英勇無畏", "多才多藝", "活力充沛", "遠見卓識", "熱心腸", "機智風趣"
            ],
            "社交能力": [
                "適應力強", "平易近人", "果斷自信", "專注傾聽", "關懷他人", "魅力非凡", "指導能力", "擅長合作", "善於溝通", "富有同情心", "充滿自信", "衝突調解", "體貼周到", "樂於配合", "彬彬有禮", "創意解難", "值得信賴", "外交手腕", "情緒智商", "同理心強", "鼓舞人心", "善於互動", "熱情洋溢", "靈活變通", "友善親切", "慷慨大方", "善於傾聽", "樂於助人", "誠實守信", "創新思維", "鼓舞人心", "內向性格", "直覺敏銳", "領導才能", "傾聽者", "指導他人", "激勵人心", "談判專家", "培育養育", "觀察入微", "思想開明", "樂觀向上", "外向活潑", "耐心十足", "說服力強", "彬彬有禮", "積極正面", "主動積極", "善解難題", "可靠穩重", "適應力強", "足智多謀", "尊重他人", "自我認知", "敏感細膩", "善於交際", "支持鼓勵", "圓滑得體", "教學能力", "團隊精神", "寬容大度", "透明公開", "值得信賴", "善解人意"
            ],
            "能力屬性": [
                "適應能力", "注重細節", "戰鬥技能", "溝通能力", "衝突解決", "創造力", "批判性思維", "決策能力", "外交手腕", "情緒智商", "創業精神", "外語能力", "領導才能", "多工處理", "談判技巧", "組織能力", "解決問題", "抗壓能力", "隨機應變", "自我激勵", "戰略規劃", "戰略思維", "壓力管理", "生存技能", "團隊合作", "專業技能", "時間管理"
            ]
        }
    
    def inject(self, base_info):
        """為基本資料注入隨機人格屬性"""
        return {
            "基本資料": base_info,
            "人格屬性": {
                "人格特質": self._random_select("人格特質", 4),
                "社交能力": self._random_select("社交能力", 3),
                "能力屬性": self._random_select("能力屬性", 2)
            }
        }
    
    def _random_select(self, category, max_items):
        """從特定類別中隨機選擇屬性"""
        return random.sample(self.attributes[category], k=random.randint(1, max_items))

# ========================================================================================

class StoryGenerator(LLM_responder):
    """背景故事生成器"""
    
    async def generate(self, character_data):
        """使用 LLM 生成背景故事"""
        # 準備提示詞
        prompt = self._create_story_prompt(character_data)
        
        # 發送 API 請求
        response = await self.full_chat_gpt_4o(usr_prompt = prompt, temperature = 0.8)
        
        return response
    
    def _create_story_prompt(self, character_data):
        """創建故事生成的提示詞"""
        base_info = character_data["基本資料"]
        traits = character_data["人格屬性"]["人格特質"]
        social = character_data["人格屬性"]["社交能力"]
        abilities = character_data["人格屬性"]["能力屬性"]
        
        prompt = f"""
        根據以下個人資料與人格特質，生成詳細背景故事：
        基本資料: {json.dumps(base_info, ensure_ascii=False)}

        人格特點：
        - 主要特質：{', '.join(traits)}
        - 社交風格：{', '.join(social)}
        - 特殊能力：{', '.join(abilities)}

        要求：
        1. 包含童年經歷、重大人生事件、教育背景、家庭背景、對話模式
        2. 說明職業選擇與興趣愛好
        3. 字數500字或以上
        4. 使用臺灣在地化用語
        5. 年齡和人生經歷須符合
        """
        return prompt

# ========================================================================================

class ToneGenerator(LLM_responder):
    """語調說話方式生成器"""
    
    async def generate(self, full_character_data):
        """使用 LLM 生成語調說話方式 gpt-4o gpt-4.1都可以"""
        
        prefix = "你是一位語言與人格建模專家。根據以下人物的基本資料，你要推論出這個人日常說話時的語氣特徵、常見詞彙風格、語調節奏，並描述其語言風格：\n基本資料："
        usr = "用一段小短文，不要列點的，用1000字以內，具體描述這個人說話的語氣、節奏、常用語、說話方式等，可以提出範例句子。避免太抽象。"
        # 發送 API 請求
        response = await self.full_chat_gpt_4o(prefix + json.dumps(full_character_data, ensure_ascii=False), usr, 0.7)
        
        return response
    
# ========================================================================================

class SummarizedBehaviorGenerator(LLM_responder):
    """簡化行為生成器 - 使用 GPT-4.1-mini 進行故事和語言行為的濃縮"""
    
    async def generate(self, character_data):
        """使用 GPT-4.1-mini 生成簡化的行為描述"""
        
        life_story = character_data.get("生平故事", "")
        language_behavior = character_data.get("語言行為", "")
        
        sys_prompt = "你是一位專業的人物特徵摘要專家。請將提供的人物生平和語言行為進行精簡濃縮，保留關鍵特徵和核心行為模式。"
        usr_prompt = f"""
        將以下人物的生平故事和語言行為濃縮為一段簡潔的描述，字數控制在200字以內：
        
        【生平故事】：
        {life_story}
        
        【語言行為】：
        {language_behavior}
        
        將核心的生平經歷和說話特點濃縮為一段簡短文字，使讀者能快速理解該人物的本質。
        """
        
        response = await self.full_chat_gpt_41_mini(sys_prompt, usr_prompt, 0.7)
        return response

# ========================================================================================

class AIParameterAnalyzer(LLM_responder):
    """AI參數分析器 - 使用 GPT-4.1-nano 分析生平故事，判斷適合的 top_p 和 temperature 值"""
    
    async def analyze(self, life_story):
        """分析生平故事並建議 AI 參數"""
        
        sys_prompt = "你是一位專業的 AI 參數優化專家。請分析提供的人物生平故事，並根據內容的複雜性、創造性和一致性需求，建議最適合的 top_p 和 temperature 參數值。"
        usr_prompt = f"""
        分析以下人物生平故事，並根據內容特點提供建議的 AI 參數：
        
        【生平故事】：
        {life_story}
        temperature// 從 0.1 到 0.9 的值，如果人格特質內有複雜/創意內容推薦較高值，精確/事實內容推薦較低值
        top_p// 從 0.1 到 1.0 的值，根據內容的多樣性需求決定

        只回覆以下格式的 JSON 內容：
        {{
          "temperature": 0.X, 
          "top_p": 0.X, 
          "reason": "簡短的推薦理由（50字以內）"
        }}
        """
        
        response = await self.full_chat_gpt_41_nano(sys_prompt, usr_prompt, 0.5)
        
        # 嘗試解析 JSON 回應
        try:
            # 在回應中尋找 JSON 格式的內容
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group(0)
                params = json.loads(json_str)
                return params
            else:
                # 如果無法找到 JSON，提供默認值
                return {"temperature": 0.7, "top_p": 0.9, "reason": "默認值（無法從回應中解析 JSON）"}
        except Exception as e:
            # 解析失敗時提供默認值
            return {"temperature": 0.7, "top_p": 0.9, "reason": f"解析錯誤：{str(e)}"}

# ========================================================================================

class CharacterGenerator(LLM_responder):
    """人格生成主流程控制"""
    def __init__(self):
        super().__init__()
        self.base_generator = BaseInfoGenerator()
        self.attribute_injector = AttributeInjector()
        self.story_generator = StoryGenerator()
        self.tone_generator = ToneGenerator()
        self.summarized_behavior_generator = SummarizedBehaviorGenerator()
        self.parameter_analyzer = AIParameterAnalyzer()

    async def generate_character_step(self):
        """生成完整角色資料"""
        # 步驟 1: 生成基本資料
        base_info = self.base_generator.generate()
        
        # 步驟 2: 注入人格屬性
        character_data = self.attribute_injector.inject(base_info)
        
        # 步驟 3: 生成背景故事
        story = await self.story_generator.generate(character_data)
        story = story.replace("\n","")
        character_data["生平故事"] = story
        
        # 步驟 4: 生成語言行為
        tone = await self.tone_generator.generate(character_data)
        tone = tone.replace("\n","")
        character_data["語言行為"] = tone

        # 步驟 5: 使用 GPT-4.1-mini 濃縮生平故事和語言行為
        summarized_behavior = await self.summarized_behavior_generator.generate(character_data)
        summarized_behavior = summarized_behavior.replace("\n","")
        character_data["簡化行為"] = summarized_behavior
        
        # 步驟 6: 使用 GPT-4.1-nano 分析生平故事，判斷適合的 AI 參數
        ai_params = await self.parameter_analyzer.analyze(story)
        character_data["AI參數"] = ai_params

        await self.photo_generate(character_data)
        return character_data
    
    async def save_character(self, character_data, filename=None):
        """將生成的角色資料保存為 JSON 檔案"""
        if filename is None:
            name = character_data["基本資料"]["姓名"]
            id = character_data["基本資料"]["id"]
            filename = f"humanoid/humanoid_database/{id}_{name}.json"
        
        async with aiofiles.open(filename, "w", encoding="utf-8") as f:
            await f.write(json.dumps(character_data, ensure_ascii=False, indent=2))

        return filename


# ========================================================================================

async def generate_a_persona():
    # 建立角色生成器
    generator = CharacterGenerator()
    
    # 生成角色
    print("開始生成角色...")
    character = await generator.generate_character_step()
    
    # 儲存角色資料
    filename = await generator.save_character(character)
    print(f"\n===========================================\n角色生成完成! 已儲存至 {filename}")
    
    # 顯示摘要資訊
    print(f"\n角色摘要:")
    print(f"姓名: {character['基本資料']['姓名']}")
    print(f"性別: {character['基本資料']['性別']}")
    print(f"年齡: {character['基本資料']['年紀']}")
    print(f"人格特質: {', '.join(character['人格屬性']['人格特質'])}")

async def run_all():
    # 並行 coroutine
    # 生成一個 persona 要花費5000token
    # 約10個 persona 花費 40秒
    await asyncio.gather(*(generate_a_persona() for _ in range(NUM_GENERATE)))

if __name__ == "__main__":
    start = time.time()
    asyncio.run(run_all())
    end = time.time()
    print(f"\n✅ 全部任務完成，共花費 {end - start:.2f} 秒")