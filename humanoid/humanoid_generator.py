import random
import json
from faker import Faker # update with more accurate zh-TW data
from module.LLM_responder import LLM_responder
import uuid
# ========================================================================================


    
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
    
    def generate(self, character_data):
        """使用 LLM 生成背景故事"""
        # 準備提示詞
        prompt = self._create_story_prompt(character_data)
        
        # 發送 API 請求
        response = self.chat_gpt_4o(prompt, 0.8)
        
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
    
    def generate(self, full_character_data):
        """使用 LLM 生成語調說話方式 gpt-4o gpt-4.1都可以"""
        
        prefix = "你是一位語言與人格建模專家。根據以下人物的基本資料，請推論出這個人日常說話時的語氣特徵、常見詞彙風格、語調節奏，並描述其語言風格：\n基本資料："
        usr = "用一段小短文，不要列點的，用1000字以內，具體描述這個人說話的語氣、節奏、常用語、說話方式等，可以提出範例句子。避免太抽象。"
        # 發送 API 請求
        response = self.full_chat_gpt_4o(prefix + json.dumps(full_character_data, ensure_ascii=False), usr, 0.7)
        
        return response
    
# ========================================================================================

class CharacterGenerator:
    """人格生成主流程控制"""
    def __init__(self):
        self.base_generator = BaseInfoGenerator()
        self.attribute_injector = AttributeInjector()
        self.story_generator = StoryGenerator()
        self.tone_generator = ToneGenerator()
    
    def generate_character(self):
        """生成完整角色資料"""
        # 步驟 1: 生成基本資料
        base_info = self.base_generator.generate()
        
        # 步驟 2: 注入人格屬性
        character_data = self.attribute_injector.inject(base_info)
        
        # 步驟 3: 生成背景故事
        story = self.story_generator.generate(character_data).replace("\n","")
        character_data["生平故事"] = story
        
        # 步驟 4: 生成語言行為
        tone = self.tone_generator.generate(character_data).replace("\n","")
        character_data["語言行為"] = tone

        return character_data
    
    def save_character(self, character_data, filename=None):
        """將生成的角色資料保存為 JSON 檔案"""
        if filename is None:
            name = character_data["基本資料"]["姓名"]
            id = character_data["基本資料"]["id"]
            filename = f"humanoid/humanoid_database/{id}_{name}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(character_data, f, ensure_ascii=False, indent=2)
        
        return filename


# ========================================================================================

def main():
    
    # 建立角色生成器
    generator = CharacterGenerator()
    
    # 生成角色
    print("開始生成角色...")
    character = generator.generate_character()
    
    # 儲存角色資料
    filename = generator.save_character(character)
    print(f"角色生成完成! 已儲存至 {filename}")
    
    # 顯示摘要資訊
    print(f"\n角色摘要:")
    print(f"姓名: {character['基本資料']['姓名']}")
    print(f"性別: {character['基本資料']['性別']}")
    print(f"年齡: {character['基本資料']['年紀']}")
    print(f"人格特質: {', '.join(character['人格屬性']['人格特質'])}")

if __name__ == "__main__":
    for _ in range(0,1):
        main()