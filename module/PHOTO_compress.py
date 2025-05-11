import aiohttp
from PIL import Image
from io import BytesIO
import os

async def download_and_convert_to_jpg_async(url: str, output_path: str, quality: int = 85):
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
