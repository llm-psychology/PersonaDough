import re

def validate_persona(persona):
    """Validate the existing Chinese JSON contract without changing field names."""
    if not isinstance(persona, dict):
        raise ValueError("Persona must be an object")
    base = persona.get("基本資料", {})
    if not isinstance(base, dict) or not re.fullmatch(r"[a-f0-9]{10}", str(base.get("id", ""))):
        raise ValueError("基本資料.id must be ten lowercase hex characters")
    if not isinstance(base.get("姓名"), str) or not base["姓名"].strip():
        raise ValueError("基本資料.姓名 is required")
    if type(base.get("年紀")) is not int or not 0 <= base["年紀"] <= 120:
        raise ValueError("基本資料.年紀 must be an integer between 0 and 120")
    traits = persona.get("人格屬性", {})
    for field in ("人格特質", "社交能力", "能力屬性"):
        if not isinstance(traits, dict) or not isinstance(traits.get(field), list) or not all(isinstance(t, str) for t in traits[field]):
            raise ValueError(f"人格屬性.{field} must be a string array")
    for field in ("生平故事", "語言行為", "簡化行為"):
        if not isinstance(persona.get(field), str) or not persona[field].strip():
            raise ValueError(f"{field} is required")
    return persona
