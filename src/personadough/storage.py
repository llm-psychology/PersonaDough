import os
from pathlib import Path

def persona_directory(path=None):
    return Path(path or os.getenv("PERSONA_DATA_DIR", "humanoid/humanoid_database")).resolve()
