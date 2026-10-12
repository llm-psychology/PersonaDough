from pathlib import Path
import json

class PromptRegistry:
    def __init__(self, path):
        self.prompts = json.loads(Path(path).read_text(encoding="utf-8"))

    def get_prompt(self, prompt_key, template_keys=None, **kwargs):
        template = self.prompts[prompt_key]
        keys = set(template_keys) if template_keys else set(template)
        return {key: value.format(**kwargs) if key in keys and isinstance(value, str)
                else value for key, value in template.items()}

    def get_system_prompt(self, key, **kwargs):
        result = self.get_prompt(key, ["system", "system_template"], **kwargs)
        return result.get("system", result.get("system_template", ""))

    def get_user_prompt(self, key, **kwargs):
        result = self.get_prompt(key, ["user", "user_template", "template"], **kwargs)
        for name in ("user", "user_template", "template"):
            if name in result:
                return result[name]
        raise KeyError(f"No user prompt: {key}")

    def list_available_prompts(self):
        return list(self.prompts)
