import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from personadough.contract import validate_persona
from personadough.generator import CharacterGenerator
from personadough.prompts import get_user_prompt
from personadough.runtime.responder import LLM_responder


def test_generation_stops_at_json(tmp_path):
    generator = CharacterGenerator(use_ollama=True, output_dir=tmp_path)
    generator.story_generator.generate = AsyncMock(return_value="story sentinel")
    generator.tone_generator.generate = AsyncMock(return_value="style sentinel")
    generator.summarized_behavior_generator.generate = AsyncMock(return_value="summary sentinel")
    generator.parameter_analyzer.analyze = AsyncMock(side_effect=AssertionError("Not requested"))
    async def run():
        persona = await generator.generate_character_step()
        path = await generator.save_character(persona)
        assert json.loads(open(path, encoding="utf-8").read()) == persona
        assert "AI參數" not in persona
        assert sorted(p.name for p in tmp_path.iterdir()) == [__import__('pathlib').Path(path).name]
        return persona
    persona = asyncio.run(run())
    assert validate_persona(persona) is persona


def test_tone_receives_character():
    assert "CHARACTER_SENTINEL" in get_user_prompt("tone_generator", character_data="CHARACTER_SENTINEL")


def test_template_missing_input_fails():
    with pytest.raises(KeyError):
        get_user_prompt("tone_generator")


@pytest.mark.parametrize("provider,local", [("ollama", True), ("openai", False)])
def test_provider_selection_and_model_routing(monkeypatch, provider, local):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    responder = LLM_responder()
    assert responder.use_ollama is local
    assert responder.wrapper._client is None
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])
    create = AsyncMock(return_value=response)
    responder.wrapper._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert asyncio.run(responder.full_chat_gpt_4o(usr_prompt="question")) == "ok"
    model = create.call_args.kwargs["model"]
    assert model == ("gpt-oss:20b" if local else "gpt-4o")


def test_invalid_provider_is_explicit(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "typo")
    with pytest.raises(ValueError):
        LLM_responder()
