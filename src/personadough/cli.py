import argparse
import asyncio
from .generator import CharacterGenerator

def main():
    parser = argparse.ArgumentParser(description="Shape personas and stop at JSON export")
    parser.add_argument("--provider", choices=["openai", "ollama"])
    parser.add_argument("--output-dir")
    parser.add_argument("--count", type=int, default=1)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    async def run():
        generator = CharacterGenerator(use_ollama=None if args.provider is None else args.provider == "ollama", output_dir=args.output_dir)
        for _ in range(args.count):
            persona = await generator.generate_character_step()
            print(await generator.save_character(persona))
    asyncio.run(run())
