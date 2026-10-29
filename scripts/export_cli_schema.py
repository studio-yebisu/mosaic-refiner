"""Export the compatibility CLI contract for Rust clap; no generation/setup."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def schema():
    from mosaic_refiner.__main__ import build_parser
    parser = build_parser()
    result = []
    commands = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    for name, sub in commands.choices.items():
        arguments = []
        for action in sub._actions:
            if isinstance(action, argparse._HelpAction):
                continue
            arguments.append({"name": action.dest, "flags": action.option_strings,
                "required": action.required, "boolean": isinstance(action, argparse._StoreTrueAction),
                "append": isinstance(action, argparse._AppendAction), "choices": list(action.choices) if action.choices is not None else None,
                "help": action.help or ""})
        result.append({"name": name, "arguments": arguments})
    return result


if __name__ == "__main__":
    target = ROOT / "native/host/src/cli-schema.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(schema(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
