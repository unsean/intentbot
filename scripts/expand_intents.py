"""Expand each intent's example phrases up to TARGET_TEXTS_PER_INTENT
using prefix/suffix paraphrasing. Backs up the original once.

Usage (from project root):
    python scripts/expand_intents.py
    python main.py --train
"""

import json
import random
from pathlib import Path

TARGET_TEXTS_PER_INTENT = 1000

ROOT = Path(__file__).resolve().parent.parent
INTENTS_PATH = ROOT / "data" / "intents.json"
BACKUP_PATH = ROOT / "data" / "intents_backup_original.json"


PREFIXES = [
    "",
    "hey ",
    "hi ",
    "hello ",
    "please ",
    "can you ",
    "could you ",
    "would you ",
    "i want to ",
    "i'd like to ",
]

SUFFIXES = [
    "",
    " please",
    " for me",
    " right now",
    " today",
    " if you can",
    " when you can",
]


def generate_variants(text: str, max_variants: int = 50) -> set:
    """Create simple paraphrased variants"""
    base = text.strip()
    variants = {base}
    for _ in range(max_variants):
        p = random.choice(PREFIXES)
        s = random.choice(SUFFIXES)
        new = f"{p}{base}{s}".strip()
        variants.add(new)
    return variants


def main() -> None:
    if not INTENTS_PATH.exists():
        raise SystemExit(f"intents.json not found at {INTENTS_PATH}")

    with INTENTS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "intents" not in data or not isinstance(data["intents"], list):
        raise SystemExit("intents.json does not have a top-level 'intents' list")

    # Backup original file once
    if not BACKUP_PATH.exists():
        BACKUP_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Backup of original intents.json saved to {BACKUP_PATH}")

    total_before = sum(len(intent.get("text", [])) for intent in data["intents"])

    for intent in data["intents"]:
        texts = intent.get("text", [])
        if not isinstance(texts, list):
            continue

        # Deduplicate existing
        base_texts = list(dict.fromkeys(str(t) for t in texts))
        expanded = set(base_texts)

        # Generate more variants until we reach the target
        no_progress_rounds = 0
        while len(expanded) < TARGET_TEXTS_PER_INTENT and base_texts:
            before = len(expanded)
            seed = random.choice(base_texts)
            for v in generate_variants(seed):
                expanded.add(v)
                if len(expanded) >= TARGET_TEXTS_PER_INTENT:
                    break

            # If we are no longer adding new variants, break out to
            # avoid an endless loop for small intents.
            if len(expanded) == before:
                no_progress_rounds += 1
            else:
                no_progress_rounds = 0

            if no_progress_rounds >= 10:
                break

        intent["text"] = sorted(expanded)

    total_after = sum(len(intent.get("text", [])) for intent in data["intents"])

    INTENTS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Expanded training phrases from {total_before} to {total_after} across all intents.")
    print(f"Approximate total lines in intents.json will now be well into the thousands.")


if __name__ == "__main__":
    main()
