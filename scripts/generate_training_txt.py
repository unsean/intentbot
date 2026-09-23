"""Regenerate data/train.txt with fresh labelled prompts.

This script overwrites train.txt with ~1000 new
intent<TAB>sentence training lines, independent from the
hand-written examples that were there before.

Usage (from project root):
    python scripts/generate_training_txt.py

After running, retrain the model:
    python main.py --train
"""

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
INTENTS_PATH = ROOT / "data" / "intents.json"
EXTRA_PATH = ROOT / "data" / "train.txt"

TARGET_TOTAL = 1000

# Simple prefixes/suffixes to create natural variants
PREFIXES = [
    "",
    "hey ",
    "hi ",
    "hello ",
    "could you ",
    "can you ",
    "please ",
]

SUFFIXES = [
    "",
    " please",
    " for me",
    " right now",
    " today",
]

# Seed phrases per intent; only intents that actually exist in
# intents.json will be used.
SEED_DEFS = {
    "greeting": [
        "say hi to me",
        "greet me nicely",
        "start a friendly conversation",
        "welcome me",
        "say hello in a casual way",
    ],
    "goodbye": [
        "say goodbye in a friendly way",
        "end this chat politely",
        "tell me farewell",
        "wrap up our conversation",
    ],
    "help": [
        "explain how you can assist me",
        "tell me what you are able to do",
        "guide me on how to use this chatbot",
        "help me understand your features",
    ],
    "math_question": [
        "calculate 27 plus 45",
        "work out 123 minus 58",
        "multiply 14 by 19",
        "divide 144 by 12",
        "figure out 3.75 times 8.2",
    ],
    "joke": [
        "tell me a technology joke",
        "share a short funny joke",
        "give me a clever one liner",
        "make me laugh with a silly joke",
    ],
    "weather": [
        "tell me if I need an umbrella",
        "describe today's weather forecast",
        "tell me if it might snow",
        "give me the weather outlook",
    ],
    "time_query": [
        "tell me the current time",
        "let me know what time it is",
        "inform me of the exact time now",
        "check the local time for me",
    ],
    "emotion_support": [
        "respond kindly if I say I feel anxious",
        "comfort me when I'm feeling sad",
        "encourage me when I'm stressed",
        "support me when I'm overwhelmed",
    ],
    "creative_request": [
        "write a tiny story about a robot friend",
        "create a short poem about the stars",
        "invent a fantasy adventure idea",
        "come up with a creative writing prompt",
    ],
    "travel_talk": [
        "suggest interesting places to visit",
        "recommend a travel destination",
        "talk about planning a vacation",
        "give ideas for a weekend trip",
    ],
    "health_fitness": [
        "suggest a beginner workout routine",
        "give tips for staying active",
        "share simple healthy habits",
        "talk about starting an exercise plan",
    ],
    "food_talk": [
        "recommend an easy dinner idea",
        "suggest a healthy snack option",
        "talk about trying a new cuisine",
        "give me meal inspiration",
    ],
    "music_talk": [
        "chat about relaxing music",
        "suggest genres to explore",
        "talk about favorite bands",
        "discuss different music styles",
    ],
    "technology_talk": [
        "explain artificial intelligence in simple words",
        "discuss recent tech trends",
        "talk about how technology changes daily life",
        "explain what machine learning is",
    ],
    "capabilities": [
        "describe everything you can help me with",
        "list your main abilities as a chatbot",
        "explain what kinds of questions you handle",
        "tell me what you are good at doing",
    ],
    "random_question": [
        "share a surprising piece of trivia",
        "tell me an unexpected fun fact",
        "give me a totally random topic to think about",
        "say something interesting and unusual",
    ],
    "reminder_request": [
        "act like I'm asking you to remind me about a task",
        "respond to a request to remember an appointment",
        "handle a message asking not to forget something",
        "react when I ask you to remind me later",
    ],
    "finance_budgeting": [
        "offer tips for creating a monthly budget",
        "explain how to start saving money",
        "talk about controlling everyday spending",
        "share basic personal finance rules",
    ],
    "productivity_tips": [
        "give advice to stop procrastinating",
        "suggest ways to stay focused on work",
        "share daily productivity strategies",
        "talk about managing time better",
    ],
    "study_habits": [
        "suggest effective study techniques",
        "give advice for preparing for exams",
        "talk about remembering what I study",
        "share good long term study habits",
    ],
    "sports_discussion": [
        "start a casual chat about sports",
        "talk about a favorite sports team",
        "discuss an exciting recent match",
        "chat about athletes and competitions",
    ],
    "history_questions": [
        "share an interesting fact from history",
        "talk about an important historical event",
        "mention a famous historical figure",
        "give a short piece of world history info",
    ],
    "language_learning": [
        "give tips for learning a new language",
        "talk about improving English skills",
        "suggest ways to practice speaking",
        "explain how to build a daily language habit",
    ],
}


def main() -> None:
    if not INTENTS_PATH.exists():
        raise SystemExit(f"intents.json not found at {INTENTS_PATH}")

    with INTENTS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "intents" not in data or not isinstance(data["intents"], list):
        raise SystemExit("intents.json does not have a top-level 'intents' list")

    existing_tags = {intent.get("intent") for intent in data["intents"]}

    lines = []

    # Generate labelled prompts, but only for intents that really exist.
    for intent_tag, seeds in SEED_DEFS.items():
        if intent_tag not in existing_tags:
            continue

        for seed in seeds:
            base = seed.strip()
            if not base:
                continue
            for pre in PREFIXES:
                for suf in SUFFIXES:
                    text = f"{pre}{base}{suf}".strip()
                    if not text:
                        continue
                    lines.append(f"{intent_tag}\t{text}")
                    if len(lines) >= TARGET_TOTAL:
                        break
                if len(lines) >= TARGET_TOTAL:
                    break
            if len(lines) >= TARGET_TOTAL:
                break
        if len(lines) >= TARGET_TOTAL:
            break

    # Deduplicate while preserving order
    unique_lines = list(dict.fromkeys(lines))
    if len(unique_lines) > TARGET_TOTAL:
        unique_lines = unique_lines[:TARGET_TOTAL]

    EXTRA_PATH.write_text("\n".join(unique_lines) + "\n", encoding="utf-8")

    print(f"Wrote {len(unique_lines)} labelled prompts to {EXTRA_PATH}.")


if __name__ == "__main__":
    main()
