"""Add extra high-level topic intents to data/intents.json.

This script appends several new, broader-topic intents (finance,
productivity, study, sports, history, etc.) if they do not already
exist. After running:

    python scripts/add_topics.py

then expand and retrain:

    python scripts/expand_intents.py
    python main.py --train
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INTENTS_PATH = ROOT / "data" / "intents.json"

EXTRA_INTENTS = [
    {
        "intent": "finance_budgeting",
        "text": [
            "How can I create a monthly budget?",
            "I want to start saving money, where do I begin?",
            "Give me tips for controlling my spending.",
            "How much of my income should I save each month?",
            "What are some basic personal finance rules?",
            "How can I track my expenses more effectively?",
        ],
        "responses": [
            "I can share general budgeting tips like tracking expenses, setting savings goals, and separating needs from wants.",
            "A simple approach is the 50/30/20 rule: needs, wants, and savings. I can talk more about that if you want.",
            "Start by listing your income and all your regular expenses, then see where you can reduce spending.",
        ],
        "entities": [],
    },
    {
        "intent": "productivity_tips",
        "text": [
            "How can I be more productive each day?",
            "Give me tips to stop procrastinating.",
            "I need help focusing on my work.",
            "How do I manage my time better?",
            "What are some good productivity techniques?",
        ],
        "responses": [
            "Common productivity tips include breaking tasks into smaller steps, using a to-do list, and limiting distractions.",
            "Techniques like the Pomodoro method, time blocking, and setting clear priorities can really help.",
            "Sometimes productivity is about rest too—taking short breaks can improve focus.",
        ],
        "entities": [],
    },
    {
        "intent": "study_habits",
        "text": [
            "How can I study more effectively?",
            "Give me tips for exam preparation.",
            "I have trouble remembering what I study.",
            "What are some good study habits?",
            "How do I stop cramming at the last minute?",
        ],
        "responses": [
            "Active recall, spaced repetition, and teaching the material to someone else are powerful study methods.",
            "Creating a study schedule and breaking topics into small chunks can make learning easier.",
            "Try to study in a quiet space, remove distractions, and take regular breaks to avoid burnout.",
        ],
        "entities": [],
    },
    {
        "intent": "sports_discussion",
        "text": [
            "Let's talk about sports.",
            "Who is your favorite athlete?",
            "Do you follow football or soccer?",
            "Tell me something interesting about basketball.",
            "I want to chat about the latest sports news.",
        ],
        "responses": [
            "I can chat about many sports—football, basketball, soccer, and more. What do you like most?",
            "Sports can be exciting to follow! Do you have a favorite team or athlete?",
            "We can talk about memorable games, players, or strategies if you like.",
        ],
        "entities": [],
    },
    {
        "intent": "history_questions",
        "text": [
            "Tell me something interesting about history.",
            "Who are some important historical figures?",
            "What big events changed the world?",
            "I want to learn a bit of world history.",
            "Share a cool history fact with me.",
        ],
        "responses": [
            "History is full of fascinating events and people. Is there a time period you're curious about?",
            "We can talk about ancient civilizations, world wars, inventions, and more.",
            "I can share general historical information, but remember I'm not a perfect replacement for a history textbook.",
        ],
        "entities": [],
    },
    {
        "intent": "language_learning",
        "text": [
            "Give me tips for learning a new language.",
            "How can I improve my English skills?",
            "What is the best way to practice speaking a foreign language?",
            "I want to start learning Spanish, where should I begin?",
        ],
        "responses": [
            "Consistent practice, listening, speaking, and using the language in context are key to learning.",
            "You can combine apps, reading, watching videos, and talking with native speakers if possible.",
            "Setting small daily goals and building habits is often more effective than cramming.",
        ],
        "entities": [],
    },
    {
        "intent": "small_talk_general",
        "text": [
            "Let's just chat about anything.",
            "I feel like having a casual conversation.",
            "Talk with me about random things.",
            "Let's have some small talk.",
            "I just want to talk for a bit.",
        ],
        "responses": [
            "Sure, we can chat about whatever you'd like. What's on your mind?",
            "I'm here to talk! Tell me about your day or anything you're thinking about.",
            "Casual conversation is welcome. How are you feeling right now?",
        ],
        "entities": [],
    },
    {
        "intent": "daily_routine_chat",
        "text": [
            "Ask me about my day.",
            "Talk with me about my daily routine.",
            "I want to share what I did today.",
            "Chat about how my day has been.",
        ],
        "responses": [
            "I'd love to hear about your day. What have you been up to?",
            "Tell me about your routine today—anything interesting happen?",
            "How has your day been going so far?",
        ],
        "entities": [],
    },
    {
        "intent": "feelings_mood_checkin",
        "text": [
            "Talk with me about how I'm feeling.",
            "Ask me about my mood.",
            "I want to share my feelings.",
            "Check in on how I'm doing emotionally.",
        ],
        "responses": [
            "I'm here to listen. How are you feeling right now?",
            "Your feelings matter. Do you want to talk about what's on your mind?",
            "We can talk about your mood—good or bad. I'm here for you.",
        ],
        "entities": [],
    },
    {
        "intent": "work_career_chat",
        "text": [
            "I want to talk about my job.",
            "Chat with me about work stress.",
            "Discuss my career plans.",
            "I need to vent about my workplace.",
        ],
        "responses": [
            "Work can be challenging. What's going on with your job?",
            "We can talk about your career plans or any stress you're dealing with at work.",
            "I'm here to listen if you want to share how work has been lately.",
        ],
        "entities": [],
    },
    {
        "intent": "school_study_chat",
        "text": [
            "I want to talk about school.",
            "Chat with me about my classes.",
            "Discuss my study load.",
            "I feel stressed about exams.",
        ],
        "responses": [
            "School can be a lot to handle. What are you studying right now?",
            "We can talk about your classes, homework, or exam stress if you like.",
            "Tell me what's been hardest about school lately.",
        ],
        "entities": [],
    },
    {
        "intent": "hobbies_daily_chat",
        "text": [
            "I want to talk about my hobbies.",
            "Ask me what I like to do for fun.",
            "Chat about my interests.",
            "Let's talk about free time activities.",
        ],
        "responses": [
            "Hobbies are a big part of life. What do you enjoy doing for fun?",
            "Tell me about your interests—games, art, sports, anything.",
            "We can chat about any activities you like in your free time.",
        ],
        "entities": [],
    },
    {
        "intent": "family_relationships_chat",
        "text": [
            "I want to talk about my family.",
            "Chat with me about family relationships.",
            "Discuss how things are with my parents.",
            "Talk about spending time with family.",
        ],
        "responses": [
            "Family relationships can be complicated. What's been on your mind?",
            "We can talk about your family if you feel like sharing.",
            "How are things going with your family lately?",
        ],
        "entities": [],
    },
    {
        "intent": "friends_social_chat",
        "text": [
            "I want to talk about my friends.",
            "Chat about my social life.",
            "Discuss hanging out with people.",
            "Talk about making new friends.",
        ],
        "responses": [
            "Friends are important. Do you want to talk about your social life?",
            "We can chat about hanging out, making friends, or anything social.",
            "How have things been with your friends recently?",
        ],
        "entities": [],
    },
    {
        "intent": "shopping_chat",
        "text": [
            "I want ideas for things to buy.",
            "Chat about online shopping.",
            "Talk about buying gifts for people.",
            "Discuss choosing between different products.",
        ],
        "responses": [
            "We can talk through what you're thinking of buying and what matters most to you.",
            "Shopping decisions can be tricky. Are you comparing options or just looking for ideas?",
            "Tell me what you're considering buying and we can think it through.",
        ],
        "entities": [],
    },
    {
        "intent": "entertainment_movies_chat",
        "text": [
            "I want to talk about movies.",
            "Chat with me about TV shows.",
            "Discuss what to watch next.",
            "Talk about my favorite films.",
        ],
        "responses": [
            "Movies and shows are fun to talk about. What have you watched lately?",
            "We can chat about genres, recommendations, or your favorite characters.",
            "Tell me about a movie or show you really enjoyed.",
        ],
        "entities": [],
    },
    {
        "intent": "entertainment_games_chat",
        "text": [
            "I want to talk about video games.",
            "Chat with me about gaming.",
            "Discuss my favorite games.",
            "Talk about what I'm playing right now.",
        ],
        "responses": [
            "Games can be a great way to relax. What have you been playing?",
            "We can chat about genres, favorite titles, or game stories.",
            "Tell me about a game you really like and why.",
        ],
        "entities": [],
    },
    {
        "intent": "news_current_events_chat",
        "text": [
            "I want to talk about current events.",
            "Chat with me about the news.",
            "Discuss what's happening in the world.",
            "Talk about recent events and headlines.",
        ],
        "responses": [
            "We can talk in general terms about news and events, but I don't have live updates.",
            "Current events can be intense. Is there a particular topic you want to discuss?",
            "Tell me what you've heard recently and we can chat about it.",
        ],
        "entities": [],
    },
    {
        "intent": "tech_help_basic",
        "text": [
            "I need help understanding some basic tech stuff.",
            "Explain a simple computer concept to me.",
            "Talk about how something works in technology.",
            "Help me with a basic tech question.",
        ],
        "responses": [
            "I can try to explain technology in simple terms. What are you curious about?",
            "Ask me about a tech topic and I'll do my best to break it down.",
            "We can go step by step through any basic tech question you have.",
        ],
        "entities": [],
    },
    {
        "intent": "cooking_daily_chat",
        "text": [
            "I want to talk about cooking.",
            "Chat with me about recipes.",
            "Discuss what I could cook tonight.",
            "Talk about simple meals I can make.",
        ],
        "responses": [
            "Cooking can be fun! What ingredients do you have or what do you feel like eating?",
            "We can chat about easy recipes or meal ideas.",
            "Tell me what kind of food you like and we can think of something to cook.",
        ],
        "entities": [],
    },
    {
        "intent": "sleep_health_chat",
        "text": [
            "I want to talk about my sleep.",
            "Chat with me about sleeping better.",
            "Discuss my sleep schedule.",
            "Talk about feeling tired all the time.",
        ],
        "responses": [
            "Sleep habits can really affect how you feel. What's your sleep like lately?",
            "We can talk about general tips for better sleep, though I'm not a medical professional.",
            "Tell me about your routine around bedtime and waking up.",
        ],
        "entities": [],
    },
    {
        "intent": "motivation_encouragement_chat",
        "text": [
            "I need some motivation.",
            "Encourage me to keep going.",
            "Help me feel more positive.",
            "Give me a bit of encouragement.",
        ],
        "responses": [
            "You're doing better than you think. Want to tell me what's been challenging?",
            "It's okay to struggle—progress is still progress, even if it's small.",
            "I'm here to cheer you on. What goal are you working toward right now?",
        ],
        "entities": [],
    },
    {
        "intent": "habit_building_plan",
        "text": [
            "Help me build a new habit.",
            "I want to build a habit of exercising.",
            "I want to get into the habit of reading every day.",
            "Help me create a habit of coding regularly.",
            "I want to start a meditation habit.",
        ],
        "responses": [
            "We can break your habit into small, repeatable steps and attach it to your daily routine.",
            "Habits grow from small, consistent actions. Let's think about what you want to do and how often.",
            "Tell me what habit you want to build and how often you want to do it.",
        ],
        "entities": [
            {
                "entity": "HABIT",
                "patterns": [
                    "exercise",
                    "work out",
                    "running",
                    "reading",
                    "coding",
                    "programming",
                    "meditation",
                    "journaling",
                    "language practice"
                ],
            },
            {
                "entity": "FREQUENCY",
                "patterns": [
                    "every day",
                    "everyday",
                    "daily",
                    "each morning",
                    "in the morning",
                    "at night",
                    "three times a week",
                    "3 times a week",
                    "once a week",
                    "on weekends"
                ],
            },
        ],
    },
    {
        "intent": "goal_setting_planner",
        "text": [
            "Help me plan my goals.",
            "I want to set goals for this year.",
            "Help me set some study and career goals.",
            "I want to plan health and fitness goals.",
            "Help me break down my long term goals.",
        ],
        "responses": [
            "We can turn your ideas into clear goals with timeframes and small steps.",
            "Let's choose one area of life and create simple, realistic goals for it.",
            "Tell me what you want to improve, and we can plan goals around it.",
        ],
        "entities": [
            {
                "entity": "GOAL_AREA",
                "patterns": [
                    "career",
                    "job",
                    "work",
                    "health",
                    "fitness",
                    "money",
                    "finances",
                    "study",
                    "school",
                    "relationships",
                    "family",
                    "friends"
                ],
            },
            {
                "entity": "TIMEFRAME",
                "patterns": [
                    "this week",
                    "this month",
                    "this year",
                    "next week",
                    "next month",
                    "next year",
                    "in 3 months",
                    "in six months",
                    "long term",
                    "short term"
                ],
            },
        ],
    },
    {
        "intent": "travel_itinerary_planner",
        "text": [
            "Help me plan a short trip.",
            "Plan a weekend trip to a city.",
            "I want to plan a beach vacation.",
            "Help me plan a travel itinerary for a week.",
            "Suggest a simple travel plan on a budget.",
        ],
        "responses": [
            "We can think about destination, time, and budget to build a simple itinerary.",
            "Tell me roughly where you want to go, for how long, and your budget level.",
            "Let's outline days, activities, and rest time for your trip.",
        ],
        "entities": [
            {
                "entity": "DESTINATION_TYPE",
                "patterns": [
                    "city",
                    "beach",
                    "mountain",
                    "countryside",
                    "island"
                ],
            },
            {
                "entity": "REGION",
                "patterns": [
                    "europe",
                    "asia",
                    "america",
                    "africa",
                    "australia"
                ],
            },
            {
                "entity": "TRAVEL_BUDGET",
                "patterns": [
                    "cheap",
                    "budget",
                    "affordable",
                    "mid-range",
                    "luxury"
                ],
            },
        ],
    },
    {
        "intent": "learning_programming_chat",
        "text": [
            "I want to learn programming.",
            "Help me start learning Python.",
            "How do I begin with web development?",
            "Give me advice for learning to code as a beginner.",
            "I want to improve my programming skills.",
        ],
        "responses": [
            "We can talk about languages, learning resources, and small practice projects.",
            "Tell me what programming language or area interests you, and your current level.",
            "Starting with clear practice goals and small projects can help you learn to code.",
        ],
        "entities": [
            {
                "entity": "LANGUAGE_NAME",
                "patterns": [
                    "python",
                    "java",
                    "javascript",
                    "c++",
                    "c#",
                    "go",
                    "rust",
                    "html",
                    "css",
                    "sql"
                ],
            },
            {
                "entity": "EXPERIENCE_LEVEL",
                "patterns": [
                    "beginner",
                    "just starting",
                    "new to this",
                    "intermediate",
                    "advanced"
                ],
            },
        ],
    },
    {
        "intent": "finance_saving_plan",
        "text": [
            "Help me plan my savings.",
            "I want to save money for an emergency fund.",
            "Help me create a savings plan for a big purchase.",
            "How can I save more each month on my salary?",
            "Help me choose a savings goal and plan.",
        ],
        "responses": [
            "We can look at your goals, timeframes, and regular income to shape a savings plan.",
            "Tell me what you want to save for and roughly how long you have.",
            "Small, regular savings can add up over time. Let's think about a realistic amount.",
        ],
        "entities": [
            {
                "entity": "SAVING_GOAL",
                "patterns": [
                    "emergency fund",
                    "house",
                    "home",
                    "car",
                    "vacation",
                    "trip",
                    "retirement",
                    "wedding"
                ],
            },
            {
                "entity": "CURRENCY",
                "patterns": [
                    "dollar",
                    "dollars",
                    "euro",
                    "euros",
                    "yen",
                    "rupiah",
                    "pound",
                    "pounds"
                ],
            },
            {
                "entity": "TIMEFRAME",
                "patterns": [
                    "this year",
                    "this month",
                    "next year",
                    "next few months",
                    "over the next year",
                    "over the next few years"
                ],
            },
        ],
    },
]


def main() -> None:
    if not INTENTS_PATH.exists():
        raise SystemExit(f"intents.json not found at {INTENTS_PATH}")

    with INTENTS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "intents" not in data or not isinstance(data["intents"], list):
        raise SystemExit("intents.json does not have a top-level 'intents' list")

    existing = {intent.get("intent") for intent in data["intents"]}

    added = 0
    for extra in EXTRA_INTENTS:
        tag = extra.get("intent")
        if not tag or tag in existing:
            continue
        data["intents"].append(extra)
        existing.add(tag)
        added += 1

    if not added:
        print("No new intents added (they may already exist).")
        return

    INTENTS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Added {added} new high-level topic intents to {INTENTS_PATH}.")


if __name__ == "__main__":
    main()
