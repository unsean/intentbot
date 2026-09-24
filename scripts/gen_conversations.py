"""Generate multi-turn conversation data combinatorially.

Builds data/conversations_gen.json from scenario templates x topic fills.
Each turn may carry a "t" field — a short reasoning trace the transformer
learns to emit before its reply ("[think] ... [answer] ..." format).

Run: python scripts/gen_conversations.py
"""

import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "conversations_gen.json"
rng = random.Random(7)

# ---------------------------------------------------------------------------
# Topic dialogues: user opens a topic -> bot engages -> user follows up ->
# bot goes deeper -> user asks for a recommendation -> bot recommends.
# ---------------------------------------------------------------------------

TOPICS = [
    ("movies", "movies are great escapism",
     "what genre pulls you in?", "genre says a lot about taste",
     "start with a classic in that genre, then a modern twist"),
    ("music", "music shapes mood like nothing else",
     "what do you listen to?", "taste in music is personal",
     "make a playlist of five songs that fit different moods"),
    ("video games", "games are the most interactive art form",
     "what kind of games do you play?", "play style reveals a lot",
     "try something outside your usual genre - indies surprise people"),
    ("books", "books let you live a hundred lives",
     "fiction or non-fiction?", "reading taste maps curiosity",
     "pick a short book first - finishing builds the habit"),
    ("cooking", "cooking is chemistry you can eat",
     "what cuisine interests you?", "cuisine choice sets the flavor profile",
     "master one dish completely before jumping to the next"),
    ("travel", "travel resets your perspective",
     "mountains, beaches, or cities?", "destination style shows what you need",
     "start with a weekend trip close by - momentum beats perfection"),
    ("programming", "code is the closest thing to magic we have",
     "what language are you learning?", "language choice shapes thinking",
     "build a tiny project you actually want - tutorials fade, projects stick"),
    ("fitness", "movement is the cheapest antidepressant",
     "what do you enjoy - lifting, running, sports?", "enjoyment predicts consistency",
     "pick the thing you'll actually do three times a week"),
    ("photography", "photography is noticing, not equipment",
     "what do you like to shoot?", "subject choice shows what you notice",
     "shoot the same subject ten different ways - constraints teach fast"),
    ("space", "space puts everything in perspective",
     "what fascinates you - planets, black holes, rockets?", "curiosity about scale is natural",
     "look up tonight - the ISS pass times are free to check"),
    ("history", "history is just people making choices under pressure",
     "which era interests you?", "the era you pick shows what questions you carry",
     "start with one person, not one century - stories stick better"),
    ("art", "art is how we say things words can't carry",
     "do you make art or just appreciate it?", "both paths are valid",
     "visit one gallery or draw one sketch - small contact beats big plans"),
    ("psychology", "the mind studying itself is the strangest loop",
     "behavior or brain stuff?", "each lens explains different quirks",
     "notice one habit of yours this week - self-observation is the start"),
    ("food", "food is memory you can taste",
     "sweet, spicy, or comfort food?", "cravings are usually nostalgia",
     "recreate one dish you loved as a kid - taste unlocks memory"),
    ("cars", "cars are engineering you can feel",
     "speed or style?", "preference splits enthusiasts neatly",
     "learn how one system works - engine, brakes, anything concrete"),
    ("football", "football is chess played at sprint speed",
     "do you play or mostly watch?", "players and watchers see different games",
     "watch one match focusing on positioning, not the ball"),
    ("basketball", "basketball is flow state as a sport",
     "pickup games or league watching?", "street ball and pro ball differ",
     "practice free throws - boring reps build real skill"),
    ("fashion", "style is communication without words",
     "what style do you lean toward?", "style is identity in fabric",
     "audit your closet - wear what fits, donate the fantasy self"),
    ("nature", "nature is the original technology",
     "hiking, gardening, or just watching?", "contact level varies by person",
     "spend twenty minutes outside without your phone this week"),
    ("science", "science is organized curiosity",
     "physics, biology, or chemistry?", "each field asks different questions",
     "watch one experiment video and try to predict the outcome first"),
]

OPENERS = [
    "lets talk about {t}", "can we discuss {t}", "i want to chat about {t}",
    "tell me about {t}", "what do you think about {t}", "i love {t}",
    "aku suka {t}", "cerita soal {t} dong",
]
FOLLOWUPS = [
    "why do you say that", "interesting, tell me more", "what makes you think so",
    "hmm i see", "fair enough", "lanjut", "kok gitu", "oh ya",
]
ASK_REC = [
    "any recommendations", "what should i try", "recommend me something",
    "where do i start", "what do you suggest", "kasih rekomendasi dong",
]

# Emotional/advice dialogues: feeling -> probe -> user detail -> advice
SCENARIOS = [
    ("i feel anxious", "anxiety usually has a shape - what's it circling?",
     ["about the future", "about work", "about people", "tentang masa depan", "kerjaan"],
     "future-anxiety wants a plan: write down the three worst-case outcomes and one concrete step against each. It shrinks fast on paper."),
    ("i feel lonely", "loneliness is a signal, not a flaw. When does it hit hardest?",
     ["at night", "on weekends", "when i see friends online", "malam-malam", "pas weekend"],
     "those windows matter - schedule one recurring thing (a call, a club, a class) that lands exactly there."),
    ("im stressed out", "stress usually has a source - workload, people, or money?",
     ["work mostly", "school stuff", "family things", "kerjaan", "keluarga"],
     "name the top stressor specifically, then split it into what you can control today vs what you can't. Act on the first list only."),
    ("i cant focus lately", "focus problems are usually sleep, dopamine, or stress. Which sounds right?",
     ["probably sleep", "too much phone", "stress i think", "hp terus", "kurang tidur"],
     "pick the single biggest suspect and run a one-week experiment on just that. Changing one variable shows you the cause."),
    ("i feel unmotivated", "motivation follows action, not the other way. What's the thing you're avoiding?",
     ["studying", "my project", "exercise", "belajar", "kerjaan"],
     "then do two minutes of it right after this chat - literally two. Starting is the wall; continuing is easy."),
    ("i had a fight with my friend", "fights with friends hurt more than fights with strangers. What was it about?",
     ["a misunderstanding", "they said something mean", "money stuff", "salah paham"],
     "misunderstandings die with a direct conversation. Text them one honest line - not an apology essay, just 'that bothered me, can we talk'."),
    ("im scared of failing", "fear of failure is really fear of judgment - yours or others. Which one?",
     ["mostly mine", "other peoples", "both honestly", "dua-duanya"],
     "write down what you'd actually lose if it failed - usually it's just pride. Then decide if pride is worth never trying."),
]

# ---------------------------------------------------------------------------

def topic_dialogues():
    convos = []
    for topic, opener_b, q, depth_note, rec in TOPICS:
        for o_t, f_t, r_t in zip(
            rng.sample(OPENERS, 4), rng.sample(FOLLOWUPS, 4), rng.sample(ASK_REC, 4)
        ):
            convos.append({"turns": [
                {"u": o_t.format(t=topic),
                 "t": f"user wants to chat about {topic}. open with a hook plus a question.",
                 "b": f"{opener_b.capitalize()}. {q.capitalize()}"},
                {"u": f_t,
                 "t": "they engaged. deepen the topic instead of repeating it.",
                 "b": f"Honestly, {depth_note}. That's the interesting part."},
                {"u": r_t,
                 "t": f"they want a concrete {topic} recommendation now.",
                 "b": f"My take: {rec}."},
            ]})
    return convos


def scenario_dialogues():
    convos = []
    for feeling, probe, details, advice in SCENARIOS:
        for detail in details:
            convos.append({"turns": [
                {"u": feeling,
                 "t": "user shares a feeling. acknowledge it, then probe the cause.",
                 "b": f"That's real. {probe}"},
                {"u": detail,
                 "t": f"they answered: {detail}. now give specific advice, not platitudes.",
                 "b": advice},
            ]})
    return convos


def main():
    convos = topic_dialogues() + scenario_dialogues()
    rng.shuffle(convos)
    OUT.write_text(
        json.dumps({"conversations": convos}, indent=1, ensure_ascii=False),
        encoding="utf-8",
    )
    n_pairs = sum(len(c["turns"]) - 1 for c in convos)
    print(f"wrote {len(convos)} conversations -> {n_pairs} context pairs")


if __name__ == "__main__":
    main()
