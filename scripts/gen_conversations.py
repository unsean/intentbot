"""Generate multi-turn conversation data combinatorially.

Builds data/conversations_gen.json. Three kinds of dialogues:

1. TOPIC chains - open a topic, follow up, ask for a recommendation.
2. SCENARIO chains - share a feeling, get probed, get specific advice.
3. CONTRASTIVE chains - the SAME follow-up message appears after different
   setups with DIFFERENT replies. This is what actually forces the model
   to read the context instead of pattern-matching the last message.

Turns may carry "t" - a reasoning trace the transformer learns to emit
before replying. Thoughts that name the prior turn teach context use.

Run: python scripts/gen_conversations.py
"""

import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "conversations_gen.json"
rng = random.Random(11)

# ---------------------------------------------------------------------------
# 1. Topic dialogues
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
     "what cuisine interests you?", "cuisine sets the flavor profile",
     "master one dish completely before jumping to the next"),
    ("travel", "travel resets your perspective",
     "mountains, beaches, or cities?", "destination shows what you need",
     "start with a weekend trip close by - momentum beats perfection"),
    ("programming", "code is the closest thing to magic we have",
     "what language are you learning?", "language choice shapes thinking",
     "build a tiny project you actually want - tutorials fade, projects stick"),
    ("fitness", "movement is the cheapest antidepressant",
     "lifting, running, or sports?", "enjoyment predicts consistency",
     "pick the thing you'll actually do three times a week"),
    ("photography", "photography is noticing, not equipment",
     "what do you like to shoot?", "subject choice shows what you notice",
     "shoot the same subject ten different ways - constraints teach fast"),
    ("space", "space puts everything in perspective",
     "planets, black holes, or rockets?", "curiosity about scale is natural",
     "look up tonight - ISS pass times are free to check"),
    ("history", "history is people making choices under pressure",
     "which era interests you?", "the era you pick shows your questions",
     "start with one person, not one century - stories stick"),
    ("art", "art says what words can't carry",
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
     "do you play or mostly watch?", "players and watchers see differently",
     "watch one match focusing on positioning, not the ball"),
    ("basketball", "basketball is flow state as a sport",
     "pickup games or league watching?", "street and pro ball differ",
     "practice free throws - boring reps build real skill"),
    ("fashion", "style is communication without words",
     "what style do you lean toward?", "style is identity in fabric",
     "audit your closet - wear what fits, donate the fantasy self"),
    ("nature", "nature is the original technology",
     "hiking, gardening, or just watching?", "contact level varies",
     "spend twenty minutes outside without your phone this week"),
    ("science", "science is organized curiosity",
     "physics, biology, or chemistry?", "each field asks differently",
     "watch one experiment video and predict the outcome first"),
    ("anime", "anime hits emotional notes western shows skip",
     "shonen, slice of life, or dark stuff?", "subgenre shows your mood",
     "start with a short series - twelve episodes is enough to judge"),
    ("podcasts", "podcasts turn dead time into learning time",
     "true crime, comedy, or educational?", "format choice shapes the habit",
     "pick one show and one commute - consistency over variety"),
    ("meditation", "meditation is attention training, not emptying your mind",
     "tried apps or just silence?", "the method matters less than showing up",
     "try two minutes of counting breaths - shorter than you think works"),
    ("writing", "writing is thinking with the brakes off",
     "journaling, fiction, or essays?", "each form trains a different muscle",
     "write one ugly paragraph daily - editing is where quality lives"),
    ("chess", "chess is a conversation where both sides lie",
     "do you play online or over the board?", "speed formats differ a lot",
     "learn one opening deeply instead of ten shallowly"),
    ("gardening", "gardening is patience you can eat",
     "vegetables, flowers, or houseplants?", "indoor and outdoor differ",
     "start with herbs on a windowsill - hard to kill, quick reward"),
    ("photography gear", "gear matters less than light",
     "phone or dedicated camera?", "the best camera is the charged one",
     "learn your current camera's limits before upgrading"),
    ("personal finance", "money is stored freedom",
     "saving, investing, or budgeting?", "each lever works differently",
     "track spending for one month - awareness comes before control"),
    ("learning languages", "a second language is a second operating system",
     "which language are you into?", "motivation beats method",
     "ten minutes daily beats two hours on sunday"),
    ("startups", "startups are experiments with payroll",
     "building something or just curious?", "doing teaches fastest",
     "talk to five potential users before writing any code"),
    ("ai", "ai is the strangest tool we've built",
     "excited or worried about it?", "both reactions are reasonable",
     "use it on one real task this week - hands-on beats hot takes"),
    ("drawing", "drawing is seeing, not hand skill",
     "digital or paper?", "medium changes the workflow",
     "draw the same object daily for a week - watch yourself improve"),
    ("hiking", "hiking is walking with better views",
     "day hikes or multi-day?", "gear needs scale with length",
     "pick a short local trail first - finish it, then level up"),
    ("cycling", "cycling is meditation at 25 km/h",
     "road, mountain, or commuting?", "terrain decides the bike",
     "check secondhand markets - good bikes depreciate fast"),
    ("sleep habits", "sleep is the lever under everything else",
     "trouble falling or staying asleep?", "the fix differs by problem",
     "same wake time every day first - it anchors the whole cycle"),
]

OPENERS = [
    "lets talk about {t}", "can we discuss {t}", "i want to chat about {t}",
    "tell me about {t}", "what do you think about {t}", "i love {t}",
    "aku suka {t}", "cerita soal {t} dong", "im really into {t}",
    "got any thoughts on {t}",
]
FOLLOWUPS = [
    "why do you say that", "interesting, tell me more", "what makes you think so",
    "hmm i see", "fair enough", "lanjut", "kok gitu", "oh ya",
    "go on", "and then what", "terus gimana",
]
ASK_REC = [
    "any recommendations", "what should i try", "recommend me something",
    "where do i start", "what do you suggest", "kasih rekomendasi dong",
    "what would you pick", "got tips",
]

# ---------------------------------------------------------------------------
# 2. Feeling/advice scenarios - probe -> user detail -> specific advice
# ---------------------------------------------------------------------------

SCENARIOS = [
    ("i feel anxious", "anxiety usually has a shape - what's it circling?",
     ["about the future", "about work", "about people", "tentang masa depan", "kerjaan"],
     "future-anxiety wants a plan: write the three worst outcomes and one step against each. It shrinks on paper."),
    ("i feel lonely", "loneliness is a signal, not a flaw. When does it hit hardest?",
     ["at night", "on weekends", "when i see friends online", "malam-malam", "pas weekend"],
     "those windows matter - schedule one recurring thing (a call, a club, a class) that lands exactly there."),
    ("im stressed out", "stress usually has a source - work, people, or money?",
     ["work mostly", "school stuff", "family things", "kerjaan", "keluarga"],
     "name the top stressor, split it into what you control today vs can't. Act on the first list only."),
    ("i cant focus lately", "focus problems are usually sleep, dopamine, or stress. Which sounds right?",
     ["probably sleep", "too much phone", "stress i think", "hp terus", "kurang tidur"],
     "pick the biggest suspect and run a one-week experiment on just that. One variable shows the cause."),
    ("i feel unmotivated", "motivation follows action, not the other way. What are you avoiding?",
     ["studying", "my project", "exercise", "belajar", "kerjaan"],
     "then do two minutes of it right after this chat - literally two. Starting is the wall; continuing is easy."),
    ("i had a fight with my friend", "fights with friends hurt more than with strangers. What was it about?",
     ["a misunderstanding", "they said something mean", "money stuff", "salah paham"],
     "misunderstandings die with a direct line - not an apology essay, just 'that bothered me, can we talk'."),
    ("im scared of failing", "fear of failure is really fear of judgment - yours or others. Which one?",
     ["mostly mine", "other peoples", "both honestly", "dua-duanya"],
     "write what you'd actually lose if it failed - usually just pride. Is pride worth never trying?"),
    ("i procrastinate a lot", "procrastination is usually avoidance of a feeling, not laziness. What task do you dodge most?",
     ["assignments", "my side project", "emails and replies", "tugas", "project pribadi"],
     "for that one, shrink the first step until it's stupid-easy - open the file, read one page. Momentum does the rest."),
    ("i overthink everything", "overthinking is thinking without a decision point. What's the current loop about?",
     ["relationships", "my choices", "what people think of me", "kata orang", "keputusan"],
     "give the loop a deadline: 'I'll decide by friday.' Undecided thoughts just orbit - a date forces landing."),
    ("i feel behind in life", "behind compared to what - other people's highlight reels?",
     ["my friends are ahead", "social media", "family expectations", "temen-temen udah jauh"],
     "compare yourself to you six months ago instead - that's the only fair baseline. What moved since then?"),
    ("my sleep schedule is ruined", "ruined how - sleeping late, or waking at weird times?",
     ["sleep super late", "wake up at noon", "nap too much", "tidur subuh terus"],
     "fix the wake time first, keep it identical even on weekends - the sleep time follows the anchor."),
    ("i keep comparing myself to others", "comparing up or comparing sideways - friends or strangers?",
     ["friends mostly", "people online", "everyone honestly", "temen sendiri"],
     "mute the comparison sources for two weeks and write down three things you actually did. Data beats feelings."),
    ("i feel burned out", "burnout is a debt - how long has it been building?",
     ["months honestly", "since this semester", "since i started working", "udah lama"],
     "debts get paid with rest plus boundaries, not just a weekend off. What can you drop for two weeks?"),
    ("i miss my ex", "missing them is normal - what do you miss, them or the routine?",
     ["the routine honestly", "them specifically", "not being alone", "rutinitasnya sih"],
     "if it's the routine, rebuild it with yourself or friends first. If it's them - what specifically?"),
    ("im nervous about an interview", "nervous means it matters. Which part scares you most?",
     ["the questions", "freezing up", "being judged", "takut jawab salah"],
     "for freezing: rehearse three stories out loud - not answers, stories. Retrieval beats recall under pressure."),
    ("i failed my exam", "one exam or a pattern? The fix differs.",
     ["just this one", "a few actually", "i keep failing", "baru sekali"],
     "if it's one - review which questions died and why. If it's a pattern - the study method needs changing, not effort."),
    ("i argue with my parents a lot", "what's the recurring theme - control, choices, or communication?",
     ["they dont listen", "too controlling", "my life choices", "terlalu ngatur"],
     "pick the calmest moment this week and say 'i want to tell you something without debate first.' Framing beats volume."),
]

# ---------------------------------------------------------------------------
# 3. Contrastive dialogues - same message, different context, different reply.
# This is the mechanism that forces the model to read context.
# ---------------------------------------------------------------------------

AMBIGUOUS = [
    # (setup_user, setup_bot, ambiguous_followup, contextual_reply, thought)
    ("i love sci-fi movies", "Sci-fi is great - seen anything good lately?",
     "yes", "Which one? I'm always hunting for a good sci-fi watch.",
     "they confirmed seeing sci-fi. ask which title to continue the topic."),
    ("do you like pizza", "I love the idea of pizza. You a fan?",
     "yes", "Thin crust or thick? That's the real dividing line.",
     "they said yes to liking pizza. deepen with a follow-up question."),
    ("are you into kpop", "Kpop has a massive world - do you follow any groups?",
     "yes", "Which group? The fandom details are the fun part.",
     "yes answers the kpop question - ask which group specifically."),
    ("lets talk about travel", "Travel's my favorite subject. Been anywhere recently?",
     "yes", "Where'd you go? Recent trips always have the best stories.",
     "yes means they traveled recently - ask where."),
    ("do you play guitar", "I wish I had fingers for it - do you play?",
     "yes", "How long? Guitar skill is pure time-on-strings.",
     "yes = they play guitar. ask about their experience."),
    ("should i learn python", "Python's a solid first language - any coding background?",
     "yes", "Then you'll pick it up in days - what did you code in before?",
     "yes means they have coding background - ask which language."),
    ("i just finished a book", "Nice - did you like it?",
     "no", "Fair - what killed it for you? Bad endings or slow middle?",
     "no = they didn't like the book. probe why specifically."),
    ("did you watch the match", "Can't watch - was it a good game?",
     "no", "Ouch, a boring one? What happened - bad scoreline?",
     "no = the match wasn't good. ask what made it bad."),
    ("want a movie suggestion", "Always - what have you got?",
     "no wait, game suggestion", "Even better - what platform are you on?",
     "they pivoted from movies to games mid-thought. follow the pivot."),
    ("i work in tech", "Tech's a wide field - which part?",
     "the boring part", "Ha, honest answer. What makes it boring - the meetings or the code?",
     "'the boring part' answers which-part-of-tech - riff on it."),
    ("what should i eat", "Depends - sweet or savory mood?",
     "the first one", "Sweet it is - pancakes, pastries, or straight dessert?",
     "'the first one' = sweet. recommend within that branch."),
    ("coffee or tea", "Tough call - which side are you on?",
     "the second", "Tea person - green, black, or herbal?",
     "'the second' = tea. ask which kind."),
    ("pick a number 1 or 2", "Hmm... I'll say 2. What do I win?",
     "wrong, it was 1", "Rigged game! Rematch - this time I pick 1.",
     "they said my pick was wrong - play along with the bit."),
    ("i have two dogs", "Two! Names?",
     "the older is max", "Max the elder - and the younger one's name?",
     "they gave the older dog's name - ask for the younger's."),
    ("my brother is annoying", "Siblings - built-in chaos. Older or younger brother?",
     "younger", "Younger brothers are legally required to be annoying. What's his latest crime?",
     "younger answers the question - joke within that frame."),
    ("i live with my family", "Big household or small?",
     "five of us", "Five is a full house - are you the oldest kid?",
     "'five of us' answers household size - continue the family thread."),
    ("guess my age", "Hmm... 19? Am I close?",
     "close, im 20", "One year off - I'll take it. Is 20 treating you well?",
     "they corrected my guess to 20 - acknowledge and continue."),
    ("im thinking of two hobbies", "Shoot - what are they?",
     "drawing and coding", "Interesting mix - visual plus logical. Which came first?",
     "they named both hobbies - compare them."),
]


def topic_dialogues():
    convos = []
    for topic, hook, q, depth_note, rec in TOPICS:
        picks = rng.sample(OPENERS, 3) + [None]
        for o_t in picks:
            f_t = rng.choice(FOLLOWUPS)
            r_t = rng.choice(ASK_REC)
            opener = o_t.format(t=topic) if o_t else f"im obsessed with {topic} lately"
            convos.append({"turns": [
                {"u": opener,
                 "t": f"user opens the {topic} topic. hook them, then ask a question.",
                 "b": f"{hook.capitalize()}. {q.capitalize()}"},
                {"u": f_t,
                 "t": f"they want more on {topic}. go deeper, don't repeat the opener.",
                 "b": f"Honestly, {depth_note}. That's the interesting part."},
                {"u": r_t,
                 "t": f"now they want a concrete {topic} recommendation.",
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
                 "t": f"they answered with '{detail}'. give advice specific to that, not generic.",
                 "b": advice},
                {"u": rng.choice(["that actually helps", "hmm maybe", "idk if that works", "bisa juga sih"]),
                 "t": "they're reacting to the advice. validate and add one small push.",
                 "b": "Try it for a week - small experiments, not life overhauls. You can report back."},
            ]})
    return convos


def contrastive_dialogues():
    """Each ambiguous follow-up becomes its own mini-dialogue so the same
    surface text maps to different replies depending on context."""
    convos = []
    for setup_u, setup_b, follow, reply, thought in AMBIGUOUS:
        convos.append({"turns": [
            {"u": setup_u,
             "t": "user opens a thread. engage it.",
             "b": setup_b},
            {"u": follow,
             "t": thought,
             "b": reply},
        ]})
    return convos


def main():
    convos = (
        topic_dialogues() + scenario_dialogues() + contrastive_dialogues()
    )
    rng.shuffle(convos)
    OUT.write_text(
        json.dumps({"conversations": convos}, indent=1, ensure_ascii=False),
        encoding="utf-8",
    )
    n_pairs = sum(len(c["turns"]) - 1 for c in convos)
    print(f"wrote {len(convos)} conversations -> {n_pairs} context pairs")


if __name__ == "__main__":
    main()
