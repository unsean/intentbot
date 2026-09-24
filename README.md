# AI Chatbox

A chatbot built from scratch with PyTorch. Two models work together:

- **Intent classifier** — decides what you mean (166 intents, ~94% accuracy)
- **Transformer** — writes replies word-by-word for open conversation
  (encoder-decoder from *Attention Is All You Need*, d_model 256, 8 heads)

Deterministic rules handle math, games, and remembering your name — a small
model can't compute those, so we don't ask it to.

## Tools (real, live)

- `search for X` / `cari X` — DuckDuckGo instant answers, Wikipedia fallback
- `wikipedia X` — article summaries
- `weather in X` / `cuaca di X` — real conditions via Open-Meteo
- `bitcoin price` / `berapa harga eth` — live prices via CoinGecko
- `what time is it` / `jam berapa` — real date & time

All free APIs, no keys needed. Tool commands run deterministically so they
always hit the right function.

## Context

The transformer reads the previous turn — its input is
`intent | prev_user | prev_bot | message`, and it's trained on real
multi-turn dialogues (`data/conversations.json`) plus auto-wrapped context
pairs, so replies condition on what was just said, not just the last message.

The bot also remembers what it asked you — if a tool needs an argument it
asks, and your next message is the answer:

```
You: weather
Bot: Which city? Try 'Jakarta'.
You: tokyo          ← understood as the pending answer
Bot: Tokyo, Japan: 19.1C, mostly clear, ...
```

Same flow works for `search`, `wikipedia`, and `crypto price`. Short
answers (≤2 words) feed the pending question; real commands interrupt it.
Persistent memory covers your name and the bot's name across restarts.

## Thinking

Every reply shows a `(thinking)` line above the answer:

- **Generative routes** — the transformer literally writes the thought:
  every training target is `[think] reasoning [answer] reply`, so the
  model emits reasoning first, then the answer. Context-conditioned
  thoughts reference the previous turn ("earlier they said X, now Y").
- **Deterministic routes** (math, names, tools, games) — the thought is
  the real routing reason ("this is arithmetic - computing exactly").
  Honest pipeline state, not fake reasoning.

`python main.py --think` additionally prints the full decision trace:
which rule fired, classifier top-3 intents, which route answered.

## More data

`scripts/gen_conversations.py` generates multi-turn dialogues
combinatorially (topics x phrasings x advice scenarios) into
`data/conversations_gen.json` — rerun it after editing the templates to
grow the context dataset. Both conversation files feed the transformer.

## Real LLM backend (optional)

The transformer is trained from scratch — for real GPT-quality replies,
point the bot at any OpenAI-compatible server (Ollama, LM Studio,
llama.cpp, or hosted):

```bash
# example: Ollama running llama3.2 locally
set AICHAT_LLM_URL=http://localhost:11434/v1
set AICHAT_LLM_MODEL=llama3.2
python main.py
```

When configured and reachable, open-domain replies go to the LLM (with
real chat history); rules, tools, and the fallback transformer still run
locally. Unset it and the bot is fully offline again.

## Run it

```bash
pip install -r requirements.txt
python main.py            # CLI — trains on first run, then chat
python web.py             # web UI at http://localhost:5000
```

Options:

```bash
python main.py --train    # force retrain both models
python main.py --think    # show the decision trace too
python main.py --no-gen   # classifier only, no generation
python -m pytest tests/   # test suite
```

## How a message flows

```
you type
   │
   ├─ rules        → math, names, game state, help, reset
   ├─ classifier   → intent + confidence
   │      ├─ confident + canned intent → templated response
   │      └─ unsure or open-domain     → transformer generates a reply
   └─ profile      → your name / bot name saved between runs
```

Generated replies stream token-by-token in both the CLI and the web UI.

## Files

```
main.py                  # assistant logic, classifier, CLI
generative.py            # transformer: vocab, training, beam search, streaming
tools.py                 # live tools: search, wikipedia, weather, crypto, time
llm.py                   # optional OpenAI-compatible LLM backend
data/conversations.json      # hand-written multi-turn dialogues
data/conversations_gen.json  # generated multi-turn dialogues (script)
web.py                   # Flask app (POST /chat, SSE /chat/stream, /status)
templates/index.html     # dark web UI with live streaming
data/intents.json        # core intents
data/intents_extra.json  # casual chat (idk, lol, wyd, ...)
data/intents_topics.json # deeper topics + Indonesian casual
data/intents_tools.json  # tool intents (search, weather, crypto, datetime)
data/train.txt           # extra labelled examples
scripts/                 # dataset utilities
tests/                   # pytest suite
```

## Retraining

Edit the `data/*.json` files or `train.txt` — the bot detects the change
and retrains automatically on next launch. Or force it with `--train`.

## Honest note

The transformer is ~18M parameters trained from scratch on 60k local
pairs — no pretrained weights. It writes real generated text and stays
on-topic, but expect short, occasionally awkward replies. This project is
about implementing the full pipeline (attention, masking, teacher forcing,
beam search), not competing with a large LM.
