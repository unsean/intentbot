# AI Chatbox

A chatbot built from scratch with PyTorch. Two models work together:

- **Intent classifier** — decides what you mean (166 intents, ~94% accuracy)
- **Transformer** — writes replies word-by-word for open conversation
  (encoder-decoder from *Attention Is All You Need*, d_model 256, 8 heads)

Deterministic rules handle math, games, and remembering your name — a small
model can't compute those, so we don't ask it to.

## Run it

```bash
pip install -r requirements.txt
python main.py            # CLI — trains on first run, then chat
python web.py             # web UI at http://localhost:5000
```

Options:

```bash
python main.py --train    # force retrain both models
python main.py --gen-all  # let the transformer answer everything it can
python main.py --no-gen   # classifier only, no generation
python -m pytest tests/   # 60 tests
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
web.py                   # Flask app (POST /chat, SSE /chat/stream, /status)
templates/index.html     # dark web UI with live streaming
data/intents.json        # core intents
data/intents_extra.json  # casual chat (idk, lol, wyd, ...)
data/intents_topics.json # deeper topics + Indonesian casual
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
