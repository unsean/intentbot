"""Intent-based chatbot: preprocessing, neural classifier, extensions, CLI.

Usage:
    python main.py            # load existing model or train a new one
    python main.py --train    # force retraining
    python main.py --eval     # print validation metrics after training
"""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import logging
import operator
import os
import random
import re
import sys
import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from generative import (
    UNK,
    GenTrainResult,
    GenVocab,
    Seq2SeqTransformer,
    build_pairs,
    clean_reply,
    to_ascii,
    train_generator,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_INTENTS_PATH = DATA_DIR / "intents.json"
EXTRA_INTENTS_PATHS = [
    DATA_DIR / "intents_extra.json",
    DATA_DIR / "intents_topics.json",
]
DEFAULT_EXTRA_TRAINING = DATA_DIR / "train.txt"
MODEL_PATH = ROOT / "enhanced_model.pth"
CONFIG_PATH = ROOT / "enhanced_config.json"
GEN_MODEL_PATH = ROOT / "generative_model.pth"
GEN_CONFIG_PATH = ROOT / "generative_config.json"
LOG_PATH = ROOT / "conversations.jsonl"
PROFILE_PATH = ROOT / "user_profile.json"
CHATBOT_LOG = ROOT / "chatbot.log"

PREPROCESSOR_VERSION = 4
MAX_BIGRAMS = 8000
MAX_INPUT_CHARS = 2000
MAX_TEXTS_PER_INTENT = 200
AUGMENT_PER_TEXT = 2

logger = logging.getLogger("aichatbox")


def _setup_logging() -> None:
    if logger.handlers:
        return
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(CHATBOT_LOG, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)  # keep chat UI clean
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False


def _ensure_nltk_data() -> None:
    for resource in ("corpora/wordnet", "corpora/stopwords"):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split("/", 1)[1], quiet=True)


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_MULTI_SPACE = re.compile(r"\s+")

MATH_TOKEN_MAP = {
    "+": "plus",
    "-": "minus",
    "*": "times",
    "x": "times",
    "/": "over",
    "%": "percent",
    "^": "power",
}


def _normalize_operators(text: str) -> str:
    text = text.replace("**", " power ")
    for char, word in MATH_TOKEN_MAP.items():
        if char in text:
            text = text.replace(char, f" {word} ")
    return text


# --- training-data augmentation --------------------------------------------
# Surface-level variants so the classifier sees casual phrasing, contractions
# and common typos, not only the clean expanded templates in intents.json.

_CONTRACTION_MAP = [
    (r"\bwhat is\b", ["whats", "what's"]),
    (r"\bhow is\b", ["hows", "how's"]),
    (r"\bi am\b", ["im", "i'm"]),
    (r"\bi will\b", ["ill", "i'll"]),
    (r"\bi would\b", ["id", "i'd"]),
    (r"\bi have\b", ["ive", "i've"]),
    (r"\bdo not\b", ["dont", "don't"]),
    (r"\bdoes not\b", ["doesnt", "doesn't"]),
    (r"\bcannot\b", ["cant", "can't"]),
    (r"\bcan not\b", ["cant"]),
    (r"\bcould not\b", ["couldnt", "couldn't"]),
    (r"\bwill not\b", ["wont", "won't"]),
    (r"\bis not\b", ["isnt", "isn't"]),
    (r"\bare not\b", ["arent", "aren't"]),
    (r"\bit is\b", ["its", "it's"]),
    (r"\bthat is\b", ["thats", "that's"]),
    (r"\bwant to\b", ["wanna"]),
    (r"\bgoing to\b", ["gonna"]),
    (r"\bhave to\b", ["gotta", "hafta"]),
    (r"\byou are\b", ["youre", "you're", "ur"]),
    (r"\bplease\b", ["pls", "plz"]),
    (r"\bthanks\b", ["thx", "ty"]),
    (r"\bbecause\b", ["cuz", "cos", "bc"]),
    (r"\breally\b", ["rly"]),
]

_CASUAL_PREFIXES = [
    "", "", "", "so ", "ok ", "ok so ", "hey ", "bro ", "dude ",
    "lol ", "omg ", "tbh ", "ngl ", "honestly ", "lowkey ", "btw ",
]
_CASUAL_SUFFIXES = [
    "", "", "", " pls", " plz", " lol", " tbh", " ngl", " rn",
    " bro", " fr", " tho", " nvm", " asap", " haha",
]


def augment_text(text: str, rng: random.Random, n: int = AUGMENT_PER_TEXT) -> List[str]:
    """Generate up to n casual/contracted variants of a training phrase."""
    base = text.lower()
    variants = set()

    contracted = base
    applied = False
    for pattern, repls in _CONTRACTION_MAP:
        if re.search(pattern, contracted):
            contracted = re.sub(pattern, rng.choice(repls), contracted, count=1)
            applied = True
    if applied and contracted != base:
        variants.add(contracted)

    wrapped = f"{rng.choice(_CASUAL_PREFIXES)}{base}{rng.choice(_CASUAL_SUFFIXES)}".strip()
    if wrapped != base:
        variants.add(wrapped)

    variants.discard(base)
    return list(variants)[:n]


class Preprocessor:
    """Tokenizes, lemmatizes, and vectorizes text into a bag-of-words
    feature vector over unigrams plus frequent bigrams."""

    def __init__(self, max_bigrams: int = MAX_BIGRAMS) -> None:
        _ensure_nltk_data()
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.stop_words = set(nltk.corpus.stopwords.words("english"))
        self.max_bigrams = max_bigrams
        self.vocabulary: List[str] = []
        self._vocab_index: Dict[str, int] = {}

    # -- tokenization ----------------------------------------------------

    def raw_tokens(self, text: str) -> List[str]:
        """Lowercase, expand math operators to words, drop other punctuation."""
        text = _normalize_operators(text.lower())
        text = _NON_ALNUM.sub(" ", text)
        return _MULTI_SPACE.sub(" ", text).split()

    def lemmatize(self, tokens: List[str]) -> List[str]:
        return [self.lemmatizer.lemmatize(tok) for tok in tokens]

    def unigram_tokens(self, text: str) -> List[str]:
        """Lemmatized tokens with stopwords removed (model input words)."""
        tokens = self.lemmatize(self.raw_tokens(text))
        return [
            t
            for t in tokens
            if t not in self.stop_words and (len(t) >= 2 or t.isdigit())
        ]

    def bigram_tokens(self, text: str) -> List[str]:
        """Bigrams over the lemmatized token stream (stopwords kept so that
        phrases like 'my name is' still form 'my name' / 'name is')."""
        tokens = self.lemmatize(self.raw_tokens(text))
        return [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]

    # -- vocabulary --------------------------------------------------------

    def fit(self, documents: List[Tuple[List[str], List[str]]]) -> None:
        """Build vocabulary from (unigram_tokens, bigram_tokens) pairs."""
        word_counts: Counter = Counter()
        bigram_counts: Counter = Counter()
        for unigrams, bigrams in documents:
            word_counts.update(unigrams)
            bigram_counts.update(bigrams)

        unigrams = sorted(w for w, c in word_counts.items() if c >= 2)
        bigrams = sorted(
            w for w, c in bigram_counts.most_common(self.max_bigrams) if c >= 2
        )
        self.vocabulary = unigrams + bigrams
        self._vocab_index = {w: i for i, w in enumerate(self.vocabulary)}
        logger.info(
            "Vocabulary built: %d unigrams + %d bigrams",
            len(unigrams),
            len(bigrams),
        )

    # -- vectorization -----------------------------------------------------

    def vectorize(self, text: str) -> List[int]:
        """Binary bag-of-words vector over unigrams + bigrams."""
        features = self.unigram_tokens(text) + self.bigram_tokens(text)
        seen = set(features)
        return [1 if word in seen else 0 for word in self.vocabulary]

    # -- serialization -------------------------------------------------------

    def state_dict(self) -> Dict:
        return {"version": PREPROCESSOR_VERSION, "vocabulary": self.vocabulary}

    def load_state_dict(self, state: Dict) -> None:
        self.vocabulary = state["vocabulary"]
        self._vocab_index = {w: i for i, w in enumerate(self.vocabulary)}


# ---------------------------------------------------------------------------
# Neural model
# ---------------------------------------------------------------------------


class ChatNet(nn.Module):
    """Feed-forward intent classifier over bag-of-words features."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: Tuple[int, ...] = (512, 256, 128),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# Backwards-compatible alias so older saved configs still make sense.
ImprovedChatModel = ChatNet


@dataclass
class TrainResult:
    epochs: int
    best_val_loss: float
    best_val_accuracy: float
    num_samples: int
    num_features: int


class ModelTrainer:
    """Trains ChatNet with early stopping, label smoothing, gradient
    clipping, and a cosine learning-rate schedule."""

    def __init__(self, preprocessor: Preprocessor, device: Optional[str] = None):
        self.preprocessor = preprocessor
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

    def train(
        self,
        documents: List[Tuple[List[str], List[str], str]],
        intents: List[str],
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 64,
        patience: int = 25,
        augment: bool = True,
        seed: int = 42,
    ) -> Tuple[ChatNet, TrainResult]:
        """documents: (unigram_tokens, bigram_tokens, intent_tag)."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        X, y = self._build_dataset(documents, intents, augment=augment)
        logger.info(
            "Training data: %d samples, %d features, %d intents",
            X.shape[0],
            X.shape[1],
            len(intents),
        )

        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )
        train_size = int(0.85 * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)

        model = ChatNet(len(self.preprocessor.vocabulary), len(intents)).to(
            self.device
        )
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )

        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_state: Optional[Dict] = None
        epochs_without_improvement = 0
        epochs_run = 0

        for epoch in range(epochs):
            epochs_run = epoch + 1
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                train_loss += loss.item()
            scheduler.step()

            val_loss, val_acc = self._evaluate(model, val_loader, criterion)
            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(
                    "Epoch %d/%d: train_loss=%.4f val_loss=%.4f val_acc=%.2f%%",
                    epoch + 1,
                    epochs,
                    train_loss / max(1, len(train_loader)),
                    val_loss,
                    val_acc * 100,
                )

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        return model, TrainResult(
            epochs=epochs_run,
            best_val_loss=best_val_loss,
            best_val_accuracy=best_val_acc,
            num_samples=int(X.shape[0]),
            num_features=int(X.shape[1]),
        )

    def _build_dataset(
        self,
        documents: List[Tuple[List[str], List[str], str]],
        intents: List[str],
        augment: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        X: List[List[int]] = []
        y: List[int] = []
        for unigrams, bigrams, tag in documents:
            X.append(self._tokens_to_vector(unigrams, bigrams))
            y.append(intents.index(tag))

            if augment and len(unigrams) > 3:
                # one dropout-augmented copy per example
                aug_unigrams = unigrams.copy()
                del aug_unigrams[random.randrange(len(aug_unigrams))]
                aug_bigrams = [
                    f"{a} {b}" for a, b in zip(aug_unigrams, aug_unigrams[1:])
                ]
                X.append(self._tokens_to_vector(aug_unigrams, aug_bigrams))
                y.append(intents.index(tag))

        return (
            np.array(X, dtype=np.float32),
            np.array(y, dtype=np.int64),
        )

    def _tokens_to_vector(
        self, unigrams: List[str], bigrams: List[str]
    ) -> List[int]:
        seen = set(unigrams) | set(bigrams)
        return [1 if w in seen else 0 for w in self.preprocessor.vocabulary]

    @torch.no_grad()
    def _evaluate(
        self,
        model: ChatNet,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, float]:
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            outputs = model(X_batch)
            total_loss += criterion(outputs, y_batch).item()
            correct += (outputs.argmax(1) == y_batch).sum().item()
            total += y_batch.size(0)
        return total_loss / max(1, len(loader)), (correct / total if total else 0.0)


# ---------------------------------------------------------------------------
# Extension functions (side effects and dynamic responses)
# ---------------------------------------------------------------------------


def get_time() -> str:
    return f"The current time is {datetime.now().strftime('%H:%M:%S')}"


def get_date() -> str:
    return f"Today's date is {datetime.now().strftime('%Y-%m-%d')}"


def get_weather() -> str:
    return "It's sunny with a few clouds today."


JOKES = [
    "Why don't scientists trust atoms? Because they make up everything!",
    "I told my wife she was drawing her eyebrows too high. She looked surprised!",
    "Why don't eggs tell jokes? They'd crack each other up!",
    "What do you call a fake noodle? An impasta!",
    "Why did the scarecrow win an award? He was outstanding in his field!",
    "Why don't skeletons fight each other? They don't have the guts.",
    "Why did the math book look sad? Because it had too many problems.",
    "Why don't programmers like nature? It has too many bugs!",
    "What's a computer's favorite snack? Microchips!",
    "How do you comfort a JavaScript bug? You console it!",
    "Why do Python programmers prefer snakes? Because they don't like Java!",
    "There are only 10 kinds of people: those who understand binary and those who don't.",
]


def get_joke() -> str:
    return random.choice(JOKES)


FACTS = [
    "Octopuses have three hearts and blue blood!",
    "Honey never spoils - archaeologists have found edible honey in ancient Egyptian tombs!",
    "A group of flamingos is called a 'flamboyance'!",
    "Bananas are berries, but strawberries aren't!",
    "The shortest war in history lasted only 38-45 minutes!",
    "A shrimp's heart is in its head!",
    "Butterflies taste with their feet!",
    "The human brain uses about 20% of the body's energy!",
]


def get_random_fact() -> str:
    return random.choice(FACTS)


MOTIVATION = [
    "Believe you can and you're halfway there.",
    "It does not matter how slowly you go as long as you do not stop.",
    "Success is not final, failure is not fatal: it is the courage to continue that counts.",
    "Don't watch the clock; do what it does. Keep going.",
    "You miss 100% of the shots you don't take.",
]


def get_motivation() -> str:
    return random.choice(MOTIVATION)


ADVICE = [
    "Remember, every expert was once a beginner. Don't be afraid to start something new!",
    "The best time to plant a tree was 20 years ago. The second best time is now.",
    "Focus on progress, not perfection. Small steps forward are still steps forward.",
    "Be kind to yourself. You're doing better than you think you are.",
    "Listen more than you speak, and you'll learn more than you teach.",
]


def give_advice() -> str:
    return random.choice(ADVICE)


ACTIVITIES = [
    "Why not try drawing or painting?",
    "You could go for a walk or do some exercise.",
    "Have you considered learning a new language or skill?",
    "Maybe you could read a book or watch a movie?",
    "You could play a game or do a puzzle.",
]


def get_fun_activity() -> str:
    return random.choice(ACTIVITIES)


TRIVIA = [
    "What is the capital of Australia? (Answer: Canberra)",
    "Which planet is known as the Red Planet? (Answer: Mars)",
    "What is the largest mammal in the world? (Answer: Blue Whale)",
    "In which year did World War II end? (Answer: 1945)",
    "What is the chemical symbol for gold? (Answer: Au)",
]


def get_trivia_question() -> str:
    return random.choice(TRIVIA)


STORIES = [
    "Once upon a time, in a digital realm, there lived an AI who dreamed of understanding human emotions. Every conversation taught it something new about the complexity of feelings.",
    "In a small town, a curious robot discovered that the secret to happiness wasn't in its programming, but in the connections it made with the people it met.",
    "There was once a chatbot who collected jokes. Each laugh it generated filled its memory banks with joy, proving that humor truly is a universal language.",
]


def generate_story() -> str:
    return random.choice(STORIES)


def start_game() -> str:
    return random.choice(
        [
            "Let's play 20 questions! Think of something and I'll try to guess it.",
            "How about rock, paper, scissors? Just say your choice!",
            "Let's play a riddle game! Here's one: What has keys but no locks?",
            "Want to play word association? I'll say a word, you say the first thing that comes to mind!",
        ]
    )


def explain_technology() -> str:
    return (
        "Technology covers many areas like AI, devices, programming and the "
        "internet. Ask me about a specific topic and we can explore it."
    )


def explain_science() -> str:
    return (
        "Science helps us understand the world through subjects like physics, "
        "chemistry and biology. Ask me about something you're curious about."
    )


def suggest_recipe() -> str:
    return (
        "I don't have full recipes yet, but I can help you think of meal ideas "
        "based on ingredients or cuisines you like."
    )


def provide_coping_support() -> str:
    return (
        "I'm not a professional, but some general coping ideas are deep "
        "breathing, taking a short walk, journaling or talking to someone you trust."
    )


def play_music() -> str:
    return (
        "I can't play music directly, but I recommend checking out Spotify, "
        "Apple Music, or YouTube Music for your listening needs!"
    )


def set_reminder(message: str = "") -> str:
    return (
        "I'd love to set reminders, but I don't have access to a calendar "
        f"system yet. You might want to use your phone's reminder app for: {message}"
        if message
        else "I'd love to set reminders, but I don't have a calendar yet."
    )


def update_human_profile(user_name: str = "", **_) -> str:
    return f"Nice to meet you, {user_name}! I'll remember your name."


def set_ai_name(ai_name: str = "", **_) -> str:
    if not ai_name:
        return "You haven't given me a new name. Try 'your name is Nova'."
    return f"Thank you! I will go by {ai_name} from now on."


def get_time_info(message: str = "") -> str:
    return (
        "I'm not connected to a calendar yet, but right now the time is "
        + datetime.now().strftime("%H:%M:%S")
    )


def translate_text(message: str = "") -> str:
    return (
        "I can help with basic translations! Try asking 'How do you say hello "
        "in Spanish?'"
    )


# -- safe math evaluation ---------------------------------------------------

MATH_WORDS = {
    "plus": "+",
    "add": "+",
    "added": "+",
    "minus": "-",
    "subtract": "-",
    "subtracted": "-",
    "less": "-",
    "times": "*",
    "multiply": "*",
    "multiplied": "*",
    "divided": "/",
    "divide": "/",
    "over": "/",
    "power": "**",
    "squared": "**2",
    "cubed": "**3",
    "mod": "%",
    "modulo": "%",
    "remainder": "%",
    "percent": "/100*",
    "x": "*",
    "to": "",
    "by": "",
    "from": "",
}

MATH_FILLER = {
    "what",
    "is",
    "whats",
    "how",
    "much",
    "many",
    "the",
    "of",
    "a",
    "an",
    "calculate",
    "compute",
    "solve",
    "evaluate",
    "tell",
    "me",
    "please",
    "equals",
    "equal",
    "answer",
    "result",
    "and",
    "are",
    "to",
    "can",
    "you",
    "get",
}

_MATH_CHAR = re.compile(r"[^0-9a-zA-Z+\-*/().\s]")
_NUMBER = r"(\d+(?:\.\d+)?)"

# Imperative forms where infix word-mapping gets the operand order wrong:
# "subtract 5 from 20" -> "20 - 5", "multiply 6 by 7" -> "6 * 7", etc.
_IMPERATIVE_RULES = [
    (re.compile(rf"\badd\s+{_NUMBER}\s+(?:to|and)\s+{_NUMBER}"), r"\1 + \2"),
    (re.compile(rf"\bsubtract\s+{_NUMBER}\s+from\s+{_NUMBER}"), r"\2 - \1"),
    (re.compile(rf"\b{_NUMBER}\s+subtracted\s+from\s+{_NUMBER}"), r"\2 - \1"),
    (re.compile(rf"\bmultiply\s+{_NUMBER}\s+(?:by|and)\s+{_NUMBER}"), r"\1 * \2"),
    (re.compile(rf"\bdivide\s+{_NUMBER}\s+by\s+{_NUMBER}"), r"\1 / \2"),
    (re.compile(rf"\b{_NUMBER}\s+less\s+than\s+{_NUMBER}"), r"\2 - \1"),
]
_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
_MAX_AST_NODES = 60


def _normalize_math(text: str) -> str:
    text = text.lower()
    for pattern, repl in _IMPERATIVE_RULES:
        text = pattern.sub(repl, text)
    text = _MATH_CHAR.sub(" ", text)
    tokens = []
    for tok in text.split():
        if tok in MATH_FILLER:
            continue
        tokens.append(MATH_WORDS.get(tok, tok))
    return _MULTI_SPACE.sub(" ", " ".join(tokens)).strip()


def _eval_math(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_math(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError("unsupported constant")
    if isinstance(node, ast.BinOp):
        op = _ALLOWED_OPS.get(type(node.op))
        if op is None:
            raise ValueError("unsupported operator")
        left = _eval_math(node.left)
        right = _eval_math(node.right)
        if isinstance(node.op, ast.Pow) and (abs(left) > 1e6 or abs(right) > 100):
            raise ValueError("exponent too large")
        if isinstance(node.op, (ast.Div, ast.Mod)) and right == 0:
            raise ZeroDivisionError
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = _ALLOWED_OPS.get(type(node.op))
        if op is None:
            raise ValueError("unsupported unary operator")
        return op(_eval_math(node.operand))
    raise ValueError("unsupported expression")


def calculate_math(message: str = "") -> str:
    """Safely evaluate a natural-language math expression via the AST."""
    normalized = _normalize_math(message or "")
    if not normalized or not any(c.isdigit() for c in normalized):
        return (
            "I couldn't find a math expression. Try something like "
            "'what is 12 times 8'."
        )
    try:
        tree = ast.parse(normalized, mode="eval")
        if len(list(ast.walk(tree))) > _MAX_AST_NODES:
            return "That expression is too long for me to compute."
        result = _eval_math(tree)
        if isinstance(result, float):
            result = int(result) if result.is_integer() else round(result, 6)
        return f"The answer is {result}."
    except ZeroDivisionError:
        return "I can't divide by zero."
    except Exception:
        return "I couldn't compute that. Try a simpler expression."


def looks_like_math(text: str) -> bool:
    """Heuristic: does the message contain a computable expression?"""
    normalized = _normalize_math(text)
    if not normalized or not any(c.isdigit() for c in normalized):
        return False
    # a bare number or parenthesized number isn't a math problem
    if re.fullmatch(r"\(?\d+(\.\d+)?\)?", normalized):
        return False
    # must actually contain an operator to compute something
    if not re.search(r"[+\-*/%^]", normalized):
        return False
    try:
        ast.parse(normalized, mode="eval")
        return True
    except Exception:
        return False


# Mapping from intents.json "extension" function names to callables.
EXTENSION_FUNCTIONS: Dict[str, Callable] = {
    "get_time": get_time,
    "get_date": get_date,
    "get_weather": get_weather,
    "get_joke": get_joke,
    "extensions.math.calculate": calculate_math,
    "extensions.games.startGame": start_game,
    "extensions.gHumans.updateHuman": update_human_profile,
    "extensions.facts.getRandomFact": get_random_fact,
    "extensions.music.playMusic": play_music,
    "extensions.stories.generateStory": generate_story,
    "extensions.reminders.setReminder": set_reminder,
    "extensions.names.setAIName": set_ai_name,
    "extensions.jokes.getRandomJoke": get_joke,
    "extensions.time.getCurrentTime": get_time,
    "extensions.motivation.getMotivation": get_motivation,
    "extensions.advice.giveAdvice": give_advice,
    "extensions.funActivities.suggestActivity": get_fun_activity,
    "extensions.technology.explainTech": explain_technology,
    "extensions.science.explainScience": explain_science,
    "extensions.cooking.suggestRecipe": suggest_recipe,
    "extensions.mentalHealth.provideCoping": provide_coping_support,
    "extensions.time.getTimeInfo": get_time_info,
    "extensions.trivia.getQuestion": get_trivia_question,
    "extensions.translate.translate": translate_text,
}

# Canonical argument each extension expects (empty = no args).
EXTENSION_ARG_SOURCE: Dict[str, str] = {
    "extensions.math.calculate": "message",
    "extensions.reminders.setReminder": "message",
    "extensions.time.getTimeInfo": "message",
    "extensions.translate.translate": "message",
    "extensions.gHumans.updateHuman": "user_name",
    "extensions.names.setAIName": "ai_name",
}


# ---------------------------------------------------------------------------
# Conversation state
# ---------------------------------------------------------------------------


@dataclass
class ConversationTurn:
    user_input: str
    bot_response: str
    intent: str
    confidence: float
    entities: Dict
    timestamp: datetime
    context: Dict


@dataclass
class ChatResponse:
    text: str
    intent: str
    confidence: float
    entities: Dict = field(default_factory=dict)


NAME_PATTERNS = [
    r"(?:my name is|i am|i'm|call me)\s+([A-Za-z][A-Za-z'-]*)",
    r"this is\s+([A-Za-z][A-Za-z'-]*)",
    r"([A-Za-z][A-Za-z'-]*)\s+(?:here|speaking)",
]

_NAME_BLOCKLIST = {
    "sad", "happy", "tired", "good", "ok", "okay", "fine", "bored", "angry",
    "hungry", "sleepy", "sorry", "busy", "confused", "excited", "nervous",
    "scared", "worried", "stressed", "depressed", "lonely", "sick", "well",
    "glad", "mad", "upset", "ready", "sure", "done", "back", "here",
    "not", "so", "very", "really", "just", "also", "now", "still", "feeling",
}

AI_NAME_PATTERNS = [
    r"(?:(?:u|ur|your|yo) name (?:is|'s|will be)|call yourself|i'?ll call you|"
    r"be called|change your name to|rename yourself(?: to)?|i'?ll name you|"
    r"name yourself|your new name is)\s+([A-Za-z][A-Za-z'-]*)",
]

GREETING_PHRASES = {
    "hi", "hello", "hey", "hi there", "hello there", "hey there",
    "good morning", "good afternoon", "good evening", "yo", "hiya",
    "greetings", "sup", "heyy", "heyyy",
}

HOW_ARE_YOU_PHRASES = {
    "how are you", "how are you today", "how are you doing",
    "how's it going", "hows it going", "what's up", "whats up",
    "u good", "u ok", "you good", "you ok", "how do you feel",
}

# Exact-match intents for ultra-short messages — the classifier has too
# little signal on 1-token inputs, so route them directly.
SHORTCUT_INTENTS = {
    "lol": "laughter", "lmao": "laughter", "haha": "laughter",
    "hahaha": "laughter", "hehe": "laughter", "rofl": "laughter",
    "lmfao": "laughter", "dead": "laughter",
    "idk": "uncertainty_idk", "dunno": "uncertainty_idk",
    "thanks": "gratitude_thanks", "thankyou": "gratitude_thanks",
    "thx": "gratitude_thanks", "ty": "gratitude_thanks", "tysm": "gratitude_thanks",
    "wyd": "wyd", "hbu": "hbu", "wby": "hbu", "hby": "hbu",
    "yes": "affirmation_yes", "yeah": "affirmation_yes", "yep": "affirmation_yes",
    "yup": "affirmation_yes", "yea": "affirmation_yes", "ye": "affirmation_yes",
    "mhm": "affirmation_yes", "sure": "affirmation_yes", "bet": "affirmation_yes",
    "no": "negation_no", "nope": "negation_no", "nah": "negation_no",
    "why": "why_questions",
    "sorry": "apology", "oops": "apology",
    "brb": "be_right_back", "gtg": "be_right_back", "afk": "be_right_back",
    "bored": "boredom", "boreddd": "boredom",
    "gn": "goodnight", "gm": "goodmorning", "morning": "goodmorning",
    "night": "goodnight", "goodnight": "goodnight",
    "test": "testing_bot", "testing": "testing_bot",
    "huh": "repeat_request", "huh?": "repeat_request", "what?": "repeat_request",
    "repeat": "repeat_request",
    "hungry": "food_hunger", "starving": "food_hunger",
    "tired": "tired_sleepy", "sleepy": "tired_sleepy",
    "sing": "sing_song",
}

FOLLOWUP_TRIGGERS = {"another", "more", "again", "one more", "tell me more"}

RESET_COMMANDS = {"reset", "clear", "forget this", "start over", "new conversation"}
HELP_COMMANDS = {"help", "commands", "what can you do"}

# topic tracking for context
TOPIC_MAP = {
    "food_talk": "food",
    "music_talk": "music",
    "technology_talk": "technology",
    "emotion_support": "emotions",
    "creative_request": "creativity",
    "math_question": "mathematics",
    "sports_discussion": "sports",
    "travel_talk": "travel",
}

# Open-domain intents answered by the generative transformer when available.
GENERATIVE_INTENTS = {
    "small_talk_general", "daily_routine_chat", "feelings_mood_checkin",
    "work_career_chat", "school_study_chat", "hobbies_daily_chat",
    "family_relationships_chat", "friends_social_chat", "shopping_chat",
    "entertainment_movies_chat", "entertainment_games_chat",
    "news_current_events_chat", "sports_discussion", "history_questions",
    "music_talk", "food_talk", "travel_talk", "technology_talk",
    "health_fitness", "cooking_daily_chat", "sleep_health_chat",
    "motivation_encouragement_chat", "learning_programming_chat",
    "language_learning", "emotion_support",
}

# Intents whose replies are computed by rules/extensions — never used as
# generative training targets, and never generated even in --gen-all mode.
DYNAMIC_INTENTS = {
    "play_game", "productivity_tips", "study_habits", "math_question",
    "name_setup", "name_query", "ai_name_setup", "ai_name_setting",
    "user_name_setting", "name_introduction", "bot_name_query",
    "reset", "help", "generative", "followup", "empty", "fallback",
}


class ChatAssistant:
    """Intent-driven chat assistant combining a neural classifier,
    rule-based overrides, entity extraction, context tracking, and
    extension functions."""

    def __init__(
        self,
        intents_path: str = str(DEFAULT_INTENTS_PATH),
        function_mappings: Optional[Dict[str, Callable]] = None,
        extra_training_paths: Optional[List[str]] = None,
        conversation_log_path: str = str(LOG_PATH),
        profile_path: str = str(PROFILE_PATH),
        confidence_threshold: float = 0.40,
        max_history: int = 50,
        device: Optional[str] = None,
    ) -> None:
        _setup_logging()
        self.intents_path = intents_path
        self.function_mappings = function_mappings or EXTENSION_FUNCTIONS
        self.extra_training_paths = extra_training_paths or [
            str(DEFAULT_EXTRA_TRAINING)
        ]
        self.extra_intents_paths = [
            str(p) for p in EXTRA_INTENTS_PATHS if p.exists()
        ]
        self.conversation_log_path = conversation_log_path
        self.profile_path = profile_path
        self.confidence_threshold = confidence_threshold
        self.max_history = max_history
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.preprocessor = Preprocessor()
        self.model: Optional[ChatNet] = None
        self.gen_model: Optional[Seq2SeqTransformer] = None
        self.gen_vocab: Optional[GenVocab] = None
        self.use_generative = True
        # When True, the transformer writes every conversational reply —
        # deterministic handlers (math, game, name memory, reset) still run
        # first since they compute real state, not canned text.
        self.generative_all = False

        self.intents: List[str] = []
        self.responses: Dict[str, List[str]] = {}
        self.extensions: Dict[str, Dict] = {}
        self.contexts: Dict[str, Dict] = {}
        self.entity_types: Dict[str, str] = {}
        self.entity_patterns: Dict[str, List[Dict]] = {}
        self.documents: List[Tuple[List[str], List[str], str]] = []
        self.raw_texts: Dict[str, List[str]] = {}
        self.intent_keywords: Dict[str, set] = {}

        self.conversation_history: List[ConversationTurn] = []
        self.current_context: Dict[str, Dict] = {}
        self.user_profile: Dict = {}
        self.ai_name = "Assistant"

        self._log_lock = threading.Lock()
        self._load_user_profile()

    # -- properties ---------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    # -- data loading ---------------------------------------------------------

    def data_signature(self) -> str:
        """SHA1 signature of the training data files, used to detect when a
        saved model is stale."""
        h = hashlib.sha1()
        for path in [
            self.intents_path, *self.extra_intents_paths, *self.extra_training_paths
        ]:
            try:
                with open(path, "rb") as f:
                    h.update(f.read())
            except OSError:
                continue
        return h.hexdigest()

    def load_data(self) -> None:
        """Load intents.json + intents_extra.json plus extra training files."""
        self.intents.clear()
        self.responses.clear()
        self.extensions.clear()
        self.contexts.clear()
        self.entity_types.clear()
        self.entity_patterns.clear()
        self.documents.clear()
        self.raw_texts.clear()
        self.intent_keywords.clear()

        rng = random.Random(42)
        for path in [self.intents_path, *self.extra_intents_paths]:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._merge_intents(data["intents"], rng)

        for path in self.extra_training_paths:
            self._load_extra_training(path)

        # Deduplicate identical (tokens, tag) pairs produced by expansion scripts.
        seen = set()
        unique_docs = []
        for unigrams, bigrams, tag in self.documents:
            key = (tag, tuple(unigrams))
            if key not in seen:
                seen.add(key)
                unique_docs.append((unigrams, bigrams, tag))
        self.documents = unique_docs

        logger.info(
            "Loaded %d intents, %d training documents",
            len(self.intents),
            len(self.documents),
        )

    def _merge_intents(self, intents_data: List[Dict], rng: random.Random) -> None:
        """Merge an intents list into the corpus. Duplicate tags extend the
        existing intent rather than overwriting it."""
        for intent in intents_data:
            tag = intent["intent"]
            if tag not in self.intents:
                self.intents.append(tag)

            resp_list = self.responses.setdefault(tag, [])
            for r in intent.get("responses", []):
                if r not in resp_list:
                    resp_list.append(r)
            if not self.extensions.get(tag):
                self.extensions[tag] = intent.get("extension", {}) or {}
            if not self.contexts.get(tag):
                self.contexts[tag] = intent.get("context", {}) or {}
            self.entity_types.setdefault(tag, intent.get("entityType", "NA"))
            existing_patterns = self.entity_patterns.setdefault(tag, [])
            for p in intent.get("entities", []) or []:
                if p not in existing_patterns:
                    existing_patterns.append(p)

            keywords = self.intent_keywords.setdefault(tag, set())
            texts = list(dict.fromkeys(str(t) for t in intent.get("text", [])))
            if len(texts) > MAX_TEXTS_PER_INTENT:
                texts = rng.sample(texts, MAX_TEXTS_PER_INTENT)

            all_texts = list(texts)
            for t in texts:
                all_texts.extend(augment_text(t, rng))
            self.raw_texts.setdefault(tag, []).extend(all_texts)

            for text in all_texts:
                unigrams = self.preprocessor.unigram_tokens(text)
                if not unigrams:
                    continue
                bigrams = self.preprocessor.bigram_tokens(text)
                self.documents.append((unigrams, bigrams, tag))
                keywords.update(unigrams)

    def _load_extra_training(self, path: str) -> None:
        if not path or not os.path.exists(path):
            return
        added = 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("\t", 1)
                    if len(parts) != 2:
                        continue
                    tag, text = parts[0].strip(), parts[1].strip()
                    if not tag or not text:
                        continue
                    if tag not in self.intents:
                        self.intents.append(tag)
                        self.responses[tag] = [
                            f"I recognized your message as '{tag}', but I don't have a detailed response yet."
                        ]
                        self.extensions[tag] = {}
                        self.contexts[tag] = {}
                        self.entity_types[tag] = "NA"
                        self.entity_patterns[tag] = []
                    unigrams = self.preprocessor.unigram_tokens(text)
                    bigrams = self.preprocessor.bigram_tokens(text)
                    self.documents.append((unigrams, bigrams, tag))
                    self.intent_keywords.setdefault(tag, set()).update(unigrams)
                    self.raw_texts.setdefault(tag, []).append(text)
                    added += 1
            if added:
                logger.info("Loaded %d extra training examples from %s", added, path)
        except OSError as e:
            logger.error("Error loading extra training data %s: %s", path, e)

    # -- training -----------------------------------------------------------

    def train(
        self,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 64,
        patience: int = 25,
    ) -> TrainResult:
        if not self.documents:
            self.load_data()
        fit_docs = [(u, b) for u, b, _ in self.documents]
        self.preprocessor.fit(fit_docs)
        trainer = ModelTrainer(self.preprocessor, device=str(self.device))
        self.model, result = trainer.train(
            self.documents,
            self.intents,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            patience=patience,
        )
        return result

    # -- persistence -----------------------------------------------------------

    def save_model(
        self,
        model_path: str = str(MODEL_PATH),
        config_path: str = str(CONFIG_PATH),
    ) -> None:
        if self.model is None:
            raise RuntimeError("No model to save")
        torch.save(self.model.state_dict(), model_path)
        config = {
            "config_version": PREPROCESSOR_VERSION,
            "data_signature": self.data_signature(),
            "preprocessor": self.preprocessor.state_dict(),
            "intents": self.intents,
            "responses": self.responses,
            "extensions": self.extensions,
            "contexts": self.contexts,
            "entity_types": self.entity_types,
            "entity_patterns": self.entity_patterns,
            "user_profile": self.user_profile,
            "ai_name": self.ai_name,
            "confidence_threshold": self.confidence_threshold,
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        logger.info("Model saved to %s, config to %s", model_path, config_path)

    def load_model(
        self,
        model_path: str = str(MODEL_PATH),
        config_path: str = str(CONFIG_PATH),
    ) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        if config.get("config_version") != PREPROCESSOR_VERSION:
            raise ValueError(
                f"Config version mismatch: saved={config.get('config_version')} "
                f"expected={PREPROCESSOR_VERSION}. Retrain the model."
            )
        saved_sig = config.get("data_signature")
        if saved_sig and saved_sig != self.data_signature():
            raise ValueError(
                "Training data changed since the model was saved. Retrain the model."
            )

        self.preprocessor.load_state_dict(config["preprocessor"])
        self.intents = config["intents"]
        self.responses = config["responses"]
        self.extensions = config["extensions"]
        self.contexts = config.get("contexts", {})
        self.entity_types = config.get("entity_types", {})
        self.entity_patterns = config.get("entity_patterns", {})
        # merge profiles: the runtime user_profile.json wins over the
        # training-time snapshot embedded in the config
        merged = dict(config.get("user_profile", {}))
        merged.update(self.user_profile)
        self.user_profile = merged
        self.ai_name = self.user_profile.get(
            "ai_name", config.get("ai_name", self.ai_name)
        )
        self.confidence_threshold = config.get(
            "confidence_threshold", self.confidence_threshold
        )

        self.model = ChatNet(
            len(self.preprocessor.vocabulary), len(self.intents)
        ).to(self.device)
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        logger.info("Model loaded from %s", model_path)

    def _load_user_profile(self) -> None:
        if not os.path.exists(self.profile_path):
            return
        try:
            with open(self.profile_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.user_profile.update(data)
                if "ai_name" in data:
                    self.ai_name = data["ai_name"]
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Error loading user profile: %s", e)

    def _save_user_profile(self) -> None:
        try:
            with open(self.profile_path, "w", encoding="utf-8") as f:
                json.dump(self.user_profile, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.error("Error saving user profile: %s", e)

    # -- entity extraction ------------------------------------------------------

    def _extract_entities(self, text: str, intent_tag: str) -> Dict:
        entities: Dict[str, List[str]] = {}

        for pattern in NAME_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1)
                if name.lower() not in _NAME_BLOCKLIST:
                    entities["HUMAN"] = [name]
                break

        numbers = re.findall(r"\d+", text)
        if numbers:
            entities["NUMBER"] = numbers

        emails = re.findall(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text
        )
        if emails:
            entities["EMAIL"] = emails

        for spec in self.entity_patterns.get(intent_tag, []):
            entity_name = spec.get("entity")
            if not entity_name:
                continue
            values = []
            for p in spec.get("patterns", []):
                if not p:
                    continue
                if any(ch in p for ch in "[\\()?+*^$|"):
                    try:
                        for m in re.findall(p, text, flags=re.IGNORECASE):
                            if isinstance(m, tuple):
                                values.extend(str(x) for x in m if x)
                            else:
                                values.append(str(m))
                    except re.error:
                        if p.lower() in text.lower():
                            values.append(p)
                else:
                    if p.lower() in text.lower():
                        values.append(p)
            if values:
                existing = entities.get(entity_name, [])
                for v in values:
                    if v not in existing:
                        existing.append(v)
                entities[entity_name] = existing

        # the AI_NAME entity list is literal ("Nova", "Bot", ...) and matches
        # by substring — "novas" would wrongly extract "Nova". A regex capture
        # of what the user actually typed always wins.
        if intent_tag == "ai_name_setting":
            captured = self._extract_ai_name(text)
            if captured:
                entities["AI_NAME"] = [captured.capitalize()]

        return entities

    # -- context ------------------------------------------------------------------

    def _context_for(self, user_id: str) -> Dict:
        ctx = self.current_context.get(user_id)
        if ctx is None:
            ctx = {
                "conversation_history": [],
                "active_topics": [],
                "user_preferences": {},
                "session_start": datetime.now().isoformat(),
                "last_intent": None,
            }
            self.current_context[user_id] = ctx
        return ctx

    def _update_context(self, intent_tag: str, message: str, user_id: str) -> None:
        ctx = self._context_for(user_id)
        ctx["conversation_history"].append(
            {
                "intent": intent_tag,
                "message": message,
                "timestamp": datetime.now().isoformat(),
            }
        )
        ctx["conversation_history"] = ctx["conversation_history"][-10:]
        ctx["last_intent"] = intent_tag

        topic = TOPIC_MAP.get(intent_tag)
        if topic:
            if topic not in ctx["active_topics"]:
                ctx["active_topics"].append(topic)
            ctx["active_topics"] = ctx["active_topics"][-3:]

    def reset_context(self, user_id: str = "default") -> str:
        self.current_context.pop(user_id, None)
        return "Done - I've forgotten this conversation. What's on your mind?"

    # -- generative model -----------------------------------------------------

    @property
    def generative_ready(self) -> bool:
        return self.use_generative and self.gen_model is not None

    def train_generative(self, epochs: int = 12, max_pairs: int = 60000) -> GenTrainResult:
        """Train the seq2seq transformer on (message, response) pairs."""
        if not self.documents or not self.raw_texts:
            self.load_data()
        self.gen_vocab = GenVocab(min_freq=2)
        all_texts = [t for texts in self.raw_texts.values() for t in texts]
        all_texts += [r for rs in self.responses.values() for r in rs]
        all_texts += list(self.raw_texts.keys())  # intent tags as conditioning tokens
        self.gen_vocab.build(all_texts)

        pairs = build_pairs(
            self.documents, self.raw_texts, self.responses,
            exclude_intents=DYNAMIC_INTENTS, max_pairs=max_pairs,
        )
        logger.info("Generative training: %d pairs, vocab %d", len(pairs), len(self.gen_vocab))
        self.gen_model, result = train_generator(
            pairs, self.gen_vocab, epochs=epochs, device=str(self.device), logger=logger,
        )
        return result

    def save_generative(
        self,
        model_path: str = str(GEN_MODEL_PATH),
        config_path: str = str(GEN_CONFIG_PATH),
    ) -> None:
        if self.gen_model is None or self.gen_vocab is None:
            raise RuntimeError("No generative model to save")
        torch.save(self.gen_model.state_dict(), model_path)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "vocab": self.gen_vocab.state_dict(),
                    "data_signature": self.data_signature(),
                },
                f,
            )
        logger.info("Generative model saved to %s", model_path)

    def load_generative(
        self,
        model_path: str = str(GEN_MODEL_PATH),
        config_path: str = str(GEN_CONFIG_PATH),
    ) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        saved_sig = config.get("data_signature")
        if saved_sig and saved_sig != self.data_signature():
            raise ValueError("generative model is stale (data changed)")
        self.gen_vocab = GenVocab()
        self.gen_vocab.load_state_dict(config["vocab"])
        self.gen_model = Seq2SeqTransformer(len(self.gen_vocab)).to(self.device)
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        self.gen_model.load_state_dict(state)
        self.gen_model.eval()
        logger.info("Generative model loaded from %s", model_path)

    @torch.no_grad()
    def warmup(self) -> None:
        """Run throwaway inferences so the first real message doesn't pay
        PyTorch's cold-start cost (~1-2s of lazy MKL/kernel init)."""
        self._predict("hi")
        if self.gen_model is not None:
            src = self.gen_vocab.encode("hi", 48)
            if src:
                self.gen_model.generate_beam(src, max_len=4, device=self.device)
        logger.info("Warmup inference done")

    def _generate_reply(
        self,
        message: str,
        on_token=None,
        mode: str = "beam",
        intent_hint: Optional[str] = None,
    ) -> str:
        # prefix with the intent tag when known — the model was trained on
        # "intent_tag message" sources, so this steers the reply
        src_text = f"{intent_hint} {message}" if intent_hint else message
        src = self.gen_vocab.encode(src_text, 48)
        if not src:
            return self._fallback_response(message, [])

        if mode == "beam":
            # beam search can't stream mid-search, so emit the finished
            # reply word-by-word to keep the typing effect
            ids = self.gen_model.generate_beam(src, device=self.device)
            text = clean_reply(self.gen_vocab.decode(ids))
            if on_token and text:
                for word in text.split(" "):
                    on_token(word + " ")
            return text or self._fallback_response(message, [])

        ids: List[int] = []
        prev_text = ""
        for tok_id in self.gen_model.generate_stream(src, device=self.device):
            ids.append(tok_id)
            if on_token is not None:
                current = self.gen_vocab.decode(ids)
                delta = current[len(prev_text):]
                if delta:
                    on_token(to_ascii(delta))
                prev_text = current
        text = clean_reply(self.gen_vocab.decode(ids))
        return text or self._fallback_response(message, [])

    def _gen_has_signal(self, message: str) -> bool:
        """True if the message has at least one known vocab word — prevents
        the transformer from generating confident nonsense on gibberish."""
        return any(i != UNK for i in self.gen_vocab.encode(message, 48))

    # -- inference ----------------------------------------------------------------

    @torch.no_grad()
    def _predict(self, message: str) -> Tuple[str, float]:
        """Return (intent_tag, confidence) from the neural classifier."""
        if self.model is None:
            return "fallback", 0.0
        bow = self.preprocessor.vectorize(message)
        x = torch.tensor([bow], dtype=torch.float32, device=self.device)
        self.model.eval()
        probs = torch.softmax(self.model(x), dim=1)
        confidence, idx = torch.max(probs, 1)
        return self.intents[idx.item()], float(confidence.item())

    def _suggest_intents(self, message: str, k: int = 2) -> List[str]:
        """Keyword-overlap suggestions for low-confidence inputs."""
        words = set(self.preprocessor.unigram_tokens(message))
        if not words:
            return []
        scores = Counter()
        for tag, keywords in self.intent_keywords.items():
            overlap = len(words & keywords)
            if overlap:
                scores[tag] = overlap
        return [tag for tag, _ in scores.most_common(k)]

    # -- response pipeline -----------------------------------------------------------

    def handle_message(
        self, message: str, user_id: str = "default", on_token=None
    ) -> ChatResponse:
        """Main entry point: process a user message and return a response.

        on_token: optional callable receiving text fragments as the
        generative model produces them (word-by-word streaming)."""
        message = (message or "").strip()[:MAX_INPUT_CHARS]
        if not message:
            return ChatResponse("Say something and I'll do my best!", "empty", 0.0)

        ctx = self._context_for(user_id)

        # 1. Built-in commands
        lower = message.lower().strip()
        if lower in RESET_COMMANDS:
            text = self.reset_context(user_id)
            return self._finalize(message, text, "reset", 1.0, {}, user_id)
        if lower in HELP_COMMANDS:
            text = self._help_response()
            return self._finalize(message, text, "help", 1.0, {}, user_id)

        # 2. Active game state (number guessing) takes precedence — but a
        # confident non-game intent exits the game and falls through
        if ctx.get("active_game", {}).get("type") == "number":
            text = self._handle_number_game_turn(message, user_id)
            if text is not None:
                return self._finalize(message, text, "play_game", 1.0, {}, user_id)
        elif any(k in lower for k in ("quit game", "stop game", "end game")):
            text = "No game is running right now. Say 'play a game' to start one!"
            return self._finalize(message, text, "play_game", 1.0, {}, user_id)

        # 3. Name capture / query
        name = self._extract_name(message)
        if name:
            self.user_profile["name"] = name
            self._save_user_profile()
            text = f"Nice to meet you, {name}! I'll remember your name."
            return self._finalize(message, text, "name_setup", 1.0, {"HUMAN": [name]}, user_id)

        if self._is_name_query(lower):
            name = self.user_profile.get("name")
            text = (
                f"Your name is {name}."
                if name
                else "I don't know your name yet. What should I call you?"
            )
            return self._finalize(message, text, "name_query", 1.0, {}, user_id)

        ai_name = self._extract_ai_name(message)
        if ai_name:
            ai_name = ai_name.capitalize()
            self.ai_name = ai_name
            self.user_profile["ai_name"] = ai_name
            self._save_user_profile()
            text = f"Got it - you can call me {ai_name} from now on."
            return self._finalize(message, text, "ai_name_setup", 1.0, {"AI_NAME": [ai_name]}, user_id)

        if self._is_bot_name_query(lower):
            text = (
                f"My name is {self.ai_name}. You can rename me by saying "
                f"'your name is X'."
            )
            return self._finalize(message, text, "bot_name_query", 1.0, {}, user_id)

        # 4. Math expressions go straight to the safe evaluator
        if looks_like_math(message):
            text = calculate_math(message)
            return self._finalize(message, text, "math_question", 1.0, {}, user_id)

        # 5. Follow-up requests ("another", "more", "again")
        if lower.split()[0] in FOLLOWUP_TRIGGERS or lower in FOLLOWUP_TRIGGERS:
            if self.generative_all and self.generative_ready:
                # regenerate from the last real user message
                hist = ctx.get("conversation_history") or []
                last_msg = next(
                    (h["message"] for h in reversed(hist)
                     if h.get("intent") not in ("followup", "fallback")),
                    None,
                )
                last_intent = next(
                    (h["intent"] for h in reversed(hist)
                     if h.get("intent") not in ("followup", "fallback")),
                    None,
                )
                if last_msg and self._gen_has_signal(last_msg):
                    text = self._generate_reply(
                        last_msg, on_token, intent_hint=last_intent
                    )
                    return self._finalize(message, text, "followup", 0.9, {}, user_id)
            text = self._handle_followup(lower, user_id)
            if text:
                return self._finalize(message, text, "followup", 0.9, {}, user_id)

        # 6. Rule-based overrides for high-precision phrases
        intent_tag = self._rule_override(lower)
        if intent_tag:
            entities = self._extract_entities(message, intent_tag)
            self._update_context(intent_tag, message, user_id)
            if self.generative_all and self.generative_ready:
                text = self._generate_reply(message, on_token, intent_hint=intent_tag)
            else:
                text = self._build_response(intent_tag, entities, user_id, message)
            return self._finalize(message, text, intent_tag, 1.0, entities, user_id)

        # 7. Neural classifier
        intent_tag, confidence = self._predict(message)

        if confidence < self.confidence_threshold:
            if self.generative_ready and self._gen_has_signal(message):
                text = self._generate_reply(message, on_token, intent_hint=intent_tag)
                self._update_context("generative", message, user_id)
                return self._finalize(message, text, "generative", confidence, {}, user_id)
            suggestions = self._suggest_intents(message)
            text = self._fallback_response(message, suggestions)
            return self._finalize(message, text, "fallback", confidence, {}, user_id)

        entities = self._extract_entities(message, intent_tag)
        if "HUMAN" in entities:
            self.user_profile["name"] = entities["HUMAN"][0]
            self._save_user_profile()

        self._update_context(intent_tag, message, user_id)

        if (
            self.generative_ready
            and intent_tag not in DYNAMIC_INTENTS
            and (self.generative_all or intent_tag in GENERATIVE_INTENTS)
        ):
            text = self._generate_reply(
                message, on_token, intent_hint=intent_tag
            )
            return self._finalize(message, text, intent_tag, confidence, entities, user_id)

        text = self._build_response(intent_tag, entities, user_id, message)
        return self._finalize(message, text, intent_tag, confidence, entities, user_id)

    # Backwards-compatible signature used by interface.py and tests.
    def get_response(
        self, message: str, user_id: str = "default"
    ) -> Tuple[str, float, Dict, str]:
        r = self.handle_message(message, user_id)
        return r.text, r.confidence, r.entities, r.intent

    # -- helpers --------------------------------------------------------------

    def _extract_name(self, message: str) -> Optional[str]:
        for pattern in NAME_PATTERNS:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                name = match.group(1)
                if name.lower() not in _NAME_BLOCKLIST:
                    return name
        return None

    def _extract_ai_name(self, message: str) -> Optional[str]:
        for pattern in AI_NAME_PATTERNS:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                name = match.group(1)
                if name.lower() not in _NAME_BLOCKLIST:
                    return name
        return None

    def _is_bot_name_query(self, lower: str) -> bool:
        return any(p in lower for p in (
            "what is your name", "what's your name", "whats your name",
            "what should i call you", "what do they call you",
            "what are you called", "do you have a name",
            "tell me your name", "what are you named",
        ))

    def _is_name_query(self, lower: str) -> bool:
        return (
            "what is my name" in lower
            or "what's my name" in lower
            or "whats my name" in lower
            or ("what" in lower and "my name" in lower)
            or "do you remember my name" in lower
            or "who am i" in lower
        )

    def _rule_override(self, lower: str) -> Optional[str]:
        shortcut = SHORTCUT_INTENTS.get(lower)
        if shortcut and shortcut in self.intents:
            return shortcut
        if lower in GREETING_PHRASES:
            return "greeting" if "greeting" in self.intents else self.intents[0]
        if lower in HOW_ARE_YOU_PHRASES:
            return (
                "how_are_you" if "how_are_you" in self.intents else self.intents[0]
            )
        return None

    def _handle_followup(self, lower: str, user_id: str) -> Optional[str]:
        ctx = self._context_for(user_id)
        last_intent = ctx.get("last_intent")
        if not last_intent or last_intent in {"fallback", "reset", "help"}:
            return None
        # map keyword to a specific intent if the user says e.g. "another joke"
        if "joke" in lower:
            last_intent = "joke" if "joke" in self.intents else last_intent
        elif "fact" in lower:
            last_intent = (
                "random_question" if "random_question" in self.intents else last_intent
            )
        elif "story" in lower:
            last_intent = (
                "creative_request" if "creative_request" in self.intents else last_intent
            )

        entities: Dict = {}
        self._update_context(last_intent, "another", user_id)
        return self._build_response(last_intent, entities, user_id, "another")

    def _build_response(
        self,
        intent_tag: str,
        entities: Dict,
        user_id: str,
        message: str,
    ) -> str:
        # Special dynamic generators first
        if intent_tag == "play_game":
            try:
                lower_msg = message.lower()
                if any(k in lower_msg for k in ("quit", "stop", "end", "cancel", "nevermind")):
                    return "No game is running right now. Say 'play a game' to start one!"
                return self._start_number_game(user_id)
            except Exception as e:
                logger.error("Error starting number game: %s", e)
        elif intent_tag == "productivity_tips":
            try:
                return self._generate_productivity_plan(message, user_id, entities)
            except Exception as e:
                logger.error("Error generating productivity plan: %s", e)
        elif intent_tag == "study_habits":
            try:
                return self._generate_study_plan(message, user_id, entities)
            except Exception as e:
                logger.error("Error generating study plan: %s", e)

        # Extension function from intents.json
        func_name = self.extensions.get(intent_tag, {}).get("function", "")
        func = self.function_mappings.get(func_name)
        if func is not None:
            try:
                result = self._call_extension(func_name, func, message, entities)
                if result:
                    # ai_name_setting extension only returns text — apply it
                    if intent_tag == "ai_name_setting" and entities.get("AI_NAME"):
                        new_name = entities["AI_NAME"][0]
                        if new_name != self.ai_name:
                            self.ai_name = new_name
                            self.user_profile["ai_name"] = new_name
                            self._save_user_profile()
                    return result
            except Exception as e:
                logger.error("Error executing extension %s: %s", func_name, e)

        # Static response with entity substitution
        options = self.responses.get(intent_tag)
        if not options:
            return "I don't have a response for that yet."
        response = random.choice(options)

        # Variety for repeated intents
        ctx = self._context_for(user_id)
        recent = [h["intent"] for h in ctx["conversation_history"][-3:]]
        if recent.count(intent_tag) > 1:
            response = random.choice(
                {
                    "joke": [
                        "Here's another one for you!",
                        "I've got more where that came from!",
                        "Another joke coming up!",
                    ],
                    "math_question": [
                        "Let me solve another one!",
                        "More math? I love it!",
                        "Another calculation coming up!",
                    ],
                    "random_question": [
                        "Here's something else interesting:",
                        "Let me share another fact:",
                        "Something different this time:",
                    ],
                }.get(intent_tag, [response])
            )

        for entity_type, values in entities.items():
            if values:
                response = response.replace(f"%%{entity_type}%%", str(values[0]))
        if "name" in self.user_profile:
            response = response.replace("%%USER%%", str(self.user_profile["name"]))
        response = response.replace("%%AI_NAME%%", self.ai_name)
        response = re.sub(r"%%[A-Z_]+%%", "", response).strip()
        return response

    def _call_extension(
        self,
        func_name: str,
        func: Callable,
        message: str,
        entities: Dict,
    ) -> Optional[str]:
        source = EXTENSION_ARG_SOURCE.get(func_name)
        params = inspect.signature(func).parameters
        kwargs: Dict[str, object] = {}
        if "entities" in params:
            kwargs["entities"] = entities

        if source == "message" or "message" in params:
            return func(message=message, **kwargs)
        if source == "user_name" and "HUMAN" in entities:
            return func(user_name=entities["HUMAN"][0], **kwargs)
        if source == "ai_name" and "AI_NAME" in entities:
            return func(ai_name=entities["AI_NAME"][0], **kwargs)
        if source is None:
            return func()
        return func(**kwargs)

    def _fallback_response(self, message: str, suggestions: List[str]) -> str:
        lower = message.lower()
        if any(w in lower for w in ("help", "assist", "support")):
            return "I'd love to help! What specifically do you need assistance with?"
        if any(w in lower for w in ("study", "exam", "homework", "assignment", "revise")):
            return (
                "It sounds like you need study help. Try 'Give me tips for exam "
                "preparation' or 'How can I study more effectively?'"
            )
        if any(w in lower for w in ("productivity", "focus", "procrastinate", "time management")):
            return (
                "It sounds like you want productivity help. Try 'Give me tips to "
                "stop procrastinating' or 'How can I be more productive each day?'"
            )
        if any(w in lower for w in ("budget", "saving", "expenses", "spending", "finance", "money")):
            return (
                "It sounds like you're asking about money. Try 'How can I create a "
                "monthly budget?' or 'Give me tips for controlling my spending.'"
            )
        if suggestions:
            pretty = " or ".join(f"'{s.replace('_', ' ')}'" for s in suggestions[:2])
            return (
                "I'm not fully sure I understood. Did you mean something like "
                f"{pretty}? You can also say 'help' to see what I can do."
            )
        if "?" in message:
            return "That's a great question! I'm not sure about that yet, but I'm always learning."
        return random.choice(
            [
                "I'm not sure I understand. Could you rephrase that?",
                "I'm still learning. Can you try asking that differently?",
                "That's interesting! I'm not quite sure how to respond to that yet.",
                "Could you be more specific? I want to help but need more clarity.",
            ]
        )

    def _help_response(self) -> str:
        capabilities = [
            "chat and small talk",
            "tell jokes, facts, and stories",
            "solve math expressions (e.g. 'what is 12 times 8')",
            "play a number guessing game",
            "give productivity, study, and motivation tips",
            "remember your name and preferences",
        ]
        lines = [f"Here's what I can do ({self.ai_name}):"]
        lines += [f"  - {c}" for c in capabilities]
        lines.append("Say 'reset' to start a fresh conversation.")
        return "\n".join(lines)

    # -- games -------------------------------------------------------------------

    def _start_number_game(self, user_id: str) -> str:
        ctx = self._context_for(user_id)
        ctx["active_game"] = {
            "type": "number",
            "target": random.randint(1, 50),
            "attempts": 0,
        }
        return (
            "Let's play a number guessing game! I'm thinking of a number "
            "between 1 and 50. Type a number to guess, or 'quit game' to stop."
        )

    def _handle_number_game_turn(self, message: str, user_id: str) -> Optional[str]:
        """Returns a game reply, or None when the user clearly switched to a
        different intent — in which case the normal pipeline should proceed."""
        lower = message.lower()
        ctx = self._context_for(user_id)
        game = ctx.get("active_game") or {}

        if any(k in lower for k in (
            "quit game", "stop game", "end game", "nevermind",
            "cancel", "give up", "i give up",
        )):
            ctx.pop("active_game", None)
            return "Okay, game over. Ask me to play again anytime!"

        numbers = re.findall(r"\d+", message)
        if numbers:
            guess = int(numbers[0])
            if guess < 1 or guess > 50:
                return "Pick a number between 1 and 50!"
            target = int(game.get("target", 0))
            attempts = int(game.get("attempts", 0)) + 1
            ctx["active_game"]["attempts"] = attempts
            if guess < target:
                return f"Higher than {guess}. Try again!"
            if guess > target:
                return f"Lower than {guess}. Try again!"
            ctx.pop("active_game", None)
            return f"Nice! You guessed {target} in {attempts} tries."

        # Non-numeric input: if it clearly belongs to another intent, end the
        # game and let the normal pipeline handle the message.
        other = self._rule_override(lower)
        if not other:
            tag, conf = self._predict(message)
            if conf >= 0.55 and tag not in ("play_game", "fallback"):
                other = tag
        if other:
            ctx.pop("active_game", None)
            return None
        return "I'm still thinking of a number between 1 and 50. Guess a number, or type 'quit game'."

    # -- dynamic plan generators ----------------------------------------------------

    def _generate_productivity_plan(
        self, message: str, user_id: str, entities: Dict
    ) -> str:
        lower = message.lower()
        user_name = self.user_profile.get("name", "you")
        numbers = entities.get("NUMBER") or []
        time_hint = numbers[0] if numbers else None
        timeframe = (
            "tomorrow" if "tomorrow" in lower
            else "this week" if "week" in lower
            else "today"
        )

        ctx = self._context_for(user_id)
        recent = [h["intent"] for h in ctx["conversation_history"][-3:]]
        stress_note = ""
        if any(i in recent for i in ("emotion_support", "stress_management")):
            stress_note = "Start gentle and don't overload yourself. "

        focus_block = (
            f"Use {time_hint} focused 25-minute blocks with 5-minute breaks."
            if time_hint
            else "Use a few focused 25-minute blocks with 5-minute breaks."
        )
        lines = [
            f"Here is a simple productivity plan for {timeframe}, {user_name}:",
            "",
            "1. Define your main goal:",
            "   - Write one clear sentence about what you want to finish.",
            "2. List your top 3 tasks:",
            "   - Choose the three most important tasks that move you toward that goal.",
            "3. Create time blocks:",
            f"   - {focus_block}",
            "4. Remove obvious distractions:",
            "   - Put your phone away, close unrelated tabs, and prepare what you need.",
            "5. Start with the smallest step:",
            "   - Pick the easiest first action (like opening the document or writing a heading).",
            "6. Review at the end:",
            "   - Check what you finished and write down the next small step for tomorrow.",
        ]
        if stress_note:
            lines += ["", "Note: " + stress_note.strip()]
        return "\n".join(lines)

    def _generate_study_plan(
        self, message: str, user_id: str, entities: Dict
    ) -> str:
        lower = message.lower()
        user_name = self.user_profile.get("name", "you")
        goal = (
            "prepare for your exam" if "exam" in lower
            else "prepare for your test" if "test" in lower
            else "study more effectively"
        )
        numbers = entities.get("NUMBER") or []
        sessions_note = (
            f"Plan about {numbers[0]} focused study sessions. Adjust if that feels like too much or too little."
            if numbers
            else "Plan 2-4 focused study sessions of 25-40 minutes each."
        )

        ctx = self._context_for(user_id)
        recent = [h["intent"] for h in ctx["conversation_history"][-3:]]
        memory_note = ""
        if "memory_issues" in recent or "emotion_support" in recent:
            memory_note = (
                "Use active recall (testing yourself) and spaced repetition instead of just rereading notes."
            )

        lines = [
            f"Here is a simple plan to {goal}, {user_name}:",
            "",
            "1. Clarify the topics:",
            "   - Write a short list of chapters or concepts you need to cover.",
            "2. Break topics into small chunks:",
            "   - Turn each topic into small questions you should be able to answer.",
            "3. Schedule study blocks:",
            f"   - {sessions_note}",
            "4. Use active study methods:",
            "   - Explain the material in your own words, teach it to an imaginary friend, or write flashcards.",
            "5. Test yourself:",
            "   - Close your notes and try to recall key ideas or solve practice questions.",
            "6. Review and adjust:",
            "   - At the end of the day, note what worked and what still feels confusing.",
        ]
        if memory_note:
            lines += ["", "Extra tip: " + memory_note]
        return "\n".join(lines)

    # -- logging -----------------------------------------------------------------------

    def _finalize(
        self,
        message: str,
        text: str,
        intent_tag: str,
        confidence: float,
        entities: Dict,
        user_id: str,
    ) -> ChatResponse:
        text = to_ascii(text)  # safe rendering on non-UTF-8 consoles
        turn = ConversationTurn(
            user_input=message,
            bot_response=text,
            intent=intent_tag,
            confidence=confidence,
            entities=entities,
            timestamp=datetime.now(),
            context=(self.current_context.get(user_id) or {}).copy(),
        )
        self._log_conversation(turn, user_id)
        return ChatResponse(text=text, intent=intent_tag, confidence=confidence, entities=entities)

    def _log_conversation(self, turn: ConversationTurn, user_id: str) -> None:
        self.conversation_history.append(turn)
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
        entry = {
            "user_id": user_id,
            "timestamp": turn.timestamp.isoformat(),
            "user_input": turn.user_input,
            "bot_response": turn.bot_response,
            "intent": turn.intent,
            "confidence": turn.confidence,
            "entities": turn.entities,
        }
        try:
            with self._log_lock:
                with open(self.conversation_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.error("Error logging conversation: %s", e)

    def get_conversation_analytics(self) -> Dict:
        if not self.conversation_history:
            return {}
        total = len(self.conversation_history)
        avg_conf = float(np.mean([t.confidence for t in self.conversation_history]))
        intent_counts = Counter(t.intent for t in self.conversation_history)
        return {
            "total_conversations": total,
            "average_confidence": round(avg_conf, 4),
            "intent_distribution": dict(intent_counts),
            "low_confidence_rate": round(
                sum(1 for t in self.conversation_history if t.confidence < self.confidence_threshold)
                / total,
                4,
            ),
            "unique_intents_used": len(intent_counts),
            "ai_name": self.ai_name,
        }


# Backwards-compatible name used by interface.py and existing configs.
EnhancedChatAssistant = ChatAssistant


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------


def load_or_train_assistant(
    epochs: int = 200,
    force_train: bool = False,
    generative: bool = True,
    gen_epochs: int = 30,
    device: Optional[str] = None,
) -> ChatAssistant:
    """Load saved models when possible, otherwise train them.

    The intent classifier is loaded from enhanced_model.pth; the generative
    transformer from generative_model.pth. Missing or stale artifacts are
    retrained automatically.
    """
    assistant = ChatAssistant(device=device)
    assistant.use_generative = generative

    if not force_train and MODEL_PATH.exists() and CONFIG_PATH.exists():
        try:
            assistant.load_model(str(MODEL_PATH), str(CONFIG_PATH))
        except Exception as e:
            logger.warning("Could not load saved model (%s); retraining.", e)

    if assistant.model is None:
        assistant.load_data()
        result = assistant.train(epochs=epochs)
        assistant.save_model(str(MODEL_PATH), str(CONFIG_PATH))
        logger.info(
            "Classifier trained in %d epochs | val_loss=%.4f val_acc=%.2f%%",
            result.epochs,
            result.best_val_loss,
            result.best_val_accuracy * 100,
        )

    if generative:
        if not force_train and GEN_MODEL_PATH.exists() and GEN_CONFIG_PATH.exists():
            try:
                assistant.load_generative(str(GEN_MODEL_PATH), str(GEN_CONFIG_PATH))
            except Exception as e:
                logger.warning("Could not load generative model (%s); retraining.", e)
        if assistant.gen_model is None:
            gen_result = assistant.train_generative(epochs=gen_epochs)
            assistant.save_generative(str(GEN_MODEL_PATH), str(GEN_CONFIG_PATH))
            logger.info(
                "Generative model trained on %d pairs | final_loss=%.4f",
                gen_result.num_pairs,
                gen_result.final_loss,
            )

    assistant.warmup()
    return assistant


def interactive_chat(assistant: ChatAssistant) -> None:
    import time

    print(f"{assistant.ai_name} is ready. Type 'quit' to exit, 'help' for capabilities.")
    print("=" * 60)
    while True:
        try:
            # explicit write+flush: input() prompt can get stuck in the
            # stdout buffer when the console isn't a real TTY
            sys.stdout.write("\nYou: ")
            sys.stdout.flush()
            raw = sys.stdin.readline()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        if raw == "":  # real EOF (piped stdin exhausted or Ctrl+Z)
            print("\nGoodbye!")
            break
        user_input = raw.strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "bye"}:
            print("Goodbye!")
            break

        streamed = {"used": False}

        def emit(piece: str) -> None:
            if not streamed["used"]:
                print(f"\n{assistant.ai_name}: ", end="", flush=True)
            streamed["used"] = True
            print(piece, end="", flush=True)
            time.sleep(0.008)  # typing effect

        response = assistant.handle_message(user_input, on_token=emit)
        if streamed["used"]:
            print(flush=True)  # newline after streamed text
        else:
            print(f"\n{assistant.ai_name}: {response.text}", flush=True)
        print(f"   [{response.intent} | {response.confidence:.2f}]", flush=True)


def main() -> None:
    args = sys.argv[1:]
    force_train = any(a in ("--train", "train") for a in args)
    generative = "--no-gen" not in args
    print("Loading models...", flush=True)
    assistant = load_or_train_assistant(
        force_train=force_train, generative=generative
    )
    if "--gen-all" in args:
        assistant.generative_all = True
        print("(full generative mode — the transformer writes every reply)")
    interactive_chat(assistant)


if __name__ == "__main__":
    main()
