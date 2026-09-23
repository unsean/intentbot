"""Tests for text preprocessing and vectorization."""

from main import Preprocessor


class TestPreprocessor:
    def setup_method(self):
        self.pre = Preprocessor()

    def test_unigram_tokens_drop_stopwords(self):
        tokens = self.pre.unigram_tokens("What is the weather today?")
        assert "the" not in tokens
        assert "weather" in tokens

    def test_digits_preserved(self):
        tokens = self.pre.unigram_tokens("I am 25 years old")
        assert "25" in tokens

    def test_math_operators_become_words(self):
        tokens = self.pre.unigram_tokens("what is 5 + 5")
        assert "plus" in tokens
        assert "5" in tokens

    def test_bigram_tokens_keep_order(self):
        bigrams = self.pre.bigram_tokens("my name is Sean")
        assert "my name" in bigrams

    def test_fit_and_vectorize(self):
        docs = [
            (["hello", "there"], ["hello there"], "greeting"),
            (["hello", "there"], ["hello there"], "greeting"),
            (["goodbye", "friend"], ["goodbye friend"], "goodbye"),
            (["goodbye", "friend"], ["goodbye friend"], "goodbye"),
        ]
        self.pre.fit([(u, b) for u, b, _ in docs])
        vec = self.pre.vectorize("hello there")
        assert len(vec) == len(self.pre.vocabulary)
        assert sum(vec) >= 2  # unigram + bigram both fire

    def test_state_roundtrip(self):
        docs = [
            (["aa", "bb"], ["aa bb"], "x"),
            (["aa", "bb"], ["aa bb"], "x"),
            (["cc"], [], "y"),
            (["cc"], [], "y"),
        ]
        self.pre.fit([(u, b) for u, b, _ in docs])
        state = self.pre.state_dict()
        other = Preprocessor()
        other.load_state_dict(state)
        assert other.vocabulary == self.pre.vocabulary
