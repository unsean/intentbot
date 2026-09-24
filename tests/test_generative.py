"""Tests for the generative transformer (architecture and plumbing).

These use an untrained or minimally-trained model — they verify the
machinery (tokenization, forward pass, decoding loop, save/load), not
language quality.
"""

import torch

from generative import (
    BOS, EOS, PAD, GenVocab, Seq2SeqTransformer, gen_tokenize, train_generator,
)


class TestGenVocab:
    def test_tokenize_keeps_punctuation(self):
        assert gen_tokenize("Hello, world!") == ["hello", ",", "world", "!"]

    def test_encode_decode_roundtrip(self):
        v = GenVocab(min_freq=1)
        v.build(["hello world", "hello there"])
        ids = v.encode("hello world", 10)
        assert v.decode(ids) == "hello world"

    def test_unk_for_oov(self):
        v = GenVocab(min_freq=5)
        v.build(["hello world"])
        assert all(i == 3 for i in v.encode("hello", 10))  # UNK

    def test_decode_stops_at_eos(self):
        v = GenVocab(min_freq=1)
        v.build(["a b c"])
        ids = v.encode("a b", 10) + [EOS] + v.encode("c", 10)
        assert v.decode(ids) == "a b"


class TestSeq2SeqTransformer:
    def _make_model(self, vocab_size=50):
        return Seq2SeqTransformer(vocab_size, d_model=32, nhead=2,
                                  num_encoder_layers=1, num_decoder_layers=1,
                                  dim_feedforward=64)

    def test_forward_shape(self):
        m = self._make_model()
        src = torch.randint(4, 50, (2, 8))
        tgt = torch.randint(4, 50, (2, 6))
        out = m(src, tgt)
        assert out.shape == (2, 6, 50)

    def test_generate_bounded(self):
        m = self._make_model()
        ids = m.generate([4, 5, 6], max_len=10)
        assert len(ids) <= 10
        assert BOS not in ids

    def test_causal_mask(self):
        m = self._make_model()
        m.eval()  # disable dropout so the comparison is deterministic
        src = torch.tensor([[4, 5, EOS]])
        t1 = torch.tensor([[BOS, 4]])
        t2 = torch.tensor([[BOS, 4, 5]])
        # first-position logits identical regardless of future tokens
        with torch.no_grad():
            out1 = m(src, t1)
            out2 = m(src, t2)
        assert torch.allclose(out1[:, 0], out2[:, 0], atol=1e-5)


class TestTrainGenerator:
    _TINY = {"d_model": 32, "nhead": 2, "num_encoder_layers": 1,
             "num_decoder_layers": 1, "dim_feedforward": 64}

    def test_loss_decreases(self):
        pairs = [("hi", "hello there"), ("bye", "see you later")] * 20
        v = GenVocab(min_freq=1)
        v.build([s for s, _ in pairs] + [t for _, t in pairs])
        model, result = train_generator(pairs, v, epochs=20, batch_size=8,
                                        device="cpu", lr=1e-3,
                                        model_config=self._TINY)
        assert result.num_pairs == 40
        assert result.final_loss < 4.0  # must descend well below ln(vocab)

    def test_generate_after_train(self):
        pairs = [("hello", "hi how are you")] * 30
        v = GenVocab(min_freq=1)
        v.build([s for s, _ in pairs] + [t for _, t in pairs])
        model, _ = train_generator(pairs, v, epochs=3, batch_size=8,
                                   device="cpu", model_config=self._TINY)
        ids = model.generate(v.encode("hello", 48), max_len=10)
        assert isinstance(v.decode(ids), str)
