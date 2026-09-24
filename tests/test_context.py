"""Tests for context carry-over, think traces, and the LLM client."""

import os
from unittest.mock import patch

import llm
import main as m


def _assistant():
    a = m.ChatAssistant()
    a.use_generative = False
    a.load_data()
    return a


class TestPendingContext:
    def test_weather_pending_then_city(self):
        a = _assistant()
        r1 = a.handle_message("weather", "u1")
        assert "which city" in r1.text.lower()
        ctx = a.current_context["u1"]
        assert ctx["pending"] and ctx["pending"]["intent"] == "weather"

    def test_pending_cleared_on_new_command(self):
        a = _assistant()
        a.handle_message("weather", "u1")
        r = a.handle_message("tell me a joke", "u1")
        assert r.intent != "weather"
        assert a.current_context["u1"]["pending"] is None

    def test_no_pending_by_default(self):
        a = _assistant()
        a.handle_message("hello there", "u1")
        assert a.current_context["u1"]["pending"] is None

    def test_pending_strips_hedging(self):
        a = _assistant()
        a.handle_message("weather", "u1")
        with patch("main.tools.get_weather", return_value="ok") as gw:
            a.handle_message("i think Mexico", "u1")
        gw.assert_called_once_with("weather in Mexico")

    def test_pending_survives_tool_failure(self):
        a = _assistant()
        a.handle_message("weather", "u1")
        fail = "I couldn't find a place called 'nowhere'."
        with patch("main.tools.get_weather", return_value=fail):
            a.handle_message("nowhere", "u1")
        assert a.current_context["u1"]["pending"] is not None


class TestHistoryRecording:
    def test_bot_side_recorded(self):
        a = _assistant()
        a.handle_message("hello there", "u1")
        hist = a.current_context["u1"]["conversation_history"]
        assert hist and hist[-1].get("bot")

    def test_history_capped(self):
        a = _assistant()
        for i in range(15):
            a.handle_message("hello there", "u1")
        assert len(a.current_context["u1"]["conversation_history"]) <= 10


class TestThinkTrace:
    def test_trace_empty_when_off(self):
        a = _assistant()
        a.handle_message("tell me a joke", "u1")
        assert a.last_trace == []

    def test_trace_records_route(self):
        a = _assistant()
        a.think = True
        a.handle_message("tell me a joke", "u1")
        assert any("classifier" in s or "rule" in s for s in a.last_trace)


class TestThoughtSplit:
    def test_splits_think_and_answer(self):
        a = _assistant()
        raw = "think intent is joke. they mention humor. answer why did the chicken cross"
        assert a._split_thought(raw) == "Why did the chicken cross"
        assert "Intent is joke" in a.last_thought

    def test_bracketed_form(self):
        a = _assistant()
        raw = "[think] reasoning here [answer] real reply"
        assert a._split_thought(raw) == "Real reply"
        assert a.last_thought == "Reasoning here"

    def test_no_markers_passes_through(self):
        a = _assistant()
        assert a._split_thought("just a normal reply") == "Just a normal reply"
        assert a.last_thought == ""

    def test_thought_reset_each_call(self):
        a = _assistant()
        a._split_thought("think reasoning answer reply")
        a._split_thought("plain text")
        assert a.last_thought == ""


class TestLLMClient:
    def test_unconfigured_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            c = llm.LLMClient()
            assert not c.configured
            assert not c.available()

    def test_configured_with_env(self):
        env = {"AICHAT_LLM_URL": "http://x:1/v1", "AICHAT_LLM_MODEL": "m"}
        with patch.dict(os.environ, env, clear=True):
            c = llm.LLMClient()
            assert c.configured

    def test_chat_returns_none_on_failure(self):
        env = {"AICHAT_LLM_URL": "http://x:1/v1", "AICHAT_LLM_MODEL": "m"}
        with patch.dict(os.environ, env, clear=True):
            c = llm.LLMClient()
            c._reachable = True
            with patch.object(c, "_post", side_effect=OSError("down")):
                assert c.chat([{"role": "user", "content": "hi"}]) is None
                assert c._reachable is False

    def test_chat_parses_response(self):
        env = {"AICHAT_LLM_URL": "http://x:1/v1", "AICHAT_LLM_MODEL": "m"}
        with patch.dict(os.environ, env, clear=True):
            c = llm.LLMClient()
            payload = {"choices": [{"message": {"content": "hello!"}}]}
            with patch.object(c, "_post", return_value=payload):
                assert c.chat([{"role": "user", "content": "hi"}]) == "hello!"
