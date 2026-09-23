"""Integration tests for ChatAssistant behavior that doesn't require
the trained model (rules, games, profile, context)."""

import pytest

from main import ChatAssistant


@pytest.fixture
def assistant(tmp_path):
    a = ChatAssistant(
        conversation_log_path=str(tmp_path / "conv.jsonl"),
        profile_path=str(tmp_path / "profile.json"),
    )
    # minimal intents so rule overrides have targets
    a.intents = ["greeting", "how_are_you", "joke", "play_game"]
    a.responses = {
        "greeting": ["Hi there!"],
        "how_are_you": ["Doing well!"],
        "joke": ["Here's a joke."],
        "play_game": ["Let's play!"],
    }
    a.extensions = {"joke": {"function": "extensions.jokes.getRandomJoke"}}
    a.entity_patterns = {}
    a.entity_types = {}
    a.contexts = {}
    return a


class TestCommands:
    def test_help(self, assistant):
        r = assistant.handle_message("help")
        assert r.intent == "help"
        assert "what i can do" in r.text.lower()

    def test_reset(self, assistant):
        assistant.handle_message("hello", "u1")
        r = assistant.handle_message("reset", "u1")
        assert r.intent == "reset"
        assert "u1" not in assistant.current_context


class TestNameMemory:
    def test_name_capture_and_recall(self, assistant):
        r = assistant.handle_message("my name is Sean", "u1")
        assert r.intent == "name_setup"
        assert "Sean" in r.text
        r2 = assistant.handle_message("what is my name", "u1")
        assert r2.intent == "name_query"
        assert "Sean" in r2.text

    def test_feeling_not_treated_as_name(self, assistant):
        r = assistant.handle_message("I am sad", "u1")
        assert r.intent != "name_setup"
        assert "name" not in assistant.user_profile

    def test_unknown_name_query(self, assistant):
        r = assistant.handle_message("what is my name", "u1")
        assert "don't know" in r.text


class TestAIName:
    def test_set_ai_name(self, assistant):
        r = assistant.handle_message("call yourself Jarvis")
        assert r.intent == "ai_name_setup"
        assert assistant.ai_name == "Jarvis"
        assert "Jarvis" in r.text


class TestMathRouting:
    def test_math_routes_to_evaluator(self, assistant):
        r = assistant.handle_message("what is 12 times 8")
        assert r.intent == "math_question"
        assert "96" in r.text


class TestGame:
    def test_number_game_flow(self, assistant):
        r = assistant.handle_message("play a game", "u1")
        # game may also arrive via intent; start explicitly
        assistant._start_number_game("u1")
        ctx = assistant._context_for("u1")
        target = ctx["active_game"]["target"]
        r = assistant.handle_message(str(target), "u1")
        assert r.intent == "play_game"
        assert "guessed" in r.text

    def test_quit_game(self, assistant):
        assistant._start_number_game("u1")
        r = assistant.handle_message("quit game", "u1")
        assert "over" in r.text.lower()
        assert "active_game" not in assistant._context_for("u1")

    def test_game_exits_on_other_intent(self, assistant):
        assistant._start_number_game("u1")
        r = assistant.handle_message("hello", "u1")
        assert r.intent != "play_game"
        assert "active_game" not in assistant._context_for("u1")

    def test_game_reprompts_on_gibberish(self, assistant):
        assistant._start_number_game("u1")
        r = assistant.handle_message("xyzzy", "u1")
        assert r.intent == "play_game"
        assert "active_game" in assistant._context_for("u1")

    def test_out_of_range_guess(self, assistant):
        assistant._start_number_game("u1")
        r = assistant.handle_message("12345", "u1")
        assert "between 1 and 50" in r.text

    def test_quit_words_without_game(self, assistant):
        r = assistant.handle_message("quit game", "u1")
        assert "No game" in r.text or "no game" in r.text


class TestBotNameQuery:
    def test_bot_name_query_does_not_rename(self, assistant):
        original = assistant.ai_name
        r = assistant.handle_message("what is your name", "u1")
        assert original in r.text
        assert assistant.ai_name == original
        r = assistant.handle_message("whats your name", "u1")
        assert assistant.ai_name == original


class TestMathRoutingExtra:
    def test_bare_number_not_math(self, assistant):
        r = assistant.handle_message("60", "u1")
        assert r.intent != "math_question"


class TestFollowup:
    def test_another_joke(self, assistant):
        assistant.handle_message("tell me a joke", "u1")
        assistant._update_context("joke", "tell me a joke", "u1")
        r = assistant.handle_message("another joke", "u1")
        assert r.intent == "followup"
        assert r.text


class TestAnalytics:
    def test_analytics_shape(self, assistant):
        assistant.handle_message("hello", "u1")
        stats = assistant.get_conversation_analytics()
        assert stats["total_conversations"] == 1
        assert "average_confidence" in stats
        assert "intent_distribution" in stats


class TestBackwardsCompat:
    def test_get_response_tuple(self, assistant):
        text, conf, entities, tag = assistant.get_response("hello")
        assert isinstance(text, str)
        assert isinstance(conf, float)
        assert tag == "greeting"
