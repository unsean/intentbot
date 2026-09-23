"""Tests for tools.py — mocked HTTP so no network is needed."""

import json
from unittest.mock import patch

import tools


def _fake_json(payload):
    return payload


class TestArgExtraction:
    def test_query_strips_search_trigger(self):
        assert tools._query_after_trigger("search for black holes") == "black holes"

    def test_query_strips_wiki(self):
        assert tools._query_after_trigger("wikipedia Alan Turing") == "Alan Turing"

    def test_query_strips_cari(self):
        assert tools._query_after_trigger("cari gunung bromo") == "gunung bromo"

    def test_city_extraction(self):
        m = tools._CITY_RE.search("weather in New York")
        assert m.group(1) == "New York"

    def test_city_extraction_indonesian(self):
        m = tools._CITY_RE.search("cuaca di Jakarta")
        assert m.group(1) == "Jakarta"

    def test_coin_extraction(self):
        m = tools._COIN_RE.search("price of ethereum")
        assert (m.group(1) or m.group(2)) == "ethereum"


class TestToolsOffline:
    """Every tool must degrade gracefully when the network is down."""

    def test_web_search_offline(self):
        with patch.object(tools, "_fetch_json", side_effect=OSError("no net")):
            out = tools.web_search("search for cats")
        assert isinstance(out, str) and len(out) > 0

    def test_weather_offline(self):
        with patch.object(tools, "_fetch_json", side_effect=OSError("no net")):
            out = tools.get_weather("weather in Paris")
        assert "weather" in out.lower() or "couldn't" in out.lower()

    def test_crypto_offline(self):
        with patch.object(tools, "_fetch_json", side_effect=OSError("no net")):
            out = tools.crypto_price("bitcoin price")
        assert "couldn't" in out.lower()

    def test_wiki_offline(self):
        with patch.object(tools, "_fetch_json", side_effect=OSError("no net")):
            out = tools.wiki_lookup("wikipedia cats")
        assert isinstance(out, str) and len(out) > 0


class TestToolsMocked:
    def test_weather_happy_path(self):
        def fake(url, timeout=6):
            if "geocoding" in url:
                return {"results": [{"latitude": 35.7, "longitude": 139.7,
                                     "name": "Tokyo", "country": "Japan"}]}
            return {"current": {"temperature_2m": 20.0, "weather_code": 0,
                                "relative_humidity_2m": 50, "wind_speed_10m": 5.0}}
        with patch.object(tools, "_fetch_json", side_effect=fake):
            out = tools.get_weather("weather in Tokyo")
        assert "Tokyo" in out and "20.0" in out and "clear" in out.lower()

    def test_weather_no_city_asks(self):
        out = tools.get_weather("what is the weather")
        assert "which city" in out.lower()

    def test_crypto_happy_path(self):
        fake = lambda url, timeout=6: {"bitcoin": {"usd": 84000, "idr": 1.4e9,
                                                   "usd_24h_change": -1.5}}
        with patch.object(tools, "_fetch_json", side_effect=fake):
            out = tools.crypto_price("bitcoin price")
        assert "Bitcoin" in out and "84,000" in out and "-1.5" in out

    def test_crypto_unknown_coin(self):
        out = tools.crypto_price("price of blorp")
        assert "which coin" in out.lower()

    def test_wiki_happy_path(self):
        def fake(url, timeout=6):
            if "opensearch" in url:
                return ["turing", ["Alan Turing"], [], []]
            return {"extract": "Alan Turing was a mathematician.",
                    "content_urls": {"desktop": {"page": "https://x"}}}
        with patch.object(tools, "_fetch_json", side_effect=fake):
            out = tools.wiki_lookup("wikipedia Alan Turing")
        assert "Alan Turing" in out and "mathematician" in out

    def test_datetime_runs(self):
        out = tools.datetime_info("what time is it")
        assert "local time" in out
