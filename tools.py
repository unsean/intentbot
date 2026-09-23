"""Real-world tools for the chatbot — web search, weather, crypto, datetime.

All APIs are free and need no key:
- DuckDuckGo Instant Answers + Wikipedia REST for search/lookup
- Open-Meteo (geocoding + forecast) for weather
- CoinGecko for crypto prices

Every public function takes `message` (the raw user text), extracts its own
argument, times out fast, and degrades to a friendly string on failure.
"""

import json
import logging
import re
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

_TIMEOUT = 6
_HEADERS = {"User-Agent": "aichatbox/1.0 (portfolio chatbot)"}


def _fetch_json(url: str, timeout: int = _TIMEOUT):
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Argument extraction
# ---------------------------------------------------------------------------

_SEARCH_TRIGGERS = re.compile(
    r"^(?:please\s+)?(?:can you\s+|could you\s+)?"
    r"(?:search|google|look\s?up|find|cari|telusuri|wikipedia|wiki)\s*"
    r"(?:for|about|info(?:rmation)? (?:on|about)|tentang)?\s*",
    re.IGNORECASE,
)


def _query_after_trigger(message: str) -> str:
    """Strip leading command words to get the search topic."""
    return _SEARCH_TRIGGERS.sub("", message.strip()).strip(" ?.!")


_CITY_RE = re.compile(
    r"\b(?:in|at|for|di)\s+([a-zA-Z][a-zA-Z .'-]*?)\s*(?:[?!.,]|today|now|skrg|sekarang|$)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Web search (DuckDuckGo instant answers -> Wikipedia fallback)
# ---------------------------------------------------------------------------

def web_search(message: str = "", **_) -> str:
    query = _query_after_trigger(message)
    if not query:
        return "What should I search for? Try 'search for black holes'."
    try:
        url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode(
            {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
        )
        data = _fetch_json(url)
        abstract = (data.get("AbstractText") or "").strip()
        if abstract:
            source = data.get("AbstractSource") or "DuckDuckGo"
            return f"{abstract} [{source}]"
        answer = (data.get("Answer") or "").strip()
        if answer:
            return answer
        for topic in data.get("RelatedTopics") or []:
            text = topic.get("Text") if isinstance(topic, dict) else None
            if text:
                return f"{text} [DuckDuckGo]"
    except Exception as e:
        logger.warning("DuckDuckGo search failed: %s", e)
    return wiki_lookup(query)


def wiki_lookup(message: str = "", **_) -> str:
    query = _query_after_trigger(message)
    if not query:
        return "What should I look up? Try 'wikipedia Albert Einstein'."
    try:
        hits = _fetch_json(
            "https://en.wikipedia.org/w/api.php?"
            + urllib.parse.urlencode(
                {"action": "opensearch", "search": query, "limit": 1,
                 "namespace": 0, "format": "json"}
            )
        )
        titles = hits[1] if isinstance(hits, list) and len(hits) > 1 else []
        if not titles:
            return f"I couldn't find a Wikipedia article about '{query}'."
        title = titles[0]
        summary = _fetch_json(
            "https://en.wikipedia.org/api/rest_v1/page/summary/"
            + urllib.parse.quote(title)
        )
        extract = (summary.get("extract") or "").strip()
        if not extract:
            return f"Found '{title}' on Wikipedia but couldn't read the summary."
        link = summary.get("content_urls", {}).get("desktop", {}).get("page", "")
        return f"{extract}  (more: {link})" if link else extract
    except Exception as e:
        logger.warning("Wikipedia lookup failed: %s", e)
        return "I couldn't reach Wikipedia right now - try again in a bit."


# ---------------------------------------------------------------------------
# Weather (Open-Meteo: geocoding + current conditions)
# ---------------------------------------------------------------------------

_WMO_CODES = {
    0: "clear sky", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light showers", 81: "showers", 82: "heavy showers",
    85: "snow showers", 86: "snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with hail",
}


def get_weather(message: str = "", **_) -> str:
    m = _CITY_RE.search(message or "")
    city = m.group(1).strip() if m else None
    if not city:
        return "Which city? Try 'weather in Jakarta'."
    try:
        geo = _fetch_json(
            "https://geocoding-api.open-meteo.com/v1/search?"
            + urllib.parse.urlencode({"name": city, "count": 1})
        )
        results = geo.get("results") or []
        if not results:
            return f"I couldn't find a place called '{city}'."
        place = results[0]
        wx = _fetch_json(
            "https://api.open-meteo.com/v1/forecast?"
            + urllib.parse.urlencode(
                {
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "current": "temperature_2m,weather_code,relative_humidity_2m,wind_speed_10m",
                }
            )
        )
        cur = wx.get("current") or {}
        desc = _WMO_CODES.get(cur.get("weather_code"), "unknown conditions")
        name = place.get("name", city)
        country = place.get("country", "")
        loc = f"{name}, {country}" if country else name
        return (
            f"{loc}: {cur.get('temperature_2m')}C, {desc}, "
            f"humidity {cur.get('relative_humidity_2m')}%, "
            f"wind {cur.get('wind_speed_10m')} km/h."
        )
    except Exception as e:
        logger.warning("Weather lookup failed: %s", e)
        return "I couldn't reach the weather service right now - try again in a bit."


# ---------------------------------------------------------------------------
# Crypto prices (CoinGecko)
# ---------------------------------------------------------------------------

_COIN_IDS = {
    "btc": "bitcoin", "bitcoin": "bitcoin",
    "eth": "ethereum", "ethereum": "ethereum",
    "doge": "dogecoin", "dogecoin": "dogecoin",
    "sol": "solana", "solana": "solana",
    "bnb": "binancecoin", "binance": "binancecoin",
    "xrp": "ripple", "ripple": "ripple",
    "ada": "cardano", "cardano": "cardano",
    "ltc": "litecoin", "litecoin": "litecoin",
    "usdt": "tether", "tether": "tether",
    "dot": "polkadot", "polkadot": "polkadot",
    "link": "chainlink", "chainlink": "chainlink",
    "matic": "matic-network", "polygon": "matic-network",
}

_COIN_RE = re.compile(
    r"(?:price of|harga|how much (?:is|does)|berapa harga)\s+([a-zA-Z]+)"
    r"|\b([a-zA-Z]+)\s+(?:price|harga)\b",
    re.IGNORECASE,
)


def crypto_price(message: str = "", **_) -> str:
    m = _COIN_RE.search(message or "")
    coin_word = None
    if m:
        coin_word = (m.group(1) or m.group(2) or "").lower()
    coin_id = _COIN_IDS.get(coin_word or "")
    if not coin_id:
        known = ", ".join(sorted({v for v in _COIN_IDS.values()}))
        return f"Which coin? I know: {known}."
    try:
        data = _fetch_json(
            "https://api.coingecko.com/api/v3/simple/price?"
            + urllib.parse.urlencode(
                {"ids": coin_id, "vs_currencies": "usd,idr",
                 "include_24hr_change": "true"}
            )
        )
        row = data.get(coin_id) or {}
        usd = row.get("usd")
        if usd is None:
            return f"No price data for {coin_id} right now."
        change = row.get("usd_24h_change")
        arrow = ""
        if isinstance(change, (int, float)):
            arrow = f" ({'+' if change >= 0 else ''}{change:.1f}% 24h)"
        idr = row.get("idr")
        idr_txt = f" / Rp{idr:,.0f}" if isinstance(idr, (int, float)) else ""
        return f"{coin_id.capitalize()}: ${usd:,.2f}{idr_txt}{arrow}"
    except Exception as e:
        logger.warning("Crypto lookup failed: %s", e)
        return "I couldn't reach the price service right now - try again in a bit."


# ---------------------------------------------------------------------------
# Date & time (local — upgraded to include both)
# ---------------------------------------------------------------------------

def datetime_info(message: str = "", **_) -> str:
    now = datetime.now()
    return (
        f"It's {now.strftime('%A, %B %d, %Y')} - "
        f"local time {now.strftime('%H:%M')}."
    )
