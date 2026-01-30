# =========================
# PART 1/2
# =========================
import asyncio
import time
import os
import logging
import threading
from typing import Dict, List, Tuple, Any
from datetime import datetime
import statistics
import json
from collections import defaultdict
from pathlib import Path

import requests

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import (
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    CallbackQuery,
    Message,
)
from aiogram.client.default import DefaultBotProperties


# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("ALADDIN")


# =========================
# BOT SETUP (TOKEN FROM ENV)
# =========================
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("❌ BOT_TOKEN не найден в переменных окружения")

bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(parse_mode="Markdown")
)
dp = Dispatcher()


# =========================
# BYBIT DOMAINS (fallback)
# =========================
# Можно переопределить переменной окружения BYBIT_BASE,
# но даже без неё будет fallback api.bybit.com -> api.bytick.com
BYBIT_BASE = os.getenv("BYBIT_BASE", "").strip()

BYBIT_DOMAINS = []
if BYBIT_BASE:
    BYBIT_DOMAINS.append(BYBIT_BASE)

# дефолтные
BYBIT_DOMAINS += [
    "https://api.bybit.com",
    "https://api.bytick.com",
]


# =========================
# GLOBAL STATE + LOCKS
# =========================
state_lock = asyncio.Lock()
user_states_lock = asyncio.Lock()

# ✅ анти-гонки по кликам
user_action_lock = defaultdict(asyncio.Lock)

state = {
    "previous_signal": None,
    "previous_price": 0.0,
    "alert_chat_id": None,
}

user_states = defaultdict(lambda: {
    "agreed": False,
    "chat_id": None,
})


# =========================
# CACHE
# =========================
CACHE_SECONDS = 20
_analysis_cache = {"time": 0.0, "data": None}
_cache_lock = threading.Lock()


# =========================
# HISTORY
# =========================
HISTORY_FILE = Path("aladdin_history.json")


# =========================
# DISCLAIMER
# =========================
DISCLAIMER_TEXT = """
# 🔥 *ALADDIN v10.0 — 5000+ СВЕЧЕЙ ДАННЫХ!*

# ⚠️ *ОБЯЗАТЕЛЬНОЕ ПРЕДУПРЕЖДЕНИЕ ПЕРЕД ИСПОЛЬЗОВАНИЕМ*

#*📜 ПОЛЬЗОВАТЕЛЬСКОЕ СОГЛАСИЕ*

1. *Мы НЕ несем никакой ответственности за ваши деньги и торговые решения.*
2. *Бот предоставляется "как есть" БЕЗ ГАРАНТИЙ прибыли.*
3. *Все риски торговли ложатся исключительно на вас.*
4. *Используйте на свой страх и риск.*

#*🎯 Цель бота:* `игьнед ишав мебераз ыМ` (прочтите задом наперед)

#*⚠️ Нажимая "ОЗНАКОМЛЕН", вы подтверждаете:*
#✅ Прочитал и понял предупреждение
#✅ Принимаю ВСЕ риски на себя
#✅ Осознаю возможные убытки
#✅ Согласен с отсутствием гарантий
""".strip()


# =========================
# HTTP HELPER + BYBIT HELPER (FIXED)
# =========================
_session = requests.Session()
_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Railway; ALADDIN bot)",
    "Accept": "application/json",
})


def http_get_json(url: str, *, timeout: int = 10, retries: int = 2) -> Any:
    """
    Логирует HTTP ошибки (403/451/5xx) и первые 200 символов тела.
    """
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = _session.get(url, timeout=timeout)
            if r.status_code != 200:
                logger.error(f"HTTP {r.status_code} for {url} | body={r.text[:200]}")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
            else:
                raise last_err


def bybit_get_json(path: str, *, timeout: int = 12, retries: int = 1) -> Any:
    """
    Пробует домены Bybit по очереди: api.bybit.com -> api.bytick.com (или BYBIT_BASE).
    Возвращает json или None.
    """
    last_exc = None
    for base in BYBIT_DOMAINS:
        url = base.rstrip("/") + path
        try:
            return http_get_json(url, timeout=timeout, retries=retries)
        except Exception as e:
            last_exc = e
            logger.error(f"BYBIT domain failed: {base} | err={e}")
            continue

    if last_exc:
        logger.error(f"BYBIT all domains failed. last_err={last_exc}")
    return None


# =========================
# SAFE EDIT
# =========================
async def safe_edit_text(msg: Message, text: str, *, reply_markup=None, parse_mode="Markdown") -> None:
    try:
        await msg.edit_text(text, reply_markup=reply_markup, parse_mode=parse_mode)
    except Exception as e:
        s = str(e).lower()
        if (
            "message is not modified" in s
            or "message to edit not found" in s
            or "message can't be edited" in s
            or "message is too old" in s
        ):
            logger.info(f"safe_edit_text ignored: {e}")
            return
        raise


# =========================
# INDICATORS (SAFE)
# =========================
def calculate_ema(prices: List[float], period: int = 14) -> List[float]:
    if not prices:
        return []
    if period <= 1:
        return prices[:]
    mult = 2 / (period + 1)
    ema = [prices[0]]
    for p in prices[1:]:
        ema.append(p * mult + ema[-1] * (1 - mult))
    return ema


def calculate_wma(prices: List[float], period: int = 20) -> float:
    if not prices:
        return 0.0
    if len(prices) < period:
        return float(prices[-1])
    weights = list(range(1, period + 1))
    window = prices[-period:]
    return float(sum(p * w for p, w in zip(window, weights)) / sum(weights))


def calculate_bollinger(prices: List[float], period: int = 20, std: float = 2.0) -> Dict[str, float]:
    if not prices:
        return {"upper": 0.0, "lower": 0.0, "sma": 0.0}
    if len(prices) < period:
        p = float(prices[-1])
        return {"upper": p, "lower": p, "sma": p}

    window = prices[-period:]
    sma = float(sum(window) / period)
    try:
        std_dev = float(statistics.stdev(window))
    except statistics.StatisticsError:
        std_dev = 0.0

    return {"upper": sma + std * std_dev, "lower": sma - std * std_dev, "sma": sma}


def bb_position(price: float, bb: Dict[str, float]) -> float:
    upper = float(bb.get("upper", price))
    lower = float(bb.get("lower", price))
    if upper <= lower:
        return 50.0
    return float((price - lower) / (upper - lower) * 100.0)


def calculate_vwap(
    opens: List[float],
    highs: List[float],
    lows: List[float],
    closes: List[float],
    volumes: List[float],
    period: int = 20
) -> float:
    if not closes:
        return 0.0
    n = min(period, len(closes), len(highs), len(lows), len(volumes))
    if n <= 0:
        return float(closes[-1])

    typical = [(h + l + c) / 3.0 for h, l, c in zip(highs[-n:], lows[-n:], closes[-n:])]
    vol_sum = float(sum(volumes[-n:]))
    if vol_sum <= 0:
        return float(closes[-1])

    return float(sum(p * v for p, v in zip(typical, volumes[-n:])) / vol_sum)


def calculate_sar(highs: List[float], lows: List[float], af_step: float = 0.02, af_max: float = 0.2) -> float:
    if not highs or not lows or len(highs) < 2 or len(lows) < 2:
        return float(lows[-1]) if lows else 0.0

    lookback = min(10, len(highs) - 1)
    start = -lookback

    sar = float(lows[start - 1])
    ep = float(highs[start])
    af = float(af_step)

    for i in range(start, 0):
        if highs[i] > ep:
            ep = float(highs[i])
            af = min(af + af_step, af_max)
        sar = sar + af * (ep - sar)

    return float(sar)


def calculate_supertrend(highs: List[float], lows: List[float], closes: List[float], period: int = 10, mult: float = 3.0) -> float:
    if not highs or not lows or not closes:
        return 0.0
    n = min(period, len(highs), len(lows), len(closes))
    if n < 2:
        return float(closes[-1])

    hl2 = [(h + l) / 2.0 for h, l in zip(highs[-n:], lows[-n:])]

    tr_vals = []
    for i in range(-n + 1, 0):
        prev_close = closes[i - 1]
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - prev_close),
            abs(lows[i] - prev_close),
        )
        tr_vals.append(tr)

    atr = float(sum(tr_vals) / len(tr_vals)) if tr_vals else 0.0
    basic_upper = float(hl2[-1] + mult * atr)
    basic_lower = float(hl2[-1] - mult * atr)

    return basic_upper if (len(closes) > 1 and closes[-2] <= basic_upper) else basic_lower


def calculate_trix(prices: List[float], period: int = 14) -> float:
    if not prices or len(prices) < period * 2:
        return 0.0

    ema1 = calculate_ema(prices, period)
    ema2 = calculate_ema(ema1, period)
    ema3 = calculate_ema(ema2, period)

    if len(ema3) < 2 or ema3[-2] == 0:
        return 0.0

    return float((ema3[-1] - ema3[-2]) / ema3[-2] * 100.0)


def calculate_adx(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
    if len(highs) < period + 2 or len(lows) < period + 2 or len(closes) < period + 2:
        return 0.0

    tr_list: List[float] = []
    plus_dm: List[float] = []
    minus_dm: List[float] = []

    for i in range(1, len(closes)):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]

        pdm = up_move if (up_move > down_move and up_move > 0) else 0.0
        mdm = down_move if (down_move > up_move and down_move > 0) else 0.0

        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

        tr_list.append(float(tr))
        plus_dm.append(float(pdm))
        minus_dm.append(float(mdm))

    def wilder_smooth(values: List[float], p: int) -> List[float]:
        if len(values) < p:
            return []
        out = []
        first_sum = float(sum(values[:p]))
        out.append(first_sum)
        prev = first_sum
        for v in values[p:]:
            prev = prev - (prev / p) + float(v)
            out.append(prev)
        return out

    tr_sm = wilder_smooth(tr_list, period)
    pdm_sm = wilder_smooth(plus_dm, period)
    mdm_sm = wilder_smooth(minus_dm, period)

    if not tr_sm or not pdm_sm or not mdm_sm:
        return 0.0

    di_plus: List[float] = []
    di_minus: List[float] = []
    for t, p, m in zip(tr_sm, pdm_sm, mdm_sm):
        if t <= 0:
            di_plus.append(0.0)
            di_minus.append(0.0)
        else:
            di_plus.append(100.0 * (p / t))
            di_minus.append(100.0 * (m / t))

    dx: List[float] = []
    for p, m in zip(di_plus, di_minus):
        denom = p + m
        dx.append(0.0 if denom == 0 else (100.0 * abs(p - m) / denom))

    if len(dx) < period:
        return 0.0

    adx_sm = wilder_smooth(dx, period)
    if not adx_sm:
        return 0.0

    return float(adx_sm[-1])


def calculate_rsi(prices: List[float], period: int = 14) -> float:
    if not prices or len(prices) <= period:
        return 50.0

    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    last = deltas[-period:]

    gains = [d if d > 0 else 0.0 for d in last]
    losses = [-d if d < 0 else 0.0 for d in last]

    try:
        avg_gain = float(statistics.mean(gains))
        avg_loss = float(statistics.mean(losses))
    except Exception:
        return 50.0

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


# =========================
# HISTORY
# =========================
def load_history() -> List[Dict[str, Any]]:
    try:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
    except Exception as e:
        logger.error(f"Ошибка загрузки истории: {e}")
    return []


def save_history(history: List[Dict[str, Any]]):
    try:
        recent_history = history[-300:]
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(recent_history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Ошибка сохранения истории: {e}")


def analyze_past_predictions(history: List[Dict[str, Any]], step: int = 3) -> float:
    """
    step=3 -> сравнение с ценой через ~15 минут (если обновления каждые 5м)
    WAIT в точность не учитываем (и мы его не сохраняем вообще).
    """
    if len(history) < 2:
        return 50.0

    wins = 0
    losses = 0

    start_idx = max(0, len(history) - 80)

    for i in range(start_idx, len(history) - 1):
        if i + step >= len(history):
            break

        past = history[i]
        next_item = history[i + step]

        direction = str(past.get("direction", "")).upper()
        if "WAIT" in direction or "⚪" in direction:
            continue

        try:
            entry_price = float(past.get("price", 0) or 0)
            next_price = float(next_item.get("price", 0) or 0)
        except Exception:
            continue

        if entry_price <= 0 or next_price <= 0:
            continue

        is_long = ("LONG" in direction) or ("🟢" in direction)
        is_short = ("SHORT" in direction) or ("🔴" in direction)

        if is_long:
            if next_price > entry_price:
                wins += 1
            elif next_price < entry_price:
                losses += 1
        elif is_short:
            if next_price < entry_price:
                wins += 1
            elif next_price > entry_price:
                losses += 1

    total = wins + losses
    if total == 0:
        return 50.0

    return round((wins / total) * 100.0, 1)
# =========================
# PART 2/2
# =========================

# =========================
# TARGETS / RISK
# =========================
def calculate_targets_PRO(data: Dict[str, Any], direction: str) -> Tuple[float, float, float, float]:
    c = float(data.get("c", 0) or 0)
    if c <= 0:
        return 0.0, 0.0, 0.0, 0.0
    if direction.startswith("⚪"):
        return c, c, 0.0, 0.0

    h = float(data.get("h", c) or c)
    l = float(data.get("l", c) or c)
    rsi = float(data.get("rsi", 50) or 50)

    atr_pct = ((h - l) / c) * 100.0 if c else 0.0

    if rsi > 75:
        target_mult = 1.8
    elif rsi > 65:
        target_mult = 2.0
    else:
        target_mult = 2.4

    stop_mult = 0.9
    target_dist = atr_pct * target_mult
    stop_dist = atr_pct * stop_mult

    if ("LONG" in direction) or ("🟢" in direction):
        target_price = c * (1 + target_dist / 100.0)
        stop_price = c * (1 - stop_dist / 100.0)
        profit_pct = round((target_price - c) / c * 100.0, 1)
        loss_pct = round((stop_price - c) / c * 100.0, 1)
    else:
        target_price = c * (1 - target_dist / 100.0)
        stop_price = c * (1 + stop_dist / 100.0)
        profit_pct = round((c - target_price) / c * 100.0, 1)
        loss_pct = -round((stop_price - c) / c * 100.0, 1)

    return float(target_price), float(stop_price), float(profit_pct), float(loss_pct)


def calculate_risk(data: Dict[str, Any]) -> Tuple[int, List[str]]:
    risk_points = 0
    risk_factors: List[str] = []

    rsi = float(data.get("rsi", 50) or 50)
    spread = float(data.get("spread", 0) or 0)

    h = float(data.get("h", 0) or 0)
    l = float(data.get("l", 0) or 0)
    c = float(data.get("c", 0) or 0)

    vol_usdt = float(data.get("vol_usdt", data.get("vol", 0)) or 0)

    if rsi > 78 or rsi < 22:
        risk_points += 30
        risk_factors.append("🚨 RSI экстремальный")
    if spread > 35:
        risk_points += 20
        risk_factors.append("📉 Спред опасный")
    if c > 0 and (h - l) > c * 0.07:
        risk_points += 25
        risk_factors.append("💥 Волатильность экстремальная")

    if vol_usdt < 200_000_000:
        risk_points += 15
        risk_factors.append("📉 Низкий объем")

    return min(risk_points, 100), risk_factors


# =========================
# 4H TREND FILTER (BYBIT)
# =========================
def get_trend_filter_4h() -> Dict[str, float]:
    out = {"ema50_4h": 0.0, "ema200_4h": 0.0}
    try:
        resp = bybit_get_json(
            "/v5/market/kline?category=linear&symbol=BTCUSDT&interval=240&limit=300",
            timeout=12,
            retries=1
        )
        if not resp:
            logger.error("BYBIT 4h kline: NO RESPONSE")
            return out

        ret_code = resp.get("retCode")
        ret_msg = resp.get("retMsg")
        if ret_code not in (0, "0"):
            logger.error(f"BYBIT 4h kline retCode={ret_code} retMsg={ret_msg} resp={str(resp)[:250]}")
            return out

        lst = (((resp or {}).get("result") or {}).get("list")) or []
        if not isinstance(lst, list) or len(lst) < 210:
            logger.error(f"BYBIT 4h kline empty/short list: len={len(lst) if isinstance(lst, list) else 'NA'}")
            return out

        lst = list(reversed(lst))
        closes = [float(k[4]) for k in lst]  # [start, o, h, l, c, vol, turnover]

        ema50 = calculate_ema(closes, 50)
        ema200 = calculate_ema(closes, 200)
        out["ema50_4h"] = float(ema50[-1] if ema50 else closes[-1])
        out["ema200_4h"] = float(ema200[-1] if ema200 else closes[-1])

    except Exception as e:
        logger.exception(f"get_trend_filter_4h failed: {e}")
    return out


# =========================
# BYBIT DATA (5m)
# =========================
def get_btc_data() -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "o": 0.0,
        "h": 0.0,
        "l": 0.0,
        "c": 0.0,

        "vol_btc": 0.0,
        "vol_usdt": 0.0,
        "vol": 0.0,

        "spread": 0.0,

        "rsi": 50.0,
        "sma12": 0.0,
        "ma20": 0.0,
        "ema12": 0.0,
        "wma20": 0.0,

        "bb_position": 50.0,
        "vwap": 0.0,
        "sar": 0.0,
        "supertrend": 0.0,
        "trix": 0.0,
        "adx": 0.0,
        "vol_ratio": 1.0,
    }

    try:
        resp = bybit_get_json(
            "/v5/market/kline?category=linear&symbol=BTCUSDT&interval=5&limit=1000",
            timeout=12,
            retries=1
        )
        if not resp:
            logger.error("BYBIT 5m kline: NO RESPONSE")
            return data

        ret_code = resp.get("retCode")
        ret_msg = resp.get("retMsg")
        if ret_code not in (0, "0"):
            logger.error(f"BYBIT kline retCode={ret_code} retMsg={ret_msg} resp={str(resp)[:250]}")
            # подсказки по логам:
            # HTTP 403/451 -> домен режется
            # retCode=10006 -> лимит
            return data

        klines = (((resp or {}).get("result") or {}).get("list")) or []
        if not isinstance(klines, list) or len(klines) < 200:
            logger.error(f"BYBIT empty/short list (kline): len={len(klines) if isinstance(klines, list) else 'NA'}")
            return data

        klines = list(reversed(klines))

        # Bybit: [startTime, open, high, low, close, volume, turnover]
        opens = [float(k[1]) for k in klines]
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        closes = [float(k[4]) for k in klines]

        base_vols = [float(k[5]) for k in klines]   # BTC
        quote_vols = [float(k[6]) for k in klines]  # USDT turnover

        window = 144 if len(klines) >= 144 else len(klines)

        data["o"] = float(opens[-window])
        data["h"] = float(max(highs[-window:]))
        data["l"] = float(min(lows[-window:]))
        data["c"] = float(closes[-1])

        data["vol_btc"] = float(sum(base_vols[-window:]))
        data["vol_usdt"] = float(sum(quote_vols[-window:]))
        data["vol"] = data["vol_usdt"]

        data["sma12"] = float(sum(closes[-12:]) / 12) if len(closes) >= 12 else data["c"]
        data["ma20"] = float(sum(closes[-20:]) / 20) if len(closes) >= 20 else data["c"]

        ema12_list = calculate_ema(closes, 12)
        data["ema12"] = float(ema12_list[-1]) if ema12_list else data["c"]
        data["wma20"] = float(calculate_wma(closes, 20))

        data["rsi"] = float(calculate_rsi(closes[-200:]))

        bb = calculate_bollinger(closes, 20)
        data["bb_position"] = float(bb_position(data["c"], bb))

        data["vwap"] = float(calculate_vwap(opens, highs, lows, closes, base_vols, period=20))

        data["sar"] = float(calculate_sar(highs, lows))
        data["supertrend"] = float(calculate_supertrend(highs, lows, closes))

        data["trix"] = float(calculate_trix(closes))
        data["adx"] = float(calculate_adx(highs, lows, closes))

        if len(quote_vols) >= 40:
            avg_prev = sum(quote_vols[-40:-20]) / 20
            avg_curr = sum(quote_vols[-20:]) / 20
            data["vol_ratio"] = float(avg_curr / avg_prev) if avg_prev > 0 else 1.0
        else:
            data["vol_ratio"] = 1.0

        # spread через tickers
        tick = bybit_get_json(
            "/v5/market/tickers?category=linear&symbol=BTCUSDT",
            timeout=8,
            retries=1
        )
        if not tick:
            logger.error("BYBIT tickers: NO RESPONSE")
            return data

        t_code = tick.get("retCode")
        t_msg = tick.get("retMsg")
        if t_code not in (0, "0"):
            logger.error(f"BYBIT tickers retCode={t_code} retMsg={t_msg} resp={str(tick)[:250]}")
            return data

        tick_list = (((tick or {}).get("result") or {}).get("list")) or []
        if isinstance(tick_list, list) and tick_list:
            t0 = tick_list[0]
            bid = float(t0.get("bid1Price", data["c"]))
            ask = float(t0.get("ask1Price", data["c"]))
            data["spread"] = float(max(ask - bid, data["c"] * 0.0001))
        else:
            logger.error("BYBIT tickers list empty")

    except Exception as e:
        logger.exception(f"get_btc_data failed: {e}")

    return data


# =========================
# CORE ANALYSIS + CACHE
# =========================
def aladdin_PRO_analysis():
    data = get_btc_data()

    history = load_history()
    history_accuracy = analyze_past_predictions(history, step=3)

    trend = get_trend_filter_4h()
    ema200_4h = float(trend.get("ema200_4h", 0.0) or 0.0)

    score = 0.0
    signals: List[str] = []

    c = float(data.get("c", 0) or 0)
    rsi = float(data.get("rsi", 50) or 50)

    ma20 = float(data.get("ma20", c) or c)
    ema12 = float(data.get("ema12", c) or c)
    bb_pos = float(data.get("bb_position", 50) or 50)

    vwap = float(data.get("vwap", c) or c)
    supertrend = float(data.get("supertrend", c) or c)
    trix = float(data.get("trix", 0) or 0)
    adx = float(data.get("adx", 0) or 0)
    vol_ratio = float(data.get("vol_ratio", 1) or 1)

    if c > max(ma20, ema12):
        score += 0.8
        signals.append("📈 MA/EMA бычьи")
    elif c < min(ma20, ema12):
        score -= 0.8
        signals.append("📉 MA/EMA медвежьи")

    if bb_pos < 20:
        score += 1.0
        signals.append("📉 BOLL перепродан")
    elif bb_pos > 80:
        score -= 0.8
        signals.append("📈 BOLL перекуплен")

    if c > vwap:
        score += 0.6
        signals.append("💰 VWAP выше")
    elif c < vwap:
        score -= 0.6
        signals.append("💰 VWAP ниже")

    if c > supertrend:
        score += 1.2
        signals.append("🚀 SUPER бычий")
    elif c < supertrend:
        score -= 1.2
        signals.append("🚀 SUPER медвежий")

    if trix > 0:
        score += 0.6
        signals.append("⚡ TRIX бычий")
    elif trix < 0:
        score -= 0.6
        signals.append("⚡ TRIX медвежий")

    if adx > 30:
        score += 0.5
        signals.append("📊 ADX тренд")

    if vol_ratio > 1.5:
        score += 0.8
        signals.append("🔥 Volume spike")

    if ema200_4h > 0:
        trend_long_ok = c > ema200_4h
        trend_short_ok = c < ema200_4h
    else:
        trend_long_ok = True
        trend_short_ok = True
        signals.append("⚠️ 4H EMA200 недоступна → без фильтра тренда")

    flat_penalty = 0.0
    if adx < 18:
        flat_penalty = 1.0
        signals.append("😴 ФЛЭТ (ADX<18) → штраф к уверенности")

    if score >= 2.2 and trend_long_ok and (score - flat_penalty) >= 2.0:
        direction = "🟢 LONG"
        confidence = min((score - flat_penalty) * 18.0, 95.0)
        signals.append("✅ 4H тренд подтверждает LONG")
    elif score <= -2.2 and trend_short_ok and (abs(score) - flat_penalty) >= 2.0:
        direction = "🔴 SHORT"
        confidence = min((abs(score) - flat_penalty) * 18.0, 95.0)
        signals.append("✅ 4H тренд подтверждает SHORT")
    else:
        direction = "⚪ WAIT"
        confidence = 0.0
        signals.append("⏳ Недостаточно подтверждений → WAIT")

    risk_percent, risk_factors = calculate_risk(data)
    target, stop, profit_pct, loss_pct = calculate_targets_PRO(data, direction)

    # ✅ WAIT не сохраняем
    if not direction.startswith("⚪"):
        forecast = {
            "time": datetime.now().isoformat(),
            "price": c,
            "direction": direction,
            "confidence": confidence,
            "rsi": rsi,
            "risk": risk_percent,
            "sma12": float(data.get("sma12", c) or c),
        }
        history.append(forecast)
        save_history(history)

    alerts: List[str] = []
    prev_price = float(state.get("previous_price", 0) or 0)
    price_change = ((c - prev_price) / prev_price * 100.0) if prev_price else 0.0

    if state.get("previous_signal") and state["previous_signal"] != direction:
        alerts.append(f"🚨 СИГНАЛ СМЕНИЛСЯ: {state['previous_signal']} → {direction}")

    if abs(price_change) > 2.5:
        emoji = "📈" if price_change > 0 else "📉"
        alerts.append(f"{emoji} ДВИЖЕНИЕ {price_change:+.1f}%!")

    if rsi > 78:
        alerts.append("🔔 ⚠️ RSI ПЕРЕКУПЛЕН!")
    elif rsi < 22:
        alerts.append("🔔 🟢 RSI ПЕРЕПРОДАН!")

    state["previous_signal"] = direction
    state["previous_price"] = c

    return (
        data, direction, confidence, signals, risk_percent, risk_factors,
        history_accuracy, target, stop, profit_pct, loss_pct, alerts
    )


def aladdin_cached():
    now = time.time()
    with _cache_lock:
        cached = _analysis_cache.get("data")
        if cached is not None and (now - float(_analysis_cache["time"])) < CACHE_SECONDS:
            logger.info("✅ CACHE: данные свежие")
            return cached

    logger.info("🔄 CACHE: новый анализ")
    result = aladdin_PRO_analysis()

    with _cache_lock:
        _analysis_cache["data"] = result
        _analysis_cache["time"] = now

    return result


# =========================
# UI + HANDLERS + ALERT LOOP + MAIN
# =========================
def main_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📊 Анализ", callback_data="analyze")],
        [InlineKeyboardButton(text="📈 Индикаторы", callback_data="indicators")],
        [
            InlineKeyboardButton(text="⚠️ Риск", callback_data="risk"),
            InlineKeyboardButton(text="🚨 Активировать алерты", callback_data="alerts"),
        ],
    ])


def agreement_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="✅ ОЗНАКОМЛЕН", callback_data="agree_yes")],
        [InlineKeyboardButton(text="❌ Отказаться", callback_data="agree_no")],
    ])


@dp.message(Command("start"))
async def start_cmd(message: Message):
    user_id = message.from_user.id
    chat_id = message.chat.id

    async with user_states_lock:
        user_states[user_id]["chat_id"] = chat_id
        user_states[user_id]["agreed"] = False

    async with state_lock:
        state["alert_chat_id"] = chat_id

    await message.answer(DISCLAIMER_TEXT, reply_markup=agreement_keyboard())


@dp.callback_query(F.data == "agree_no")
async def decline_handler(callback: CallbackQuery):
    await callback.answer()
    await safe_edit_text(
        callback.message,
        "❌ *Согласие НЕ получено*\n\n"
        "# ⚠️ Бот не будет работать без принятия условий.\n"
        "Нажмите /start для повторного ознакомления.",
        reply_markup=None,
        parse_mode="Markdown",
    )


@dp.callback_query(F.data == "agree_yes")
async def agree_handler(callback: CallbackQuery):
    user_id = callback.from_user.id
    await callback.answer()

    async with user_states_lock:
        user_states[user_id]["agreed"] = True

    await safe_edit_text(
        callback.message,
        "*✅ СОГЛАСИЕ ПРИНЯТО!*\n\n"
        "# 📊 ALADDIN активирован\n\n"
        "# 📊 Выбери действие:",
        reply_markup=main_keyboard(),
        parse_mode="Markdown",
    )


@dp.callback_query(F.data == "analyze")
async def analyze_cb(callback: CallbackQuery):
    lock = user_action_lock[callback.from_user.id]
    if lock.locked():
        await callback.answer("⏳ Подожди...", show_alert=False)
        return

    async with lock:
        await callback.answer()
        await safe_edit_text(callback.message, "⏳ Анализ...", reply_markup=None)

        try:
            (
                data, direction, conf, signals, risk, risk_factors,
                hist_acc, target, stop, profit, loss, alerts
            ) = await asyncio.to_thread(aladdin_cached)

            c = float(data.get("c", 0) or 0)
            h = float(data.get("h", c) or c)
            l = float(data.get("l", c) or c)

            rr = abs(profit / loss) if loss not in (0.0, -0.0) else 0.0
            vola = ((h - l) / c * 100.0) if c else 0.0

            analysis_text = f"""# 📊 АНАЛИЗ

# {direction} (`{conf:.1f}%`)
# 📊 RSI: `{float(data.get('rsi', 50)):.1f}`
# 🌊 Volat(12h): `{vola:.1f}%`
# 📈 MA20: `{float(data.get('ma20', c)):,.0f}$`

# ⚡ EMA12: `{float(data.get('ema12', c)):,.0f}$`
# 📉 WMA20: `{float(data.get('wma20', c)):,.0f}$`
# 🎯 BOLL: `{float(data.get('bb_position', 50)):.0f}%`
# 💰 VWAP: `{float(data.get('vwap', c)):,.0f}$`

# 💎 *TRADE PLAN:*
# 🚪 Entry: `{c:,.0f}$`
# 🎯 Target: `{target:,.0f}$` ({profit:+.1f}%)
# 🛑 Stop: `{stop:,.0f}$` ({loss:+.1f}%)
# ⚖️ R:R `{rr:.1f}:1`
"""
            await safe_edit_text(callback.message, analysis_text, parse_mode="Markdown")

        except Exception as e:
            logger.exception(f"analyze_cb error: {e}")
            await safe_edit_text(callback.message, f"❌ Ошибка анализа: {str(e)[:120]}", parse_mode="Markdown")

        await callback.message.answer("Что дальше?", reply_markup=main_keyboard())


@dp.callback_query(F.data == "indicators")
async def indicators_cb(callback: CallbackQuery):
    lock = user_action_lock[callback.from_user.id]
    if lock.locked():
        await callback.answer("⏳ Подожди...", show_alert=False)
        return

    async with lock:
        await callback.answer()
        await safe_edit_text(callback.message, "⏳ Индикаторы...", reply_markup=None)

        try:
            data, direction, conf, _, _, _, _, _, _, _, _, _ = await asyncio.to_thread(aladdin_cached)
            c = float(data.get("c", 0) or 0)

            indicators_text = f"""# 📈 *ИНДИКАТОРЫ* — что они значат? 🤔

# 🔥 *ОСНОВНОЙ СИГНАЛ:* `{direction}` `{conf:.1f}%`

────────────────────

# 📊 *RSI: `{float(data.get('rsi', 50)):.1f}`*
#✅ 30-70 = нормально
#🟢 <30 = перепродано
#🔴 >70 = перекуплено

# 📈 *MA20: `{float(data.get('ma20', c)):,.0f}$`*
Цена выше = рост 📈
Цена ниже = падение 📉

# ⚡ *EMA12: `{float(data.get('ema12', c)):,.0f}$`*
Быстрая линия тренда

# 📉 *WMA20: `{float(data.get('wma20', c)):,.0f}$`*
Быстрее реагирует на движение

# 🎯 *BOLL: `{float(data.get('bb_position', 50)):.0f}%`*
#🟢 <20% = дёшево
#🔴 >80% = дорого

# 💰 *VWAP: `{float(data.get('vwap', c)):,.0f}$`*
Цена выше = выше средней

# 🚀 *SAR: `{float(data.get('sar', c)):,.0f}$`*
Цена выше SAR = рост 🟢

# ⚡ *TRIX: `{float(data.get('trix', 0)):.2f}`*
#🟢 >0 = импульс вверх
#🔴 <0 = импульс вниз

# 📊 *ADX: `{float(data.get('adx', 0)):.1f}`*
<18 = флэт 😴
>25 = тренд ✅
"""
            await safe_edit_text(callback.message, indicators_text, parse_mode="Markdown")

        except Exception as e:
            logger.exception(f"indicators_cb error: {e}")
            await safe_edit_text(callback.message, "❌ Ошибка индикаторов", parse_mode="Markdown")

        await callback.message.answer("Что дальше?", reply_markup=main_keyboard())


@dp.callback_query(F.data == "risk")
async def risk_cb(callback: CallbackQuery):
    lock = user_action_lock[callback.from_user.id]
    if lock.locked():
        await callback.answer("⏳ Подожди...", show_alert=False)
        return

    async with lock:
        await callback.answer()
        await safe_edit_text(callback.message, "⏳ Риск...", reply_markup=None)

        try:
            (
                data, direction, conf, signals, risk, risk_factors,
                hist_acc, target, stop, profit, loss, alerts
            ) = await asyncio.to_thread(aladdin_cached)

            c = float(data.get("c", 0) or 0)
            h = float(data.get("h", c) or c)
            l = float(data.get("l", c) or c)
            vola = ((h - l) / c * 100.0) if c else 0.0

            factors_text = "\n".join(risk_factors) if risk_factors else "✅ Низкий риск"

            vol_usdt = float(data.get("vol_usdt", data.get("vol", 0)) or 0)
            vol_btc = float(data.get("vol_btc", 0) or 0)

            risk_text = f"""# ⚠️ *РИСК ({risk}%) & СТАТИСТИКА*
─────────────────
# 🎚️ RSI: `{float(data.get('rsi', 50)):.1f}`
# 📊 Volat: `{vola:.1f}%`
# 📈 History: `{hist_acc}%`

# 💰 Volume(12h): `{vol_usdt:,.0f}$`
# 🪙 Volume BTC(12h): `{vol_btc:,.0f} BTC`

# 🔍 *Факторы риска:*
{factors_text}
"""

            if signals:
                risk_text += "\n# 📌 *Сигналы:*\n" + "\n".join([f"• {s}" for s in signals[:12]])

            await safe_edit_text(callback.message, risk_text, parse_mode="Markdown")

        except Exception as e:
            logger.exception(f"risk_cb error: {e}")
            await safe_edit_text(callback.message, "❌ Ошибка рисков", parse_mode="Markdown")

        await callback.message.answer("Что дальше?", reply_markup=main_keyboard())


@dp.callback_query(F.data == "alerts")
async def alerts_cb(callback: CallbackQuery):
    lock = user_action_lock[callback.from_user.id]
    if lock.locked():
        await callback.answer("⏳ Подожди...", show_alert=False)
        return

    async with lock:
        async with state_lock:
            state["alert_chat_id"] = callback.message.chat.id

        await callback.answer()
        alerts_text = """ 🚨 *АЛЕРТЫ АКТИВНЫ* ✅

# ⏰ *Каждые 5 минут проверка:*
#🔄 Смена сигнала LONG/SHORT/WAIT
#📈 Движение >2.5%
#🚨 RSI >78 / <22
"""
        await safe_edit_text(callback.message, alerts_text, parse_mode="Markdown")
        await callback.message.answer("Что дальше?", reply_markup=main_keyboard())


async def alert_loop():
    while True:
        try:
            async with state_lock:
                chat_id = state.get("alert_chat_id")

            if chat_id:
                _, direction, _, _, _, _, _, _, _, _, _, alerts = await asyncio.to_thread(aladdin_cached)

                for alert in alerts:
                    if direction.startswith("⚪") and "СИГНАЛ СМЕНИЛСЯ" in alert:
                        continue
                    await bot.send_message(chat_id, f"# 🚨 *PRO АЛЕРТ*\n# {alert}", parse_mode="Markdown")
                    await asyncio.sleep(1)

        except Exception as e:
            logger.exception(f"alert_loop error: {e}")

        await asyncio.sleep(300)


async def main():
    logger.info("🚀 ALADDIN — запуск")
    alert_task = asyncio.create_task(alert_loop())
    try:
        await dp.start_polling(bot)
    finally:
        alert_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
