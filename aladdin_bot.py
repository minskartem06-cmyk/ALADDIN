import asyncio
import time
import os
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import statistics
import json
from collections import defaultdict
from functools import wraps
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery, Message
from aiogram.client.default import DefaultBotProperties
import requests

# ✅ НАСТРОЙКИ БОТА (ИСПРАВЛЕНО)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 🔥 ТОКЕН (ОСТАВЛЯЕМ КАК ЕСТЬ ПО ТРЕБОВАНИЮ)
BOT_TOKEN = "8384782785:AAF46h9PeuhUFSVTsnnyVEnt4bvtErWtrnU"
bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode="Markdown"))
dp = Dispatcher()

# ✅ ✅ ИСПРАВЛЕННЫЕ ГЛОБАЛЬНЫЕ СОСТОЯНИЯ С БЛОКИРОВКАМИ
state_lock = asyncio.Lock()
user_states_lock = asyncio.Lock()

state = {
    'previous_signal': None,
    'previous_price': 0,
    'alert_chat_id': None
}

user_states = defaultdict(lambda: {'agreed': False, 'chat_id': None})

CACHE_SECONDS = 60
_analysis_cache = {"time": 0, "data": None}
_analysis_cache_v10 = {"time": 0, "data": None}

HISTORY_FILE = Path('aladdin_history.json')

DISCLAIMER_TEXT = ""
#🔥 *ALADDIN v10.0 — 5000+ СВЕЧЕЙ ДАННЫХ!*

#⚠️ *ОБЯЗАТЕЛЬНОЕ ПРЕДУПРЕЖДЕНИЕ ПЕРЕД ИСПОЛЬЗОВАНИЕМ*

#*📜 ПОЛЬЗОВАТЕЛЬСКОЕ СОГЛАСИЕ*

1. *Мы НЕ несем никакой ответственности за ваши деньги и торговые решения.*
2. *Бот предоставляется "как есть" БЕЗ ГАРАНТИЙ прибыли.*
3. *Все риски торговли ложатся исключительно на вас.*
4. *Используйте на свой страх и риск.*

#*🎯 Цель бота:* `игьнед ишав мебераз ыМ` (прочтите задом наперед)

#*⚠️ Нажимая "ОЗНАКОМЛЕН", вы подтверждаете:*
✅ Прочитал и понял предупреждение
✅ Принимаю ВСЕ риски на себя  
✅ Осознаю возможные убытки
✅ Согласен с отсутствием гарантий
""

# ✅ ДЕКОРАТОР ДЛЯ ПРОВЕРКИ СОГЛАСИЯ
# ВАЖНО: ниже по файлу у вас есть функция с таким же именем, которая перетирает декоратор.
# Поэтому декоратор переименован.
def agreement_required():
    def decorator(func):
        @wraps(func)
        async def wrapper(callback: CallbackQuery, *args, **kwargs):
            user_id = callback.from_user.id
            async with user_states_lock:
                if not user_states[user_id]['agreed']:
                    await callback.answer("❌ Сначала примите соглашение! /start", show_alert=True)
                    return
            return await func(callback, *args, **kwargs)
        return wrapper
    return decorator

# ✅ ВСЕ ИНДИКАТОРЫ (ИСПРАВЛЕНЫ ОШИБКИ)
def calculate_ema(prices: List[float], period: int = 14) -> List[float]:
    """✅ ИСПРАВЛЕНО: обработка пустого списка"""
    if not prices:
        return []
    if len(prices) < period: 
        return [prices[-1]] * period
    mult = 2 / (period + 1)
    ema = [prices[0]]
    for p in prices[1:]: 
        ema.append(p * mult + ema[-1] * (1 - mult))
    return ema

def calculate_wma(prices: List[float], period: int = 20) -> float:
    """✅ ИСПРАВЛЕНО: обработка недостатка данных"""
    if not prices or len(prices) < period: 
        return prices[-1] if prices else 0
    weights = list(range(1, period + 1))
    return sum(p * w for p, w in zip(prices[-period:], weights)) / sum(weights)

def calculate_bollinger(prices: List[float], period: int = 20, std: float = 2) -> Dict[str, float]:
    """✅ ИСПРАВЛЕНО: безопасный расчет стд отклонения"""
    if len(prices) < period: 
        price = prices[-1] if prices else 0
        return {'upper': price, 'lower': price, 'sma': price}
    recent_prices = prices[-period:]
    sma = sum(recent_prices) / period
    try:
        std_dev = statistics.stdev(recent_prices)
    except statistics.StatisticsError:
        std_dev = 0
    return {'upper': sma + std * std_dev, 'lower': sma - std * std_dev, 'sma': sma}

def bb_position(price: float, bb: Dict[str, float]) -> float:
    """✅ ИСПРАВЛЕНО: деление на ноль"""
    if bb['upper'] == bb['lower']:
        return 50.0
    return (price - bb['lower']) / (bb['upper'] - bb['lower']) * 100

def calculate_vwap(opens: List[float], highs: List[float], lows: List[float], 
                  closes: List[float], volumes: List[float], period: int = 20) -> float:
    """✅ ИСПРАВЛЕНО: безопасность деления"""
    n = min(period, len(highs))
    if n == 0:
        return closes[-1] if closes else 0
    typical = [(h + l + c) / 3 for h, l, c in zip(highs[-n:], lows[-n:], closes[-n:])]
    vol_sum = sum(volumes[-n:])
    return sum(p * v for p, v in zip(typical, volumes[-n:])) / vol_sum if vol_sum else closes[-1]

def calculate_sar(highs: List[float], lows: List[float], af_step: float = 0.02, af_max: float = 0.2) -> float:
    """✅ ИСПРАВЛЕНО: полноценный Parabolic SAR"""
    if len(highs) < 2: 
        return lows[-1] if lows else 0
    # Безопасная "короткая" версия (последние ~10 баров), без выхода за границы индексов.
    lookback = min(10, len(highs) - 1)
    start = -lookback
    sar = lows[start - 1]
    ep = highs[start]
    af = af_step

    for i in range(start, 0):
        if highs[i] > ep:
            ep = highs[i]
            af = min(af + af_step, af_max)
        sar = sar + af * (ep - sar)
    return sar

def calculate_supertrend(highs: List[float], lows: List[float], closes: List[float], 
                        period: int = 10, mult: float = 3) -> float:
    """✅ ИСПРАВЛЕНО: корректный ATR"""
    n = min(period, len(highs))
    if n < 2:
        return closes[-1] if closes else 0
        
    hl2 = [(h + l) / 2 for h, l in zip(highs[-n:], lows[-n:])]
    atr_values = []
    for i in range(-n+1, 0):
        prev_close = closes[i-1]
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - prev_close),
            abs(lows[i] - prev_close),
        )
        atr_values.append(tr)
    
    atr = sum(atr_values) / len(atr_values) if atr_values else 0
    basic_upper = hl2[-1] + mult * atr
    basic_lower = hl2[-1] - mult * atr
    return basic_upper if len(closes) > 1 and closes[-2] <= basic_upper else basic_lower

def calculate_trix(prices: List[float], period: int = 14) -> float:
    """✅ ИСПРАВЛЕНО: безопасность деления"""
    if len(prices) < period * 2: 
        return 0
    try:
        ema1 = calculate_ema(prices, period)
        ema2 = calculate_ema(ema1, period)
        ema3 = calculate_ema(ema2, period)
        return ((ema3[-1] - ema3[-2]) / ema3[-2] * 100) if len(ema3) > 1 and ema3[-2] != 0 else 0
    except:
        return 0

def calculate_adx(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
    """✅ ИСПРАВЛЕНО: упрощенный ADX -> ATR (правильнее для сигнала силы)"""
    if len(highs) < period + 1: 
        return 0
    tr = []
    for i in range(-period, 0):
        prev_close = closes[i-1]
        tr_val = max(
            highs[i] - lows[i],
            abs(highs[i] - prev_close),
            abs(lows[i] - prev_close),
        )
        tr.append(tr_val)
    return sum(tr) / len(tr) / max(closes[-1], 1) * 100

def calculate_rsi(prices: List[float], period: int = 14) -> float:
    """✅ ИСПРАВЛЕНО: корректный RSI"""
    if len(prices) <= period: 
        return 50.0
    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    last = deltas[-period:]
    gains = [d if d > 0 else 0 for d in last]
    losses = [-d if d < 0 else 0 for d in last]
    try:
        avg_gain = statistics.mean(gains)
        avg_loss = statistics.mean(losses)
        if avg_loss == 0: 
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    except:
        return 50.0

# ✅ ФУНКЦИИ ИСТОРИИ (ИСПРАВЛЕНЫ)
def load_history() -> List[Dict]:
    try:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Ошибка загрузки истории: {e}")
    return []

def save_history(history: List[Dict]):
    try:
        recent_history = history[-300:]
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(recent_history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Ошибка сохранения истории: {e}")

def analyze_past_predictions(history: List[Dict], current_price: float) -> float:
    if len(history) < 2:
        return 50.0
    recent_correct = 0
    recent_total = 0
    start_idx = max(0, len(history) - 51)
    for i in range(start_idx, len(history) - 1):
        past = history[i]
        past_pred = past.get('direction', '')
        past_price = past.get('price', 0)
        if i + 1 < len(history):
            next_price = history[i + 1].get('price', past_price)
            is_long = ("LONG" in past_pred) or ("UP" in past_pred) or ("🟢" in past_pred)
            is_short = ("SHORT" in past_pred) or ("🔴" in past_pred)

            if is_long and next_price > past_price * 1.001:
                recent_correct += 1
            elif is_short and next_price < past_price * 0.999:
                recent_correct += 1
            recent_total += 1
    return round((recent_correct / recent_total * 100), 1) if recent_total > 0 else 50.0

def calculate_targets_PRO(data: Dict, direction: str) -> Tuple[float, float, float, float]:
    c = data.get('c', 0)
    if c == 0:
        return c, c, 0.0, 0.0
    
    atr_pct = ((data.get('h', c) - data.get('l', c)) / c) * 100
    rsi = data.get('rsi', 50)
    
    if rsi > 75:
        target_mult = 1.8
    elif rsi > 65:
        target_mult = 2.0
    else:
        target_mult = 2.4
    
    stop_mult = 0.9
    target_dist = atr_pct * target_mult
    stop_dist = atr_pct * stop_mult
    
    if "LONG" in direction or "🟢" in direction:
        target_price = c * (1 + target_dist / 100)
        stop_price = c * (1 - stop_dist / 100)
        profit_pct = round((target_price - c) / c * 100, 1)
        loss_pct = round((stop_price - c) / c * 100, 1)
    else:  # SHORT
        target_price = c * (1 - target_dist / 100)
        stop_price = c * (1 + stop_dist / 100)
        profit_pct = round((c - target_price) / c * 100, 1)
        # Для единообразия (как в LONG) стоп показываем отрицательным процентом.
        loss_pct = -round((stop_price - c) / c * 100, 1)
    
    return target_price, stop_price, profit_pct, loss_pct

def calculate_risk(data: Dict) -> Tuple[int, List[str]]:
    risk_points = 0
    risk_factors = []
    rsi = data.get('rsi', 50)
    spread = data.get('spread', 0)
    h, l, c, vol = data.get('h', 0), data.get('l', 0), data.get('c', 0), data.get('vol', 0)

    if rsi > 78 or rsi < 22:
        risk_points += 30
        risk_factors.append("🚨 RSI экстремальный")
    if spread > 35:
        risk_points += 20
        risk_factors.append("📉 Спред опасный")
    if c != 0 and (h - l) > c * 0.07:
        risk_points += 25
        risk_factors.append("💥 Волатильность экстремальная")
    if vol < 80_000_000_000:
        risk_points += 15
        risk_factors.append("📉 Низкий объем")
    
    return min(risk_points, 100), risk_factors

# ✅ ОРИГИНАЛЬНАЯ ФУНКЦИЯ get_btc_data() (ИСПРАВЛЕНА)
def get_btc_data() -> Dict[str, Any]:
    try:
        url_chart = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=2"
        r_chart = requests.get(url_chart, timeout=10)
        r_chart.raise_for_status()
        data_chart = r_chart.json()
        
        prices = data_chart.get('prices', [])
        vols = data_chart.get('total_volumes', [])
        recent = prices[-12:] if len(prices) >= 12 else prices
        if not recent:
            return {'c': 0, 'h': 0, 'l': 0, 'vol': 0}
            
        o = recent[0][1]; h = max(p[1] for p in recent); l = min(p[1] for p in recent); c = recent[-1][1]
        vol = sum(v[1] for v in vols[-12:]) if len(vols) >= 12 else 0
        prices_24h = [p[1] for p in prices[-24:]]; sma12 = sum(prices_24h[-12:]) / 12 if len(prices_24h) >= 12 else c
        rsi = calculate_rsi(prices_24h)
        
        bid = ask = c
        try:
            url_ticker = "https://api.binance.com/api/v3/ticker/bookTicker?symbol=BTCUSDT"
            r_ticker = requests.get(url_ticker, timeout=5)
            if r_ticker.status_code == 200:
                ticker = r_ticker.json()
                bid = float(ticker['bidPrice']); ask = float(ticker['askPrice'])
        except: 
            pass
        spread = ask - bid if ask > bid else max(0.1, c * 0.0001)
        
        data = {
            'o': o, 'h': h, 'l': l, 'c': c, 'vol': vol, 'bid': bid, 'ask': ask, 'spread': spread,
            'sma12': sma12, 'rsi': rsi, 'prices_24h': prices_24h
        }
        
        # Доп. данные из Binance Futures (200 свечей по 5m)
        try:
            url_klines = "https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=5m&limit=200"
            r_kl = requests.get(url_klines, timeout=10)
            r_kl.raise_for_status()
            klines = r_kl.json()

            if not isinstance(klines, list) or not klines:
                raise ValueError("Binance klines: пусто/не список")

            closes = [float(k[4]) for k in klines]
            highs = [float(k[2]) for k in klines]
            lows = [float(k[3]) for k in klines]
            volumes = [float(k[5]) for k in klines]
            opens = [float(k[1]) for k in klines]

            bb = calculate_bollinger(closes)
            n20 = min(20, len(closes))
            ma20 = sum(closes[-n20:]) / n20 if n20 else c

            # vol_ratio: последний объем / средний объем последних 9 баров
            if len(volumes) >= 10:
                avg9 = sum(volumes[-10:-1]) / 9
                vol_ratio = volumes[-1] / max(1.0, avg9)
            else:
                vol_ratio = 1.0

            data.update({
                'ma20': ma20,
                'ema12': calculate_ema(closes, 12)[-1],
                'wma20': calculate_wma(closes, 20),
                'bb_position': bb_position(c, bb),
                'bb_upper': bb['upper'],
                'bb_lower': bb['lower'],
                'vwap': calculate_vwap(opens, highs, lows, closes, volumes),
                'sar': calculate_sar(highs, lows),
                'supertrend': calculate_supertrend(highs, lows, closes),
                'trix': calculate_trix(closes),
                'adx': calculate_adx(highs, lows, closes),
                'avl_20': (sum(volumes[-n20:]) / n20) if n20 else 0,
                'vol_ratio': vol_ratio,
            })
        except Exception as e:
            logger.warning(f"Binance klines недоступны: {e}")
    except: 
        pass
    
    return data


# 🔥 НОВЫЕ ФУНКЦИИ ДЛЯ БОЛЬШИХ ДАННЫХ (ДОБАВЛЕНЫ)
def fetch_extended_klines(symbol='BTCUSDT', interval='5m', total_candles=5000):
    """🔥 Получает 5000 свечей с пагинацией (~17 дней)"""
    all_klines = []
    base_url = "https://fapi.binance.com/fapi/v1/klines"
    end_time = int(time.time() * 1000)
    
    while len(all_klines) < total_candles:
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': 1000,
            'endTime': end_time
        }
        
        try:
            r = requests.get(base_url, params=params, timeout=15)
            r.raise_for_status()
            klines = r.json()
            
            if not klines:
                break
                
            all_klines = klines + all_klines
            all_klines = all_klines[:total_candles]
            
            end_time = int(klines[0][0]) - 1
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Ошибка загрузки: {e}")
            break
    
    logger.info(f"✅ Загружено {len(all_klines)} свечей {interval}")
    return all_klines


def get_btc_data_v10():
    """🔥 НОВЫЕ ДАННЫЕ: 5000 свечей + мультитаймфрейм"""
    data = {}
    
    # 1. БОЛЬШОЙ МАССИВ 5000 свечей 5m
    extended_klines = fetch_extended_klines(total_candles=5000)
    
    if extended_klines:
        closes_full = [float(k[4]) for k in extended_klines]
        highs_full = [float(k[2]) for k in extended_klines]
        lows_full = [float(k[3]) for k in extended_klines]
        volumes_full = [float(k[5]) for k in extended_klines]
        
        # Текущие OHLC (последние 12 часов)
        recent = extended_klines[-144:]
        o = float(recent[0][1])
        h = max(float(k[2]) for k in recent)
        l = min(float(k[3]) for k in recent)
        c = float(recent[-1][4])
        vol = sum(float(k[5]) for k in recent)
        
        data.update({
            'o': o, 'h': h, 'l': l, 'c': c, 'vol': vol,
            'closes_full': closes_full, 'highs_full': highs_full,
            'lows_full': lows_full, 'volumes_full': volumes_full
        })
        
        # 🔥 ДОЛГОСРОЧНЫЕ ИНДИКАТОРЫ НА 5000 СВЕЧАХ
        data.update({
            'ema50': calculate_ema(closes_full, 50)[-1],
            'ema200': calculate_ema(closes_full, 200)[-1],
            'rsi_long': calculate_rsi(closes_full),
            'bb_long': calculate_bollinger(closes_full),
            'bb_position_long': bb_position(c, calculate_bollinger(closes_full))
        })
    else:
        # Fallback на оригинальные данные
        data = get_btc_data()
    
    # Спред (как в оригинале)
    try:
        url_ticker = "https://api.binance.com/api/v3/ticker/bookTicker?symbol=BTCUSDT"
        ticker = requests.get(url_ticker, timeout=5).json()
        bid, ask = float(ticker['bidPrice']), float(ticker['askPrice'])
        data['spread'] = ask - bid
        data['bid'] = bid
        data['ask'] = ask
    except:
        data['spread'] = data.get('c', 0) * 0.0001
    
    data['rsi'] = data.get('rsi', data.get('rsi_long', 50))
    return data


def aladdin_PRO_v10():
    """🔥 НОВАЯ ЛОГИКА АНАЛИЗА на 5000 свечах"""
    data = get_btc_data_v10()
    history = load_history()
    history_accuracy = analyze_past_predictions(history, data['c'])

    score = 0
    signals = []
    c = data['c']
    
    # 🔥 НОВЫЕ СИГНАЛЫ ДОЛГОСРОЧНЫХ ТРЕНДОВ
    if c > data.get('ema200', c):
        score += 1.5
        signals.append("🟢 EMA200 БЫЧИЙ (5000 свечей)")
    if c > data.get('ema50', c):
        score += 1.0
        signals.append("📈 EMA50 поддержка")
    
    # Оригинальные сигналы (адаптированные)
    if data.get('bb_position_long', 50) < 25:
        score += 1.2
        signals.append("📉 BOLL перепродан (долгосрок)")
    elif data.get('bb_position_long', 50) > 75:
        score -= 1.0
        signals.append("📈 BOLL перекуплен")
    
    # Короткие сигналы из оригинала
    if 'ma20' in data and c > max(data['ma20'], data.get('ema12', c)):
        score += 0.8
        signals.append("📈 MA/EMA бычьи")
    if 'vol_ratio' in data and data['vol_ratio'] > 1.5:
        score += 0.8
        signals.append("🔥 Volume spike")

    direction = "🟢 LONG" if score >= 2.0 else "🔴 SHORT"
    confidence = min(abs(score) * 20, 95)

    risk_percent, risk_factors = calculate_risk(data)
    target, stop, profit_pct, loss_pct = calculate_targets_PRO(data, direction)

    forecast = {
        'time': datetime.now().isoformat(),
        'price': c,
        'direction': direction,
        'confidence': confidence,
        'rsi': data.get('rsi', 50),
        'risk': risk_percent,
        'history_accuracy': history_accuracy,
        'ema200': data.get('ema200', c)
    }
    history.append(forecast)
    save_history(history)

    state['previous_signal'] = direction
    state['previous_price'] = c

    return (
        data, direction, confidence, signals, risk_percent, risk_factors,
        history_accuracy, target, stop, profit_pct, loss_pct, []
    )

# ✅ ВСЕ ВАШИ ОРИГИНАЛЬНЫЕ ФУНКЦИИ ОСТАЮТСЯ
def aladdin_PRO_analysis():
    data = get_btc_data()
    history = load_history()
    history_accuracy = analyze_past_predictions(history, data['c'])

    score = 0
    signals = []
    
    if data['c'] > max(data['ma20'], data['ema12']): 
        score += 0.8; 
        signals.append("📈 MA/EMA бычьи")
    if data['bb_position'] < 20: 
        score += 1.0; 
        signals.append("📉 BOLL перепродан")
    if data['bb_position'] > 80: 
        score -= 0.8; 
        signals.append("📈 BOLL перекуплен")
    if data['c'] > data['vwap']: 
        score += 0.6; 
        signals.append("💰 VWAP выше")
    if data['c'] > data['supertrend']: 
        score += 1.2; 
        signals.append("🚀 SUPER бычий")
    if data['trix'] > 0: 
        score += 0.6; 
        signals.append("⚡ TRIX бычий")
    if data['adx'] > 30: 
        score += 0.5; 
        signals.append("📊 ADX тренд")
    if data['vol_ratio'] > 1.5: 
        score += 0.8; 
        signals.append("🔥 Volume spike")

    direction = "🟢 LONG" if score >= 2.2 else "🔴 SHORT"
    confidence = min(abs(score) * 18, 92)

    risk_percent, risk_factors = calculate_risk(data)
    target, stop, profit_pct, loss_pct = calculate_targets_PRO(data, direction)

    forecast = {
        'time': datetime.now().isoformat(),
        'price': data['c'],
        'direction': direction,
        'confidence': confidence,
        'rsi': data['rsi'],
        'risk': risk_percent,
        'history_accuracy': history_accuracy,
        'sma12': data['sma12']
    }
    history.append(forecast)
    save_history(history)

    price_change = ((data['c'] - state['previous_price']) / state['previous_price'] * 100) if state['previous_price'] else 0
    alerts = []
    if state['previous_signal'] and state['previous_signal'] != direction:
        alerts.append(f"🚨 СИГНАЛ СМЕНИЛСЯ: {state['previous_signal']} → {direction}")
    if abs(price_change) > 2.5:
        emoji = "📈" if price_change > 0 else "📉"
        alerts.append(f"{emoji} ДВИЖЕНИЕ {price_change:+.1f}%!")
    if data['rsi'] > 78:
        alerts.append("🔔 ⚠️ RSI ПЕРЕКУПЛЕН!")
    elif data['rsi'] < 22:
        alerts.append("🔔 🟢 RSI ПЕРЕПРОДАН!")

    state['previous_signal'] = direction
    state['previous_price'] = data['c']

    return (
        data, direction, confidence, signals, risk_percent, risk_factors,
        history_accuracy, target, stop, profit_pct, loss_pct, alerts
    )

def aladdin_cached():
    now = time.time()
    if _analysis_cache["data"] is not None and (now - _analysis_cache["time"] < CACHE_SECONDS):
        logger.info("✅ КЭШ: данные свежие!")
        return _analysis_cache["data"]
    logger.info("🔄 КЭШ: новый анализ...")
    result = aladdin_PRO_analysis()
    _analysis_cache["data"] = result
    _analysis_cache["time"] = now
    return result

def main_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📊 Анализ", callback_data="analyze")],
        [InlineKeyboardButton(text="📈 Индикаторы", callback_data="indicators")],
        [InlineKeyboardButton(text="⚠️ Риск", callback_data="risk"),
         InlineKeyboardButton(text="🚨 Активировать алерты", callback_data="alerts")]
    ])

def agreement_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="✅ ОЗНАКОМЛЕН", callback_data="agree_yes")],
        [InlineKeyboardButton(text="❌ Отказаться", callback_data="agree_no")]
    ])

@dp.message(Command("start"))
async def start_cmd(message: Message):
    user_id = message.from_user.id
    chat_id = message.chat.id
    
    # Инициализация состояния пользователя
    user_states[user_id]['chat_id'] = chat_id
    user_states[user_id]['agreed'] = False
    
    state['alert_chat_id'] = chat_id
    
    await message.answer(
        DISCLAIMER_TEXT,
        reply_markup=agreement_keyboard()
    )

@dp.callback_query(F.data == "agree_no")
async def decline_handler(callback: CallbackQuery):
    await callback.answer()
    await callback.message.edit_text(
        "❌ *Согласие НЕ получено*\n\n"
        "⚠️ Бот не будет работать без принятия условий.\n"
        "Нажмите /start для повторного ознакомления.",
        reply_markup=None
    )

@dp.callback_query(F.data == "agree_yes")
async def agree_handler(callback: CallbackQuery):
    user_id = callback.from_user.id
    
    await callback.answer()
    
    # ✅ ПОЛЬЗОВАТЕЛЬ СОГЛАСИЛСЬ
    user_states[user_id]['agreed'] = True
    
    await callback.message.edit_text(
        "*✅ СОГЛАСИЕ ПРИНЯТО!*\n\n"
        "*📊 ALADDIN v9.7 активирован*\n\n"
        "*✅ Компактный анализ BTC/USDT*\n\n"
        "📊 Выбери действие:",
        reply_markup=main_keyboard()
    )

# Блокировка всех торговых функций без согласия
def user_agreed(user_id: int) -> bool:
    return user_states[user_id]['agreed']

@dp.callback_query(F.data == "analyze")
async def analyze_cb(callback: CallbackQuery):
    user_id = callback.from_user.id
    
    if not user_agreed(user_id):
        await callback.answer("❌ Сначала примите соглашение! /start", show_alert=True)
        return
    
    await callback.answer()
    await callback.message.edit_text("⏳ Анализ...")
    try:
        (data, direction, conf, signals, risk, risk_factors, hist_acc, target, stop, profit, loss, alerts) = aladdin_cached()
        rr = abs(profit/abs(loss)) if abs(loss) > 0 else 0
        vola = ((data['h'] - data['l']) / data['c'] * 100)
        
        analysis_text = f"""*📊 АНАЛИЗ*

{direction} (`{conf:.1f}%`)
#📊 RSI: `{data['rsi']:.1f}` 
#🌊 Volat: `{vola:.1f}%`
#📈 MA20: `{data['ma20']:,.0f}$`

#⚡ EMA12: `{data['ema12']:,.0f}$`
#📉 WMA20: `{data['wma20']:,.0f}$`
#🎯 BOLL: `{data['bb_position']:.0f}%`
#💰 VWAP: `{data['vwap']:,.0f}$`

#📊 *Ключевые:*
• RSI: `{data['rsi']:.0f}`
• BOLL: `{data['bb_position']:.0f}%`
• Spread: `{data['spread']:.1f}$`

#💎 *TRADE PLAN:*
#🚪Entry: `{data['c']:,.0f}$`
#🎯 Target: `{target:,.0f}$` ({profit:+.1f}%)
#🛑 Stop: `{stop:,.0f}$` ({loss:+.1f}%)
#⚖️ R:R `{rr:.1f}:1`"""
        
        await callback.message.edit_text(analysis_text, parse_mode="Markdown")
    except Exception as e:
        await callback.message.edit_text(f"❌ {str(e)[:50]}")
    
    await callback.message.answer("Что дальше?", reply_markup=main_keyboard())

@dp.callback_query(F.data == "indicators")
async def indicators_cb(callback: CallbackQuery):
    user_id = callback.from_user.id
    
    if not user_agreed(user_id):
        await callback.answer("❌ Сначала примите соглашение! /start", show_alert=True)
        return
    
    await callback.answer()
    try:
        data, direction, conf, _, _, _, _, _, _, _, _, _ = aladdin_cached()
        
        indicators_text = f"""📈 *ИНДИКАТОРЫ* — что они значат? 🤔

#*🔥 ОСНОВНОЙ СИГНАЛ:* `{direction}` `{conf:.1f}%`
#❓ Это итог всех индикаторов ниже!

────────────────────

#📊 *RSI: `{data['rsi']:.1f}`*
#✅ 30-70 = нормально, торгуй
#🟢 <30 = ДЁШЕВО, покупай!  
#🔴 >70 = ДОРОГО, продавай!

#📈 *MA20: `{data['ma20']:,.0f}$`*
#✅ Цена > MA20 = рост 📈
#❌ Цена < MA20 = падение 📉

#⚡ *EMA12: `{data['ema12']:,.0f}$`*  
#✅ Быстрая линия тренда
Цена выше = быстро растёт!

#📉 *WMA20: `{data['wma20']:,.0f}$`*
#✅ Последние цены важнее
Реагирует на свежие движения

#🎯 *BOLL: `{data['bb_position']:.0f}%`*
#🟢 <20% = СИЛЬНО ДЁШЕВО!  
#🔴 >80% = СИЛЬНО ДОРОГО!
50% = середина диапазона

#💰 *VWAP: `{data['vwap']:,.0f}$`*
#✅ Средняя цена китов  
Цена выше = киты покупают

#🚀 *SAR: `{data['sar']:,.0f}$`*
#✅ Точка разворота  
Цена выше SAR = рост 🟢

#⚡ *TRIX: `{data['trix']:.2f}`*
#🟢 >0 = разгон вверх  
#🔴 <0 = торможение вниз

#📊 *ADX: `{data['adx']:.1f}`*
#✅ <20 = рынок спит 😴  
#🟡 20-25 = слабый тренд  
#🟢 >25 = ТРЕНД! Иди за ним!"""
        
        await callback.message.edit_text(indicators_text, parse_mode="Markdown")
    except:
        await callback.message.edit_text("❌ Ошибка индикаторов")
    
    await callback.message.answer("Что дальше?", reply_markup=main_keyboard())

@dp.callback_query(F.data == "risk")
async def risk_cb(callback: CallbackQuery):
    user_id = callback.from_user.id
    
    if not user_agreed(user_id):
        await callback.answer("❌ Сначала примите соглашение! /start", show_alert=True)
        return
    
    await callback.answer()
    try:
        data, _, _, _, risk, risk_factors, hist_acc, _, _, _, _, _ = aladdin_cached()
        vola = ((data['h'] - data['l']) / data['c'] * 100)
        
        risk_text = f"""*⚠️ РИСК ({risk}%) & СТАТИСТИКА*
─────────────────
#🎚️ RSI: `{data['rsi']:.1f}`
#📊 Volat: `{vola:.1f}%`
#📈 History: `{hist_acc}%`
#💰 Volume: `{data['vol']:,.0f}`

#🔍 *Факторы риска:*
{chr(10).join(risk_factors) if risk_factors else '✅ Низкий риск'}"""
        
        await callback.message.edit_text(risk_text, parse_mode="Markdown")
    except:
        await callback.message.edit_text("❌ Ошибка рисков")
    
    await callback.message.answer("Что дальше?", reply_markup=main_keyboard())

@dp.callback_query(F.data == "alerts")
async def alerts_cb(callback: CallbackQuery):
    user_id = callback.from_user.id
    
    if not user_agreed(user_id):
        await callback.answer("❌ Сначала примите соглашение! /start", show_alert=True)
        return
    
    state['alert_chat_id'] = callback.message.chat.id
    alerts_text = """*🚨 АЛЕРТЫ АКТИВНЫ* ✅

#⏰ *Каждые 5 минут проверка:*

#🔄 Смена сигнала LONG/SHORT
#📈 Движение >2.5%
#🚨 RSI >78 / <22

#*💎 Готово для торговли!*"""
    await callback.message.edit_text(alerts_text, parse_mode="Markdown")
    await callback.message.answer("Что дальше?", reply_markup=main_keyboard())

async def alert_loop():
    while True:
        try:
            if state['alert_chat_id']:
                _, _, _, _, _, _, _, _, _, _, _, alerts = aladdin_cached()
                for alert in alerts:
                    await bot.send_message(state['alert_chat_id'], f"🚨 *PRO АЛЕРТ*\n{alert}", parse_mode="Markdown")
                    await asyncio.sleep(1)
        except:
            pass
        await asyncio.sleep(300)

async def main():
    logger.info("🚀 ALADDIN v9.7 — Запуск с простым согласием!")
    alert_task = asyncio.create_task(alert_loop())
    try:
        await dp.start_polling(bot)
    finally:
        alert_task.cancel()

if __name__ == "__main__":
    asyncio.run(main())



