import ccxt
import numpy as np
import telepot
from telepot.loop import MessageLoop

bot = telepot.Bot(token="6296102104:AAFC4ddbh7gSgkGOdysFqEBUkIoWXw0-g5A")
chat_id = "5820794752"

# Binance API 설정
from binance_keys import api_key, api_secret

exchange = ccxt.binance({
    'rateLimit': 1000,
    'enableRateLimit': True,
    'apiKey': api_key,
    'secret': api_secret,
})

# 트레이딩 페어 및 타임프레임 설정
symbol = 'BTC/USDT'
timeframe = '5m'

# 레버리지 설정
leverage = 10 

# 이동평균 계산 함수 정의
def calculate_ema(data, period):
    ema = []
    alpha = 2 / (period + 1)
    for i, price in enumerate(data):
        if i == 0:
            ema.append(price)
        else:
            ema.append(alpha * price + (1 - alpha) * ema[-1])
    return ema

# 볼륨 오실레이터 계산 함수 정의
def calculate_volume_oscillator(data, short_period, long_period):
    short_ema = calculate_ema(data, short_period)
    long_ema = calculate_ema(data, long_period)
    oscillator = np.array(short_ema) - np.array(long_ema)
    return oscillator

# 히킨 아시 캔들 계산 함수 정의
def calculate_heikin_ashi_candles(candles):
    heikin_ashi_candles = []
    for candle in candles:
        timestamp, open_price, high, low, close, volume = candle
        if len(heikin_ashi_candles) == 0:
            ha_close = (open_price + high + low + close) / 4
            ha_open = open_price
        else:
            ha_close = (open_price + high + low + close) / 4
            ha_open = (heikin_ashi_candles[-1]['ha_open'] + heikin_ashi_candles[-1]['ha_close']) / 2
        ha_high = max(high, ha_open, ha_close)
        ha_low = min(low, ha_open, ha_close)

        heikin_ashi_candles.append({
            'timestamp': timestamp,
            'ha_open': ha_open,
            'ha_high': ha_high,
            'ha_low': ha_low,
            'ha_close': ha_close,
        })
    return heikin_ashi_candles

# 매수 및 매도 주문 함수 정의
def place_buy_order(quantity):
    try:
        order = exchange.create_market_buy_order(symbol, quantity)
        return order
    except Exception as e:
        error_message = f"An error occurred while placing buy order: {e}"
        send_to_telegram(error_message)
        return None

def place_sell_order(quantity):
    try:
        order = exchange.create_market_sell_order(symbol, quantity)
        return order
    except Exception as e:
        error_message = f"An error occurred while placing sell order: {e}"
        send_to_telegram(error_message)
        return None

# 매매량 계산 함수 정의
def calculate_quantity(symbol, leverage):
    try:
        balance = exchange.fetch_balance()
        usdt_balance = balance['total']['USDT']
        market_price = exchange.fetch_ticker(symbol)['last']
        precision = exchange.markets[symbol]['precision']['quantity']
        quantity = usdt_balance * leverage / market_price
        quantity = exchange.decimal_to_precision(quantity, roundingMode='DOWN', precision=precision)
        return float(quantity)
    except Exception as e:
        error_message = f"An error occurred while calculating the quantity: {e}"
        send_to_telegram(error_message)
        return None

# 메수 (롱) 진입 조건 함수 정의
def should_enter_long(ohlcv, ema9, ema18, volume_oscillator):
    # 조건 1: 9EMA가 18EMA를 상향돌파해야 함
    if ema9[-1] > ema18[-1] and ema9[-2] <= ema18[-2]:
        # 조건 2: 하이켄 아시 캔들이 EMA선 위로 올라와야 함
        heikin_ashi_candles = calculate_heikin_ashi_candles(ohlcv)
        if heikin_ashi_candles[-1]['ha_close'] > ema9[-1]:
            # 조건 3: 9EMA가 18EMA 위에 위치할 때 음봉이 EMA를 터치하고 양봉으로 반등해야 함
            for i in range(-1, -len(ohlcv) - 1, -1):
                if ema9[i] > ema18[i] and heikin_ashi_candles[i]['ha_close'] < ema9[i]:
                    for j in range(i, -len(ohlcv) - 1, -1):
                        if heikin_ashi_candles[j]['ha_close'] > heikin_ashi_candles[j]['ha_open']:
                            # 조건 4: 볼륨 오실레이터가 0 이상이여야 함
                            if volume_oscillator[j] >= 0:
                                return True
    return False

# 메도 (숏) 진입 조건 함수 정의
def should_enter_short(ohlcv, ema9, ema18, volume_oscillator):
    # 조건 1: 9EMA가 18EMA를 하향돌파해야 함
    if ema9[-1] < ema18[-1] and ema9[-2] >= ema18[-2]:
        # 조건 2: 하이켄 아시 캔들이 EMA선 아래로 내려와야 함
        heikin_ashi_candles = calculate_heikin_ashi_candles(ohlcv)
        if heikin_ashi_candles[-1]['ha_close'] < ema9[-1]:
            # 조건 3: 9EMA가 18EMA 아래에 위치할 때 양봉이 EMA를 터치하고 음봉으로 하락해야 함
            for i in range(-1, -len(ohlcv) - 1, -1):
                if ema9[i] < ema18[i] and heikin_ashi_candles[i]['ha_close'] > ema9[i]:
                    for j in range(i, -len(ohlcv) - 1, -1):
                        if heikin_ashi_candles[j]['ha_close'] < heikin_ashi_candles[j]['ha_open']:
                            # 조건 4: 볼륨 오실레이터가 0 이상이여야 함
                            if volume_oscillator[j] >= 0:
                                return True
    return False

# 포지션 종료 함수 정의
def close_position(symbol, ema18):
    try:
        positions = exchange.fetch_positions()
        for position in positions:
            if position['symbol'] == symbol:
                entry_price = float(position['entryPrice'])
                stop_loss_price = entry_price - (entry_price - ema18[-1])  # 손절가 설정
                take_profit_price = entry_price + 2 * (entry_price - stop_loss_price)  # 목표가 설정
                current_price = float(position['markPrice'])
                quantity = abs(float(position['positionAmt']))
                
                if position['positionSide'] == 'LONG':
                    if current_price <= stop_loss_price or current_price >= take_profit_price:
                        place_sell_order(quantity)
                        message = f"Closed long position of {quantity} {symbol} at market price."
                        send_to_telegram(message)
                elif position['positionSide'] == 'SHORT':
                    if current_price >= stop_loss_price or current_price <= take_profit_price:
                        place_buy_order(quantity)
                        message = f"Closed short position of {quantity} {symbol} at market price."
                        send_to_telegram(message)
    except Exception as e:
        error_message = f"An error occurred while closing the position: {e}"
        send_to_telegram(error_message)

# 텔레그램으로 메시지를 보내는 함수
def send_to_telegram(message):
    try:
        bot.sendMessage(chat_id, message)
    except Exception as e:
        print(f"An error occurred while sending to Telegram: {e}")

# 메인 루프
while True:
    try:
        # OHLCV 데이터 가져오기
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe)
        close_prices = np.array([candle[4] for candle in ohlcv])
        
        # 이동평균 계산
        ema9 = calculate_ema(close_prices, 9)
        ema18 = calculate_ema(close_prices, 18)
        
        # 볼륨 오실레이터 계산
        volume_oscillator = calculate_volume_oscillator([candle[5] for candle in ohlcv], 8, 21)
        
        # 메수 (롱) 진입 조건 확인
        if should_enter_long(ohlcv, ema9, ema18, volume_oscillator):
            quantity = calculate_quantity(symbol, leverage)
            if quantity:
                place_buy_order(quantity)
        
        # 메도 (숏) 진입 조건 확인
        if should_enter_short(ohlcv, ema9, ema18, volume_oscillator):
            quantity = calculate_quantity(symbol, leverage)
            if quantity:
                place_sell_order(quantity)
        
        # 포지션 종료 조건 확인
        close_position(symbol, ema18)
        
    except Exception as e:
        error_message = f"An error occurred: {e}"
        send_to_telegram(error_message)
