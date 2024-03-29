import ccxt
import numpy as np
import pandas as pd
import telepot
from telepot.loop import MessageLoop

bot = telepot.Bot(token="6296102104:AAFC4ddbh7gSgkGOdysFqEBUkIoWXw0-g5A")
chat_id = "5820794752"

# Binance API 설정
from binance_keys import api_key, api_secret

exchange = ccxt.binance(config={
    'rateLimit': 1000,
    'enableRateLimit': True,
    'apiKey': api_key,
    'secret': api_secret,
    'options': {
        'defaultType': 'future'
    }
})

# 트레이딩 페어 및 타임프레임 설정
symbol = 'BTCUSDT'
timeframe = '5m'

# 레버리지 설정
leverage = 10
exchange.fapiPrivate_post_leverage({'symbol': symbol, 'leverage': leverage})


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
def calculate_heikin_ashi_candles(df):
    heikin_ashi_candles = []
    for index, row in df.iterrows():
        timestamp, open_price, high, low, close, volume = row
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
        heikin_ashi_df = pd.DataFrame(heikin_ashi_candles)
    return heikin_ashi_df

# 매수 및 매도 주문 함수 정의
def place_buy_order(quantity):
    try:
        order = exchange.create_market_buy_order(symbol, quantity)
        message = f"Placed a BUY order for {quantity} {symbol} at market price."
        send_to_telegram(message)
        return order
    except Exception as e:
        error_message = f"An error occurred while placing buy order: {e}"
        send_to_telegram(error_message)
        return None

def place_sell_order(quantity):
    try:
        order = exchange.create_market_sell_order(symbol, quantity)
        message = f"Placed a SELL order for {quantity} {symbol} at market price."
        send_to_telegram(message)
        return order
    except Exception as e:
        error_message = f"An error occurred while placing sell order: {e}"
        send_to_telegram(error_message)
        return None


# 매매량 계산 함수 정의
def calculate_quantity(symbol):
    try:
        balance = exchange.fetch_balance(params={"type": "future"})
        total_balance = float(balance['total']['USDT'])
        
        # 현재 BTCUSDT 가격 조회
        ticker = exchange.fetch_ticker(symbol)
        btc_price = float(ticker['last'])
        # USDT 잔고를 BTC로 환산
        quantity = total_balance / btc_price
        
        # 소수점 이하 자리 제거
        quantity = round(quantity, 4)
        
        return quantity
    except Exception as e:
        error_message = f"An error occurred while calculating the quantity: {e}"
        send_to_telegram(error_message)
        return None
        
#메수 (롱) 진입 조건 함수 정의
def should_enter_long(ohlcv, ema9, ema18, volume_oscillator):
    # 초기 조건 값 설정
    ema9_crossed_above_ema18 = False
    heikin_ashi_candles_above_ema9 = False
    heikin_ashi_candles = calculate_heikin_ashi_candles(ohlcv)
    
    # 조건 1: 9EMA가 18EMA를 상향돌파한 시점 이후부터 검사
    for i in range(-4, -len(ohlcv) - 1, -1):  # 조건 1은 -3부터 시작
        if ema9[i] > ema18[i] and ema9[i - 1] <= ema18[i - 1]:
            ema9_crossed_above_ema18 = True

        if ema9_crossed_above_ema18:
            # 조건 2: 하이켄 아시 캔들이 EMA선 위로 올라와야 함 (양봉인 경우에만)
            if heikin_ashi_candles[i]['ha_close'] > ema9[i] and heikin_ashi_candles[i]['ha_close'] > heikin_ashi_candles[i]['ha_open']:
                heikin_ashi_candles_above_ema9 = True
                break  # 조건 충족 후 종료

    # 조건 3: 9EMA가 18EMA 위에 위치할 때 2개 이하의 음봉이 EMA를 터치하고 양봉으로 반등하거나 도지 모양이어야 함
    if ema9_crossed_above_ema18 and heikin_ashi_candles_above_ema9:
        num_consecutive_bearish_candles = 0  # 연속적인 음봉의 수를 세기 위한 변수
        for i in range(-2, -len(ohlcv) - 1, -1):  # 조건 3은 -2부터 시작
            if ema9[i] > ema18[i] and heikin_ashi_candles[i]['ha_close'] < ema9[i] and abs(heikin_ashi_candles[j]['ha_open'] - heikin_ashi_candles[j]['ha_close']) > (heikin_ashi_candles[j]['ha_high'] - heikin_ashi_candles[j]['ha_low']) * 0.3:
                num_consecutive_bearish_candles += 1
                if num_consecutive_bearish_candles <= 2:
                    for j in range(i, -len(ohlcv) - 1, -1):
                        if heikin_ashi_candles[j]['ha_close'] > heikin_ashi_candles[j]['ha_open']:
                            # 조건 4: 볼륨 오실레이터가 -5 이상이여야 함
                            if volume_oscillator[j] >= -5:
                                return True
                        elif abs(heikin_ashi_candles[j]['ha_open'] - heikin_ashi_candles[j]['ha_close']) <= (heikin_ashi_candles[j]['ha_high'] - heikin_ashi_candles[j]['ha_low']) * 0.3:
                            # 도지 모양인 경우
                            if volume_oscillator[j] >= -5:
                                return True
                else:
                    num_consecutive_bearish_candles = 0  # 양봉이 나오면 연속적인 음봉 수 초기화
    return False

# 메도 (숏) 진입 조건 함수 정의
def should_enter_short(ohlcv, ema9, ema18, volume_oscillator):
    # 초기 조건 값 설정
    ema9_crossed_below_ema18 = False
    heikin_ashi_candles_below_ema9 = False
    heikin_ashi_candles = calculate_heikin_ashi_candles(ohlcv)
    
    # 조건 1: 9EMA가 18EMA를 하향돌파한 시점 이후부터 검사
    for i in range(-4, -len(ohlcv) - 1, -1):  # 조건 1은 -3부터 시작
        if ema9[i] < ema18[i] and ema9[i - 1] >= ema18[i - 1]:
            ema9_crossed_below_ema18 = True

        if ema9_crossed_below_ema18:
            # 조건 2: 하이킨 아시 캔들이 EMA선 아래로 내려와야 함 (음봉인 경우에만)
            if heikin_ashi_candles[i]['ha_close'] < ema9[i] and heikin_ashi_candles[i]['ha_close'] < heikin_ashi_candles[i]['ha_open']:
                heikin_ashi_candles_below_ema9 = True
                break  # 조건 충족 후 종료

    # 조건 3: 9EMA가 18EMA 아래에 위치할 때 2개 이하의 음봉이 EMA를 터치하고 양봉으로 반등해야 함
    if ema9_crossed_below_ema18 and heikin_ashi_candles_below_ema9:
        num_consecutive_bearish_candles = 0  # 연속적인 음봉의 수를 세기 위한 변수
        for i in range(-2, -len(ohlcv) - 1, -1):  # 조건 3은 -2부터 시작
            if ema9[i] < ema18[i] and heikin_ashi_candles[i]['ha_close'] > ema9[i] and abs(heikin_ashi_candles[j]['ha_open'] - heikin_ashi_candles[j]['ha_close']) > (heikin_ashi_candles[j]['ha_high'] - heikin_ashi_candles[j]['ha_low']) * 0.3:
                num_consecutive_bearish_candles += 1
                if num_consecutive_bearish_candles <= 2:
                    for j in range(i, -len(ohlcv) - 1, -1):
                        if heikin_ashi_candles[j]['ha_close'] < heikin_ashi_candles[j]['ha_open']:
                            # 조건 4: 볼륨 오실레이터가 -5 이상이여야 함
                            if volume_oscillator[j] >= -5:
                                return True
                        elif abs(heikin_ashi_candles[j]['ha_open'] - heikin_ashi_candles[j]['ha_close']) <= (heikin_ashi_candles[j]['ha_high'] - heikin_ashi_candles[j]['ha_low']) * 0.3:
                            # 도지 모양인 경우
                            if volume_oscillator[j] >= -5:
                                return True    
                else:
                    num_consecutive_bearish_candles = 0  # 양봉이 나오면 연속적인 음봉 수 초기화
    return False


# 포지션 종료 함수 정의 (업데이트)
def close_position(symbol, ema18):
    global position_entered
    try:
        positions = exchange.fapiPrivateV2_get_positionrisk()
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
                        position_entered = False
                elif position['positionSide'] == 'SHORT':
                    if current_price >= stop_loss_price or current_price <= take_profit_price:
                        place_buy_order(quantity)
                        message = f"Closed short position of {quantity} {symbol} at market price."
                        send_to_telegram(message)
                        position_entered = False
    except Exception as e:
        error_message = f"An error occurred while closing the position: {e}"
        send_to_telegram(error_message)


# 텔레그램으로 메시지를 보내는 함수
def send_to_telegram(message):
    try:
        bot.sendMessage(chat_id, message)
    except Exception as e:
        print(f"An error occurred while sending to Telegram: {e}")

stop = False
def handle(msg):
    global stop
    global leverage
    content_type, chat_type, chat_id = telepot.glance(msg)
    if content_type == 'text':
        if msg['text'] == '/start':
            send_to_telegram('Starting...')
            stop = False
        elif msg['text'] == '/stop':
            send_to_telegram('Stopping...')
            stop = True
        elif msg['text'] == '/set_Leverage':
            send_to_telegram(f'현재 레버리지: {leverage}\n레버리지 설정\n/Leverage_1\n/Leverage_5\n/Leverage_10\n/Leverage_20\n/Leverage_40\n/Leverage_60')
        elif msg['text'] == '/Leverage_1':
            send_to_telegram('Leverage setting complete!')
            leverage = 1
        elif msg['text'] == '/Leverage_5':
            send_to_telegram('Leverage setting complete!')
            leverage = 5
        elif msg['text'] == '/Leverage_10':
            send_to_telegram('Leverage setting complete!')
            leverage = 10
        elif msg['text'] == '/Leverage_20':
            send_to_telegram('Leverage setting complete!')
            leverage = 20
        elif msg['text'] == '/Leverage_40':
            send_to_telegram('Leverage setting complete!')
            leverage = 40
        elif msg['text'] == '/Leverage_60':
            send_to_telegram('Leverage setting complete!')
            leverage = 60
        elif msg['text'] == '/help':
            send_to_telegram('/start - 시작\n/stop - 중지\n/set_Leverage - 레버리지 설정\n/calculate_quantity - 잔고')
        elif msg['text'] == '/calculate_quantity':
            send_to_telegram(f'현재 잔고: {calculate_quantity(symbol)}')
            
# 텔레그램 메시지 루프
MessageLoop(bot, handle).run_as_thread()


print("autotradestart")
position_entered = False  # 포지션 진입 상태를 추적하는 변수
# 메인 루프
while True:
    try:
        if not stop:
            # OHLCV 데이터 가져오기
            candles = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=None,
                limit=50
            )

            df = pd.DataFrame(data=candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            close_prices = df['close'].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 이동평균 계산
            ema9 = calculate_ema(close_prices, 9)
            ema18 = calculate_ema(close_prices, 18)

            # 볼륨 오실레이터 계산
            volume_oscillator = calculate_volume_oscillator(df['volume'].astype(float), 5, 10)
            
            # 메수 (롱) 진입 조건
            long_entry_condition = should_enter_long(df, ema9, ema18, volume_oscillator)
            
            # 메도 (숏) 진입 조건
            short_entry_condition = should_enter_short(df, ema9, ema18, volume_oscillator)
            if not position_entered:
                # 메수 (롱) 진입 조건 확인
                if long_entry_condition:
                    quantity = calculate_quantity(symbol)
                    if quantity:
                        place_buy_order(quantity)
                        position_entered = True  # 포지션 진입 상태 업데이트
                # 메도 (숏) 진입 조건 확인
                elif short_entry_condition:
                    quantity = calculate_quantity(symbol)
                    if quantity:
                        place_sell_order(quantity)
                        position_entered = True  # 포지션 진입 상태 업데이트
            # 포지션 종료 조건 확인
            if position_entered:
                close_position(symbol, ema18)

    except Exception as e:
        error_message = f"An error occurred: {e}"
        send_to_telegram(error_message)
