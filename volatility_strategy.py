import pandas as pd
import numpy as np
import ccxt
import time
import datetime
from pmdarima import auto_arima
import telepot
from telepot.loop import MessageLoop

bot = telepot.Bot(token="6296102104:AAFC4ddbh7gSgkGOdysFqEBUkIoWXw0-g5A")
chat_id = "5820794752"

# Binance API 설정
from binance_keys import api_key, api_secret

# Binance API 설정
exchange = ccxt.binance({
    'rateLimit': 1000,
    'enableRateLimit': True,
    'apiKey': api_key,
    'secret': api_secret,
    'options': {
        'defaultType': 'future'
    }
})

# 트레이딩 페어 및 타임프레임 설정
symbol = 'XRPUSDT'
timeframe = '1h'

# 레버리지 설정
leverage = 5
exchange.fapiPrivate_post_leverage({'symbol': symbol, 'leverage': leverage})

# 텔레그램으로 메시지를 보내는 함수
def send_to_telegram(message):
    try:
        bot.sendMessage(chat_id, message)
    except Exception as e:
        send_to_telegram(f"An error occurred while sending to Telegram: {e}")

stop = False
k_value = 0.35
def handle(msg):
    global stop, k_value
    content_type, chat_type, chat_id = telepot.glance(msg)
    if content_type == 'text':
        if msg['text'] == '/start':
            send_to_telegram('Starting...')
            stop = False
        elif msg['text'] == '/stop':
            send_to_telegram('Stopping...')
            stop = True
        elif msg['text'] == '/help':
            send_to_telegram(f'/start - 시작\n/stop - 중지\n/set(k) - k 값을 설정\n 현재값 : {k_value}\n/after_1m - 1분뒤 가격예측\n/after_3m - 3분뒤 가격예측\n/after_5m - 5분뒤 가격예측\n/after_15m - 15분뒤 가격예측\n/after_1h - 1시간뒤 가격예측\n/after_1d - 1일뒤 가격예측')
        elif msg['text'] == '/set(0.35)':
            k_value = 0.35
        elif msg['text'] == '/set(0.4)':
            k_value = 0.4
        elif msg['text'] == '/set(0.45)':
            k_value = 0.45
        elif msg['text'] == '/set(0.5)':
            k_value = 0.5
        elif msg['text'] == '/set(0.55)':
            k_value = 0.55
        elif msg['text'] == '/set(0.6)':
            k_value = 0.6
        elif msg['text'] == '/set(0.65)':
            k_value = 0.65
        elif msg['text'] == '/set(0.7)':
            k_value = 0.7
        elif msg['text'] == '/set(0.75)':
            k_value = 0.75
        elif msg['text'] == '/set(0.8)':
            k_value = 0.8
        elif msg['text'] == '/after_1m':
            send_to_telegram('모델학습 및 예측 중...')
            predict_price(prediction_time='1m')
            send_to_telegram(f'predicted_high_price -> {predicted_high_price}')
            send_to_telegram(f'predicted_low_price -> {predicted_low_price}')
            send_to_telegram(f'predicted_close_price -> {predicted_close_price}')
        elif msg['text'] == '/after_3m':
            send_to_telegram('모델학습 및 예측 중...')
            predict_price(prediction_time='3m')
            send_to_telegram(f'predicted_high_price -> {predicted_high_price}')
            send_to_telegram(f'predicted_low_price -> {predicted_low_price}')
            send_to_telegram(f'predicted_close_price -> {predicted_close_price}')
        elif msg['text'] == '/after_5m':
            send_to_telegram('모델학습 및 예측 중...')
            predict_price(prediction_time='5m')
            send_to_telegram(f'predicted_high_price -> {predicted_high_price}')
            send_to_telegram(f'predicted_low_price -> {predicted_low_price}')
            send_to_telegram(f'predicted_close_price -> {predicted_close_price}')
        elif msg['text'] == '/after_15m':
            send_to_telegram('모델학습 및 예측 중...')
            predict_price(prediction_time='15m')
            send_to_telegram(f'predicted_high_price -> {predicted_high_price}')
            send_to_telegram(f'predicted_low_price -> {predicted_low_price}')
            send_to_telegram(f'predicted_close_price -> {predicted_close_price}')
        elif msg['text'] == '/after_1h':
            send_to_telegram('모델학습 및 예측 중...')
            predict_price(prediction_time='1h')
            send_to_telegram(f'predicted_high_price -> {predicted_high_price}')
            send_to_telegram(f'predicted_low_price -> {predicted_low_price}')
            send_to_telegram(f'predicted_close_price -> {predicted_close_price}')
        elif msg['text'] == '/after_1d':
            send_to_telegram('모델학습 및 예측 중...')
            predict_price(prediction_time='1d')
            send_to_telegram(f'predicted_high_price -> {predicted_high_price}')
            send_to_telegram(f'predicted_low_price -> {predicted_low_price}')
            send_to_telegram(f'predicted_close_price -> {predicted_close_price}')
            
# 텔레그램 메시지 루프
MessageLoop(bot, handle).run_as_thread()

# 매수 및 매도 주문 함수 정의
def place_limit_order(symbol, side, amount, price):
    order = exchange.create_order(
        symbol=symbol,
        type="LIMIT",
        side=side,
        amount=amount,
        price=price
    )
    return order

# 전체 잔액 정보 조회
def get_balance():
    balance = exchange.fetch_balance(params={"type": "future"})
    return balance

# 매매량 계산 함수 정의
def calculate_quantity(symbol):
    try:
        balance = get_balance()
        total_balance = float(balance['total']['USDT'])
        
        # 현재 BTCUSDT 가격 조회
        ticker = exchange.fetch_ticker(symbol)
        btc_price = float(ticker['last'])
        
        # USDT 잔고를 BTC로 환산
        quantity = total_balance / btc_price 
        
        # 소수점 이하 자리 제거
        quantity = round(quantity, 3)
        
        return quantity
    except Exception as e:
        error_message = f"An error occurred while calculating the quantity: {e}"
        send_to_telegram(error_message)
        return None

predicted_close_price = 0


def predict_price(prediction_time='1h'):
    """Auto ARIMA로 다음 종가, 고가, 저가 가격 예측"""
    candles = exchange.fetch_ohlcv(
            symbol=symbol,
            timeframe=prediction_time,
            since=None,
            limit=200
        )

    df = pd.DataFrame(data=candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    global predicted_close_price, predicted_high_price, predicted_low_price
    
    # ARIMA 모델에 사용할 열 선택 및 이름 변경
    df = df.rename(columns={'timestamp': 'ds', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'y', 'volume': 'volume'})
    
    # 데이터프레임에서 시간 열을 인덱스로 설정
    df.set_index('ds', inplace=True)
    
    # Auto ARIMA 모델 초기화 및 학습
    model = auto_arima(df['y'], seasonal=False, suppress_warnings=True)
    
    # 다음 n분 후를 예측할 데이터 포인트 생성
    if prediction_time == '1m':
        minutes_to_add = 1
    elif prediction_time == '3m':
        minutes_to_add = 3
    elif prediction_time == '5m':
        minutes_to_add = 5    
    elif prediction_time == '15m':
        minutes_to_add = 15
    elif prediction_time == '1h':
        minutes_to_add = 60
    elif prediction_time == '1d':
        minutes_to_add = 24 * 60
        
    future = pd.DataFrame(index=[df.index[-1] + pd.Timedelta(minutes=minutes_to_add)])
    future['open'] = df['open'].iloc[-1]
    future['high'] = df['high'].iloc[-1]
    future['low'] = df['low'].iloc[-1]
    future['volume'] = df['volume'].iloc[-1]
    
    # 예측 수행
    forecast, conf_int = model.predict(n_periods=1, exogenous=[future.values], return_conf_int=True)
    
    # 예측된 종가, 고가, 저가 출력
    close_value = forecast[0]
    predicted_close_price = close_value
    
    # 다음과 같이 최대값과 최소값을 구할 수 있습니다.
    predicted_high_price = conf_int[0][1]
    predicted_low_price = conf_int[0][0]


    
buy_signal = False
sell_signal = False
# 변동성 돌파 전략을 적용한 매매 로직
def volatility_breakout_strategy(symbol, df, k_value):
    # 변동성 돌파 전략
    range = (df['high'].iloc[-2] - df['low'].iloc[-2])*k_value
    target_long = df['close'].iloc[-2] + range
    target_short = df['close'].iloc[-2] - range
    global buy_signal
    global sell_signal
    global entry_time
    global long_stop_loss
    global short_stop_loss
    global future_close_price
    global long_quantity
    global short_quantity
    # 매수 및 매도 주문
    if buy_signal == False:
        if df['open'].iloc[-1] < df['close'].iloc[-1]:
            if df['high'].iloc[-2] > target_long:
                predict_price(prediction_time='3m')
                send_to_telegram(f"최적매수가격 : {predicted_low_price}")
                if df['close'].iloc[-1]<predicted_low_price:
                    long_quantity = calculate_quantity(symbol)*(leverage-0.2)
                    place_limit_order(symbol, 'buy', long_quantity, df['close'].iloc[-1])
                    long_stop_loss = df['low'].iloc[-1]
                    send_to_telegram(f"매수 - Price: {df['close'].iloc[-1]}, Quantity: {long_quantity}")
                    buy_signal = True
                    predict_price(prediction_time='1h')
                    future_close_price = predicted_high_price
                    entry_time = datetime.datetime.now()
    elif sell_signal == False:
        if df['open'].iloc[-1] > df['close'].iloc[-1]:
            if df['close'].iloc[-1] < target_short:
                predict_price(prediction_time='3m')
                send_to_telegram(f"최적매도가격 : {predicted_high_price}")
                if df['close'].iloc[-1] > predicted_high_price:
                    short_quantity = calculate_quantity(symbol)*(leverage-0.2)
                    place_limit_order(symbol, 'sell', short_quantity, df['close'].iloc[-1])
                    short_stop_loss = df['high'].iloc[-1]
                    send_to_telegram(f"매도 - Price: {df['close'].iloc[-1]}, Quantity: {short_quantity}")
                    sell_signal = True
                    predict_price(prediction_time='1h')
                    entry_time = datetime.datetime.now()
                    future_close_price = predicted_low_price

    if buy_signal and (datetime.datetime.now() >=  entry_time + datetime.timedelta(hours=(1 - entry_time.hour % 1 - 1/60))):
        place_limit_order(symbol, 'sell', long_quantity, df['close'].iloc[-1])
        send_to_telegram(f"롱포지션 종료 - Quantity: {long_quantity}")
        buy_signal = False

    elif sell_signal and (datetime.datetime.now() >= entry_time + datetime.timedelta(hours=(1 - entry_time.hour % 1 - 1/60))):
        place_limit_order(symbol, 'buy', short_quantity, df['close'].iloc[-1])
        send_to_telegram(f"숏포지션 종료 - Quantity: {short_quantity}")
        sell_signal = False

    if buy_signal == True:
        if future_close_price<df['close'].iloc[-1]:
            place_limit_order(symbol, 'sell', long_quantity, df['close'].iloc[-1])
            send_to_telegram(f"롱포지션 종료 - Quantity: {long_quantity}")
            buy_signal = False

    elif sell_signal == True:
        if future_close_price>df['close'].iloc[-1]:
            place_limit_order(symbol, 'buy', short_quantity, df['close'].iloc[-1])
            send_to_telegram(f"숏포지션 종료 - Quantity: {short_quantity}")    
            sell_signal = False

# 매매 주기 (예: 5분마다 전략 실행)
trade_interval = 2  # 초 단위
count=0
while True:
    try:
        if not stop:
            candles = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=None,
                limit=50
            )
        
            df = pd.DataFrame(data=candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 변동성 돌파 전략 실행
            volatility_breakout_strategy(symbol, df, k_value)
            
            # 대기 시간
            time.sleep(trade_interval)
        elif stop:
            buy_signal = False
            sell_signal = False
            time.sleep(60)
    except Exception as e:
        send_to_telegram(f"An error occurred: {e}")
        count+=1
        if count==10:
            stop = True
            count=0
        pass
