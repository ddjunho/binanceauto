import pandas as pd
import numpy as np
import ccxt
import time
import schedule
from statsmodels.tsa.statespace.sarimax import SARIMAX
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
symbol = 'ETHUSDT'
timeframe = '5m'

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
k_value = 5.5
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
            send_to_telegram(f'/start - 시작\n/stop - 중지\n/set(k) - k 값을 설정\n 현재값 : {k_value}')
        elif msg['text'] == '/set(4)':
            k_value = 4
        elif msg['text'] == '/set(4.5)':
            k_value = 4.5
        elif msg['text'] == '/set(5)':
            k_value = 5
        elif msg['text'] == '/set(5.5)':
            k_value = 5.5
        elif msg['text'] == '/set(6)':
            k_value = 6
        elif msg['text'] == '/set(6.5)':
            k_value = 6.5
        elif msg['text'] == '/set(7)':
            k_value = 7
        elif msg['text'] == '/set(7.5)':
            k_value = 7.5
        elif msg['text'] == '/set(8)':
            k_value = 8
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

def predict_price():
    """SARIMA로 다음 5분 후 종가 가격 예측"""
    candles = exchange.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=None,
            limit=50
        )

    df = pd.DataFrame(data=candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    global predicted_close_price
    
    # SARIMA 모델에 사용할 열 선택 및 이름 변경
    df = df.rename(columns={'timestamp': 'ds', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'y', 'volume': 'volume'})
    
    # 데이터프레임에서 시간 열을 인덱스로 설정
    df.set_index('ds', inplace=True)
    
    # SARIMA 모델 초기화 및 학습
    order = (1, 1, 1)  # (p, d, q): ARIMA의 차수
    model = SARIMAX(df['y'], order=order)
    results = model.fit(disp=False)
    
    # 다음 5분 후를 예측할 데이터 포인트 생성
    future = pd.DataFrame(index=[df.index[-1] + pd.Timedelta(minutes=5)])
    future['open'] = df['open'].iloc[-1]
    future['high'] = df['high'].iloc[-1]
    future['low'] = df['low'].iloc[-1]
    future['volume'] = df['volume'].iloc[-1]
    
    # 예측 수행
    forecast = results.get_forecast(steps=1, exog=future)
    
    # 예측된 종가 출력
    close_value = forecast.predicted_mean.iloc[0]
    predicted_close_price = close_value
    
schedule.every(5).minutes.do(predict_price)

# 변동성 돌파 전략을 적용한 매매 로직
def volatility_breakout_strategy(symbol, df, k_value):
    # 변동성 돌파 전략
    df['range'] = (df['high'].iloc[-2] - df['low'].iloc[-2])*k_value
    df['target_long'] = df['open'].iloc[-1] + df['range'].shift(1)
    df['target_short'] = df['open'].iloc[-1] - df['range'].shift(1)
    global buy_signal
    global sell_signal
    global long_stop_loss
    global short_stop_loss
    global future_close_price
    buy_signal = False
    sell_signal = False
    long_stop_loss = df['low'].iloc[-2]
    short_stop_loss = df['high'].iloc[-2]
    # 매수 및 매도 주문
    if df['high'].iloc[-1] > df['target_long']:
        if buy_signal == False:
            if predicted_close_price>df['close'].iloc[-1]:
                quantity = calculate_quantity(symbol)
                place_limit_order(symbol, 'buy', quantity, df['close'].iloc[-1])
                send_to_telegram(f"매수 - Price: {df['close'].iloc[-1]}, Quantity: {quantity}")
                future_close_price = predicted_close_price
                buy_signal = True
                send_to_telegram(f"예측가 - {future_close_price}")
    elif df['low'].iloc[-1] < df['target_short']:
        if sell_signal == False:
            if predicted_close_price<df['close'].iloc[-1]:
                quantity = calculate_quantity(symbol)
                place_limit_order(symbol, 'sell', quantity, df['close'].iloc[-1])
                send_to_telegram(f"매도 - Price: {df['close'].iloc[-1]}, Quantity: {quantity}")
                future_close_price = predicted_close_price
                sell_signal = True
                send_to_telegram(f"예측가 - {future_close_price}")

    if buy_signal == True:
        if future_close_price<df['close'].iloc[-1]:
            quantity = calculate_quantity(symbol)
            place_limit_order(symbol, 'sell', quantity, df['close'].iloc[-1])
            send_to_telegram(f"롱포지션 종료 - Quantity: {quantity}")
    elif sell_signal == True:
        if future_close_price>df['close'].iloc[-1]:
            quantity = calculate_quantity(symbol)
            place_limit_order(symbol, 'buy', quantity, df['close'].iloc[-1])
            send_to_telegram(f"숏포지션 종료 - Quantity: {quantity}")    
    #손절
    if df['close'].iloc[-1]<long_stop_loss:
        quantity = calculate_quantity(symbol)
        place_limit_order(symbol, 'sell', quantity, df['close'].iloc[-1])
        send_to_telegram(f"롱포지션 손절 - Quantity: {quantity}")
    elif df['close'].iloc[-1]>short_stop_loss:
        quantity = calculate_quantity(symbol)
        place_limit_order(symbol, 'buy', quantity, df['close'].iloc[-1])
        send_to_telegram(f"숏포지션 손절 - Quantity: {quantity}")

# 매매 주기 (예: 5분마다 전략 실행)
trade_interval = 1  # 초 단위
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
            schedule.run_pending()
            volatility_breakout_strategy(symbol, df, k_value)
            
            # 대기 시간
            time.sleep(trade_interval)
        elif stop:
            time.sleep(60)
    except Exception as e:
        send_to_telegram(f"An error occurred: {e}")
        count+=1
        if count==10:
            stop = True
        pass
