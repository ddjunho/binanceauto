import pandas as pd
import numpy as np
import ccxt
import time
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
k_value=5.5
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
            send_to_telegram('/start - 시작\n/stop - 중지\n/set_k - k 값을 설정\n 예시: /set_k 5.5')

        # 추가된 부분
        elif msg['text'].startswith('/set_k'):
            try:
                # 예시: /set_k 5.5
                new_k_value = float(msg['text'].split(' ')[1])
                k_value = new_k_value
                send_to_telegram(f'k 값을 {k_value}로 설정하였습니다.')
            except Exception as e:
                send_to_telegram(f'k 값을 설정하는 도중 오류가 발생했습니다: {e}')

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

def predict_price(df):
    """SARIMA로 다음 5분 후 종가 가격 예측"""
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

# 변동성 돌파 전략을 적용한 매매 로직
def volatility_breakout_strategy(symbol, df, k_value):
    # 변동성 돌파 전략
    df['range'] = df['high'].iloc[-2] - df['low'].iloc[-2]
    df['target'] = df['open'].iloc[-1] + df['range'].shift(1) + k_value
    df['buy_signal'] = np.where(df['high'].iloc[-1] > df['target'], 1, 0)
    df['sell_signal'] = np.where(df['low'].iloc[-1] < df['target'], 1, 0)
    
    # 매수 및 매도 주문
    for index, row in df.iterrows():
        if row['buy_signal'] == 1:
            if predicted_close_price>df['high'].iloc[-1]:
                quantity = calculate_quantity(symbol)
                if quantity:
                    place_limit_order(symbol, 'buy', quantity, row['open'])
                    send_to_telegram(f"Buy Order - Price: {row['open']}, Quantity: {quantity}")

        elif row['sell_signal'] == 1:
            if predicted_close_price<df['low'].iloc[-1]:
                quantity = calculate_quantity(symbol)
                if quantity:
                    place_limit_order(symbol, 'sell', quantity, row['open'])
                    send_to_telegram(f"Sell Order - Price: {row['open']}, Quantity: {quantity}")

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
            
            # 종가 예측 실행
            predict_price(df)
            # 변동성 돌파 전략 실행
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
