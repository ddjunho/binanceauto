import pandas as pd
import numpy as np
import ccxt
import time
import datetime
import schedule
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
symbol = 'SOLUSDT'
timeframe = '4h'

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
            send_to_telegram(f'/start - 시작\n/stop - 중지\n/set(k) - k 값을 설정\n 현재값 : {k_value}\n/after_3m - 3분뒤 가격예측\n/after_5m - 5분뒤 가격예측\n/after_15m - 15분뒤 가격예측\n/after_1h - 1시간뒤 가격예측\n/after_4h - 4시간뒤 가격예측\n/after_1d - 1일뒤 가격예측')
        elif msg['text'] == '/reset_signals':
            reset_signals
        elif msg['text'] == '/set(0.2)':
            k_value = 0.2
        elif msg['text'] == '/set(0.25)':
            k_value = 0.25
        elif msg['text'] == '/set(0.3)':
            k_value = 0.3
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
        elif msg['text'] == '/after_4h':
            send_to_telegram('모델학습 및 예측 중...')
            predict_price(prediction_time='4h')
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
    if prediction_time == '4h':
        minutes_to_add = 60*4
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

def reset_signals():
    global waiting_sell_signal, waiting_buy_signal
    waiting_sell_signal = False
    waiting_buy_signal = False

schedule.every(4).hours.do(reset_signals)
signal = False
buy_signal = False
sell_signal = False
waiting_buy_signal = False
waiting_sell_signal = False
# 변동성 돌파 전략을 적용한 매매 로직
def volatility_breakout_strategy(symbol, df, k_value):
    # 변동성 돌파 전략
    global range
    global target_long
    global target_short
    global waiting_buy_signal
    global waiting_sell_signal
    global buy_signal
    global sell_signal
    global predicted_buy_low_price
    global predicted_sell_high_price
    global entry_time
    global long_stop_loss
    global short_stop_loss
    global future_close_price
    global long_quantity
    global short_quantity
    global limit_order

    # 변동성 범위 계산
    range = (df['high'].iloc[-2] - df['low'].iloc[-2]) * k_value
    # 롱(매수) 목표가 및 숏(매도) 목표가 설정
    target_long = df['close'].iloc[-2] + range
    target_short = df['close'].iloc[-2] - range

    # 매수 및 매도 주문 로직
    if buy_signal == False:
        # 어제 종가보다 오늘 시가가 높고, 오늘 고가가 목표 롱 가격을 돌파한 경우
        if df['open'].iloc[-2] < df['close'].iloc[-2]:
            if df['high'].iloc[-1] > target_long:
                if waiting_buy_signal == False:
                    # 3분 뒤 가격 예측 및 텔레그램 전송
                    predict_price(prediction_time='3m')
                    send_to_telegram(f"돌파가격 : {target_long}")
                    predicted_buy_low_price = predicted_low_price
                    send_to_telegram(f"최적매수가격 : {predicted_buy_low_price}")
                    waiting_buy_signal = True
                # 현재 가격이 예측한 최적 매수가격보다 낮으면 매수 주문 실행
                if df['close'].iloc[-1] < predicted_buy_low_price:
                    long_quantity = calculate_quantity(symbol) * (leverage - 0.2)
                    place_limit_order(symbol, 'buy', long_quantity, df['close'].iloc[-1])
                    long_stop_loss = df['low'].iloc[-1]
                    limit_order = send_to_telegram(f"매수 - Price: {df['close'].iloc[-1]}, Quantity: {long_quantity}")
                    send_to_telegram(f"손절가 - {long_stop_loss}")
                    buy_signal = True
                    waiting_buy_signal = False
                    # 1시간 뒤의 가격 예측
                    predict_price(prediction_time='4h')
                    future_close_price = predicted_high_price  # 과매수
                    entry_time = datetime.datetime.now()

    if sell_signal == False:
        # 어제 종가보다 오늘 시가가 낮고, 오늘 저가가 목표 숏 가격을 돌파한 경우
        if df['open'].iloc[-2] > df['close'].iloc[-2]:
            if df['low'].iloc[-1] < target_short:
                if waiting_sell_signal == False:
                    # 3분 뒤 가격 예측 및 텔레그램 전송
                    predict_price(prediction_time='3m')
                    send_to_telegram(f"돌파가격 : {target_short}")
                    predicted_sell_high_price = predicted_high_price
                    send_to_telegram(f"최적매도가격 : {predicted_sell_high_price}")
                    waiting_sell_signal = True
                # 현재 가격이 예측한 최적 매도가격보다 높으면 매도 주문 실행
                if df['close'].iloc[-1] > predicted_sell_high_price:
                    short_quantity = calculate_quantity(symbol) * (leverage - 0.2)
                    limit_order = place_limit_order(symbol, 'sell', short_quantity, df['close'].iloc[-1])
                    short_stop_loss = df['high'].iloc[-1]
                    send_to_telegram(f"매도 - Price: {df['close'].iloc[-1]}, Quantity: {short_quantity}")
                    send_to_telegram(f"손절가 - {short_stop_loss}")
                    sell_signal = True
                    waiting_sell_signal = False
                    # 1시간 뒤의 가격 예측
                    predict_price(prediction_time='1h')
                    future_close_price = predicted_low_price  # 과매도
                    entry_time = datetime.datetime.now()

    # 매수 또는 매도 신호가 발생한 경우
    if buy_signal or sell_signal:
        # 주문 정보 가져오기
        order_info = exchange.fetch_order(limit_order['id'], symbol)
        order_status = order_info['status']
        if order_status == 'open':
            # 지정된 시간이 경과하면 주문을 종료하고 이익을 실현
            if buy_signal and (datetime.datetime.now() >= entry_time + datetime.timedelta(hours=(1 - entry_time.hour % 1 - 1/60))):
                place_limit_order(symbol, 'sell', long_quantity, df['close'].iloc[-1])
                send_to_telegram(f"롱포지션 종료 - Quantity: {long_quantity}")
                buy_signal = False
                waiting_buy_signal = False

            elif sell_signal and (datetime.datetime.now() >= entry_time + datetime.timedelta(hours=(1 - entry_time.hour % 1 - 1/60))):
                place_limit_order(symbol, 'buy', short_quantity, df['close'].iloc[-1])
                send_to_telegram(f"숏포지션 종료 - Quantity: {short_quantity}")
                sell_signal = False
                waiting_sell_signal = False

            # 과매수시 포지션 종료
            if buy_signal == True:
                if future_close_price < df['close'].iloc[-1]:
                    place_limit_order(symbol, 'sell', long_quantity, df['close'].iloc[-1])
                    send_to_telegram(f"롱포지션 종료 - Quantity: {long_quantity}")
                    buy_signal = False
                    waiting_buy_signal = False

            # 과매도시 포지션 종료
            elif sell_signal == True:
                if future_close_price > df['close'].iloc[-1]:
                    place_limit_order(symbol, 'buy', short_quantity, df['close'].iloc[-1])
                    send_to_telegram(f"숏포지션 종료 - Quantity: {short_quantity}")
                    sell_signal = False
                    waiting_sell_signal = False

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
            schedule.run_pending()
            # 대기 시간
            time.sleep(trade_interval)
        elif stop:
            buy_signal = False
            sell_signal = False
            signal = False
            time.sleep(60)
    except Exception as e:
        send_to_telegram(f"An error occurred: {e}")
        count+=1
        if count==10:
            stop = True
            count=0
        pass
