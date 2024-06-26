import pandas as pd
import numpy as np
import ccxt
import time
import datetime
import schedule
from pmdarima import auto_arima
import telepot
from telepot.loop import MessageLoop

bot = telepot.Bot(token="")
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
timeframe = '6h'

# 레버리지 설정
leverage = 5
exchange.fapiPrivate_post_leverage({'symbol': symbol, 'leverage': leverage*2})

# 텔레그램으로 메시지를 보내는 함수
def send_to_telegram(message):
    try:
        bot.sendMessage(chat_id, message)
    except Exception as e:
        send_to_telegram(f"An error occurred while sending to Telegram: {e}")

stop = False
k_value = 0.55
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
            send_to_telegram(f'/start - 시작\n/stop - 중지\n/set(k) - k 값을 설정\n 현재값 : {k_value}\n/after_3m - 3분뒤 가격예측\n/after_5m - 5분뒤 가격예측\n/after_15m - 15분뒤 가격예측\n/after_1h - 1시간뒤 가격예측\n/after_6h - 6시간뒤 가격예측\n/after_1d - 1일뒤 가격예측')
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
        elif msg['text'] == '/after_6h':
            send_to_telegram('모델학습 및 예측 중...')
            predict_price(prediction_time='6h')
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


def predict_price(prediction_time='1h',add_mintes=0):
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
    if prediction_time == '6h':
        minutes_to_add = 60*6 + add_mintes
    elif prediction_time == '3m':
        minutes_to_add = 3 + add_mintes
    elif prediction_time == '5m':
        minutes_to_add = 5 + add_mintes  
    elif prediction_time == '10m':
        add_mintes = 5
        minutes_to_add = 5 + add_mintes  
    elif prediction_time == '15m':
        minutes_to_add = 15 + add_mintes
    elif prediction_time == '1h':
        minutes_to_add = 60 + add_mintes
    elif prediction_time == '1d':
        minutes_to_add = 24 * 60 + add_mintes
        
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

# Bollinger Bands 계산 함수 정의
def calculate_bollinger_bands(data, window, num_std_dev):
    rolling_mean = data['close'].rolling(window=window).mean()
    rolling_std = data['close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band, lower_band

def calculate_ema(data, period):
    ema = data['close'].ewm(span=period, adjust=False).mean()
    return ema


def calculate_rsi(data, period=14):
    delta = data['close'].diff()
    avg_gain = delta.where(delta > 0, 0).rolling(window=period, center = False).mean()
    avg_loss = -delta.where(delta < 0, 0).rolling(window=period, center = False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    data['rsi'] = rsi
    return data

def stochastic_rsi(data, period=14, smooth_k=3, smooth_d=3):
    """
    스토케스틱 RSI를 계산하는 함수.

    매개변수:
    - data: 'high', 'low', 'close' 열을 포함한 DataFrame.
    - period: RSI 기간 (기본값은 14).
    - smooth_k: %K를 부드럽게 만들기 위한 기간 (기본값은 3).
    - smooth_d: %D를 부드럽게 만들기 위한 기간 (기본값은 3).
    반환값:
    - 'stoch_rsi_k' 및 'stoch_rsi_d'.
    """
    # RSI 계산
    data = calculate_rsi(data, period)

    # 스토케스틱 RSI (%K) 계산
    min_rsi = data['rsi'].rolling(window=period, center = False).min()
    max_rsi = data['rsi'].rolling(window=period, center = False).max()
    stoch = 100 * (data['rsi'] - min_rsi) / (max_rsi - min_rsi)
    stoch_rsi_k = stoch.rolling(window=smooth_k, center = False).mean()
    # 스토케스틱 RSI (%D) 계산
    stoch_rsi_d = stoch_rsi_k.rolling(window=smooth_d, center = False).mean()

    return stoch_rsi_k, stoch_rsi_d


def is_doji_candle(df, threshold=0.2):
    price_range = df['high'].iloc[-2] - df['low'].iloc[-2]
    body_range = abs(df['open'].iloc[-2] - df['close'].iloc[-2])

    # 도지 캔들 판별 조건
    if body_range < threshold * price_range:
        return True
    else: 
        return False


def reset_signals():
    global waiting_sell_signal, waiting_buy_signal
    waiting_sell_signal = False
    waiting_buy_signal = False

schedule.every().day.at("00:01").do(reset_signals)
schedule.every().day.at("06:01").do(reset_signals)
schedule.every().day.at("12:01").do(reset_signals)
schedule.every().day.at("18:01").do(reset_signals)

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
    global buy_price
    global sell_price
    global profit
    
    # 변동성 범위 계산
    range = (df['high'].iloc[-2] - df['low'].iloc[-2]) * k_value
    # 롱(매수) 목표가 및 숏(매도) 목표가 설정
    target_long = df['close'].iloc[-2] + range
    target_short = df['close'].iloc[-2] - range
    target_long2 = (df['close'].iloc[-2] + range*2)
    target_short2 = (df['close'].iloc[-2] - range*2)
    stoch_rsi_k, stoch_rsi_d = stochastic_rsi(df, period=14, smooth_k=3, smooth_d=3)
    is_doji=is_doji_candle(df)
    if (df['high'].iloc[-1] > target_long2) or (df['low'].iloc[-1] < target_short2):
        is_doji = False
    # 매수 및 매도 주문 로직
    if buy_signal == False and is_doji == False:
        if stoch_rsi_k.iloc[-1]<100 and stoch_rsi_d.iloc[-1] < 95:
            # 어제 종가보다 오늘 시가가 높고, 오늘 고가가 목표 롱 가격을 돌파한 경우 혹은 이중 돌파시
            if (df['open'].iloc[-2] < df['close'].iloc[-2]) or (df['high'].iloc[-1] > target_long2):
                if df['high'].iloc[-1] > target_long:
                    if waiting_buy_signal == False:
                        # 5분 뒤 가격 예측 및 텔레그램 전송
                        predict_price(prediction_time='5m')
                        send_to_telegram(f"현재가 : {df['close'].iloc[-1]}")
                        send_to_telegram(f"돌파가격 : {target_long}")
                        predicted_buy_low_price = predicted_low_price
                        send_to_telegram(f"최적매수가격 : {predicted_buy_low_price}")
                        waiting_buy_signal = True
                    # 현재 가격이 예측한 최적 매수가격보다 낮으면 매수 주문 실행
                    if df['close'].iloc[-1] < predicted_buy_low_price:
                        long_quantity = calculate_quantity(symbol) * (leverage - 0.2)
                        limit_order = place_limit_order(symbol, 'buy', long_quantity, df['close'].iloc[-1])
                        long_stop_loss = (df['low'].iloc[-1] + df['open'].iloc[-2])/2 
                        buy_price = df['close'].iloc[-1]
                        send_to_telegram(f"매수 - Price: {buy_price}, Quantity: {long_quantity}")
                        send_to_telegram(f"손절가 - {long_stop_loss}")
                        buy_signal = True
                        upper_band, lower_band = calculate_bollinger_bands(df, window=20, num_std_dev=2.5)
                        future_close_price = upper_band.iloc[-1] # 과매수
                        now = datetime.datetime.now()
                        entry_time = datetime.datetime(now.year, now.month, now.day, now.hour, 0)

    if sell_signal == False and is_doji == False:
        if stoch_rsi_k.iloc[-1] > 0 and stoch_rsi_d.iloc[-1] > 5:
            # 어제 종가보다 오늘 시가가 낮고, 오늘 저가가 목표 숏 가격을 돌파한 경우 혹은 이중 돌파시
            if (df['open'].iloc[-2] > df['close'].iloc[-2]) or (df['low'].iloc[-1] < target_short2):
                if df['low'].iloc[-1] < target_short:
                    if waiting_sell_signal == False:
                        # 5분 뒤 가격 예측 및 텔레그램 전송
                        predict_price(prediction_time='5m')
                        send_to_telegram(f"현재가 : {df['close'].iloc[-1]}")
                        send_to_telegram(f"돌파가격 : {target_short}")
                        predicted_sell_high_price = predicted_high_price
                        send_to_telegram(f"최적매도가격 : {predicted_sell_high_price}")
                        waiting_sell_signal = True
                    # 현재 가격이 예측한 최적 매도가격보다 높으면 매도 주문 실행
                    if df['close'].iloc[-1] > predicted_sell_high_price:
                        short_quantity = calculate_quantity(symbol) * (leverage - 0.2)
                        limit_order = place_limit_order(symbol, 'sell', short_quantity, df['close'].iloc[-1])
                        short_stop_loss = (df['high'].iloc[-1] + df['open'].iloc[-2])/2
                        sell_price = df['close'].iloc[-1]
                        send_to_telegram(f"매도 - Price: {df['close'].iloc[-1]}, Quantity: {short_quantity}")
                        send_to_telegram(f"손절가 - {short_stop_loss}")
                        sell_signal = True
                        upper_band, lower_band = calculate_bollinger_bands(df, window=20, num_std_dev=2.5)
                        future_close_price = lower_band.iloc[-1] # 과매도
                        now = datetime.datetime.now()
                        entry_time = datetime.datetime(now.year, now.month, now.day, now.hour, 0)

    # 매수 또는 매도 신호가 발생한 경우
    if buy_signal or sell_signal:
        # 주문 정보 가져오기
        order_info = exchange.fetch_order(limit_order['id'], symbol)
        order_status = None
        if order_info is not None:
            order_status = order_info['status']
        if order_status == 'open':
            if (datetime.datetime.now() >= entry_time + datetime.timedelta(hours=(6 - entry_time.hour % 6))):
                exchange.cancel_order(limit_order['id'], symbol)
                buy_signal = False
                sell_signal = False
                send_to_telegram(f'시간초과로 인한 포지션 취소')
        
        else :
            # 지정된 시간이 경과하면 주문을 종료하고 이익을 실현
            if buy_signal and (datetime.datetime.now() >= entry_time + datetime.timedelta(hours=(6 - entry_time.hour % 6))):
                place_limit_order(symbol, 'sell', long_quantity, df['close'].iloc[-1])
                profit = (df['close'].iloc[-1] - buy_price) / buy_price * 100
                send_to_telegram(f"롱포지션 종료 \nQuantity: {long_quantity}\nprofit: {profit}")
                buy_signal = False

            elif sell_signal and (datetime.datetime.now() >= entry_time + datetime.timedelta(hours=(6 - entry_time.hour % 6))):
                place_limit_order(symbol, 'buy', short_quantity, df['close'].iloc[-1])
                profit = -(df['close'].iloc[-1] - sell_price) / sell_price * 100
                send_to_telegram(f"숏포지션 종료 \nQuantity: {short_quantity}\nprofit: {profit}")
                sell_signal = False

            # 과매수시 익절
            if buy_signal == True:
                if future_close_price < df['close'].iloc[-1] :
                    place_limit_order(symbol, 'sell', long_quantity, df['close'].iloc[-1])
                    profit = (df['close'].iloc[-1] - buy_price) / buy_price * 100
                    send_to_telegram(f"롱포지션 종료 \nQuantity: {long_quantity}\nprofit: {profit}")
                    buy_signal = False
                    waiting_buy_signal = False
                    time.sleep(60*60)
                #손절
                elif long_stop_loss > df['close'].iloc[-1]:
                    place_limit_order(symbol, 'sell', long_quantity, df['close'].iloc[-1])
                    profit = (df['close'].iloc[-1] - buy_price) / buy_price * 100
                    send_to_telegram(f"롱포지션 손절 \nQuantity: {long_quantity}\nprofit: {profit}")
                    buy_signal = False
                    waiting_buy_signal = False

            # 과매도시 익절
            elif sell_signal == True:
                if future_close_price > df['close'].iloc[-1]:
                    place_limit_order(symbol, 'buy', short_quantity, df['close'].iloc[-1])
                    profit = -(df['close'].iloc[-1] - sell_price) / sell_price * 100
                    send_to_telegram(f"숏포지션 종료 \nQuantity: {short_quantity}\nprofit: {profit}")
                    sell_signal = False
                    waiting_sell_signal = False
                    time.sleep(60*60)
                #손절
                elif short_stop_loss < df['close'].iloc[-1]:
                    place_limit_order(symbol, 'buy', short_quantity, df['close'].iloc[-1])
                    profit = -(df['close'].iloc[-1] - sell_price) / sell_price * 100
                    send_to_telegram(f"숏포지션 손절 \nQuantity: {short_quantity}\nprofit: {profit}")
                    sell_signal = False
                    waiting_sell_signal = False

            # 1프로 손익시 포지션 종료
            if buy_signal == True:
                if df['close'].iloc[-1]> predicted_buy_low_price + predicted_buy_low_price/100:
                    place_limit_order(symbol, 'sell', long_quantity, df['close'].iloc[-1])
                    profit = (df['close'].iloc[-1] - buy_price) / buy_price * 100
                    send_to_telegram(f"롱포지션 종료 \nQuantity: {long_quantity}\nprofit: {profit}")
                    buy_signal = False
                    waiting_buy_signal = False
            elif sell_signal == True:
                if df['close'].iloc[-1]< predicted_sell_high_price - predicted_sell_high_price/100:
                    place_limit_order(symbol, 'buy', short_quantity, df['close'].iloc[-1])
                    profit = -(df['close'].iloc[-1] - sell_price) / sell_price * 100
                    send_to_telegram(f"숏포지션 종료 \nQuantity: {short_quantity}\nprofit: {profit}")
                    sell_signal = False
                    waiting_sell_signal = False

# 매매 주기 (예: 1초마다 전략 실행)
trade_interval = 1  # 초 단위
count=0
start = True
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
            if start == True:
                # 변동성 조건 임의 계산
                range = (df['high'].iloc[-2] - df['low'].iloc[-2]) * k_value
                target_long = df['close'].iloc[-2] + range
                target_short = df['close'].iloc[-2] - range
                target_long2 = (df['close'].iloc[-2] + range*2)
                target_short2 = (df['close'].iloc[-2] - range*2)
                upper_band, lower_band = calculate_bollinger_bands(df, window=20, num_std_dev=2.5)
                stoch_rsi_k, stoch_rsi_d = stochastic_rsi(df, period=14, smooth_k=3, smooth_d=3)
                rsi = calculate_rsi(df, period=14)
                send_to_telegram(f"현재시간 : {datetime.datetime.now()}")
                send_to_telegram(f"range : {range}\ntarget_long : {target_long}\ntarget_short : {target_short}\ntarget_long2 : {target_long2}\ntarget_short2 : {target_short2}\nupper_band, lower_band : {upper_band.iloc[-1], lower_band.iloc[-1]}\nstoch_rsi_k, stoch_rsi_d : {stoch_rsi_k.iloc[-1], stoch_rsi_d.iloc[-1]}\nrsi : {rsi['rsi'].iloc[-1]}\n종가 : {df['close'].iloc[-1]}")
                send_to_telegram(f"{symbol} 매매 시작")
                start = False
            # 대기 시간
            time.sleep(trade_interval)
        elif stop:
            buy_signal = False
            waiting_buy_signal = False
            sell_signal = False
            waiting_sell_signal = False
            start = True
            time.sleep(60)
    except Exception as e:
        send_to_telegram(f"An error occurred: {e}")
        count+=1
        if count==10:
            stop = True
            count=0
        pass
print("nohup python3 volatility_strategy_binance_auto.py > output.log &")
