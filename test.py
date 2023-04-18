import time
import datetime
import json
import pandas as pd
import numpy as np
import schedule
import telepot
import tensorflow as tf
import requests.exceptions
import simplejson.errors
from binance.client import Client
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from binance_keys import api_key, api_secret
from telepot.loop import MessageLoop
tf.config.run_functions_eagerly(True)
buy_unit = 0.2  # 분할 매수 금액 단위 설정

# 로그인
client = Client(api_key, api_secret)
COIN = "BTCUSDT" #코인명
bot = telepot.Bot(token="6296102104:AAFC4ddbh7gSgkGOdysFqEBUkIoWXw0-g5A")

def get_balance(ticker):
    # 잔고 조회
    try:
        balances = client.futures_account_balance()
        for b in balances:
            if b['asset'] == ticker:
                if b['balance'] is not None:
                    return float(b['balance'])
                else:
                    return 0
        # 해당 티커의 잔고가 없을 경우 0을 반환
        return 0
    except (requests.exceptions.RequestException, simplejson.errors.JSONDecodeError) as e:
        print(f"에러 발생: {e}")
    return 0

def get_current_price(ticker):
    # 현재가 조회
    try:
        return float(client.futures_symbol_ticker(symbol=ticker)['price'])
    except Exception as e:
        print(e)

def predict_target_price(ticker, target_type):
    # 데이터 불러오기
    candles = list(map(float, client.futures_klines(symbol=ticker, interval='4h', limit=1000)))
    df = pd.DataFrame(candles, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'])
    # 입력 데이터 전처리
    X = df[['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']]
    X_scaler = StandardScaler()
    X = X_scaler.fit_transform(X)
    # 출력 데이터 전처리
    y = df[target_type].values
    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y.reshape((-1, 1)))
    # 학습 데이터 생성
    X_train = []
    y_train = []
    data=999
    for i in range(data, len(X)):
        X_train.append(X[i - data:i, :])
        y_train.append(y[i, 0])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    # Tensorflow 모델 구성
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(128, input_shape=(data, 9)),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.005)),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.005)),
        tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.005)),
        tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.005)),
        tf.keras.layers.Dense(1)
    ])
    # 모델 컴파일
    model.compile(optimizer='adam', loss='mse', run_eagerly=True)
    # 학습
    model.fit(X_train, y_train, epochs=1, verbose=1)
    # 새로운 데이터에 대한 예측
    last_data = df[['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']].iloc[-data:].values
    last_data_mean = last_data.mean(axis=0)
    last_data_std = last_data.std(axis=0)
    last_data = (last_data - last_data_mean) / last_data_std
    # 예측할 데이터의 shape를 (1,999, values)로 변경
    last_data = np.expand_dims(last_data, axis=0)
    predicted_price = model.predict(last_data)
    predicted_price = y_scaler.inverse_transform(predicted_price)
    predicted_price = predicted_price.flatten()[0]  # 이중 리스트를 일차원으로 변경하고 첫 번째 원소를 선택
    return float(predicted_price)

def is_bull_market(ticker, time):
    candles = client.futures_klines(symbol=ticker, interval=time, limit=1000)
    candles = pd.DataFrame(candles, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'])
    # 기술적 지표 추가
    candles['ma5'] = candles['close'].rolling(window=5).mean()
    candles['ma10'] = candles['close'].rolling(window=10).mean()
    candles['ma20'] = candles['close'].rolling(window=20).mean()
    candles['ma60'] = candles['close'].rolling(window=60).mean()
    candles['ma120'] = candles['close'].rolling(window=120).mean()
    # RSI 계산
    delta = candles['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    candles['rsi'] = 100 - (100 / (1 + rs))
    # MACD 계산
    exp1 = candles['close'].ewm(span=12, adjust=False).mean()
    exp2 = candles['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    candles['macd'] = macd
    candles['macdsignal'] = signal
    candles['macdhist'] = hist
    # 결측값 제거
    candles = candles.dropna()
    # 입력 데이터와 출력 데이터 분리
    X = candles[['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored', 'rsi', 'macd', 'macdsignal', 'macdhist']]
    y = (candles['close'].shift(-1) < candles['close']).astype(int) # time 뒤의 가격이 낮을 확률 예측
    # 학습 데이터와 검증 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    # 모델 구성
    model = RandomForestClassifier(n_estimators=100)
    # 학습
    model.fit(X_train, y_train)
    # 예측 확률 계산
    proba = model.predict_proba(X_test.iloc[-1].values.reshape(1,-1))[0][1]
    proba = round(proba, 4)
    return proba
stop = False
isForceStart = False
Leverage = 1
def handle(msg):
    global stop
    global isForceStart
    content_type, chat_type, chat_id = telepot.glance(msg)
    if content_type == 'text':
        if msg['text'] == '/start':
            bot.sendMessage(chat_id, 'Starting...')
            stop = False
        elif msg['text'] == '/stop':
            bot.sendMessage(chat_id, 'Stopping...')
            stop = True
        elif msg['text'] == '/isForceStart':
            bot.sendMessage(chat_id, '일부 매매조건을 무시하고 매매합니다...')
            isForceStart = True
        elif msg['text'] == '/isNormalStart':
            bot.sendMessage(chat_id, '일부 매매조건을 무시하지않고 매매합니다....')
            isForceStart = False
        elif msg['text'] == '/set_Leverage':
            bot.sendMessage(chat_id, '현재 레버리지 : Leverage\n레버리지 설정\n/Leverage = 1\n/Leverage = 5\n/Leverage = 10\n/Leverage = 20\n/Leverage = 40\n/Leverage = 60\n/Leverage = 100')
        elif msg['text'] == '/Leverage = 1':
            bot.sendMessage(chat_id, 'Leverage setting complete!')
            Leverage = 1
        elif msg['text'] == '/Leverage = 5':
            bot.sendMessage(chat_id, 'Leverage setting complete!')
            Leverage = 5
        elif msg['text'] == '/Leverage = 10':
            bot.sendMessage(chat_id, 'Leverage setting complete!')
            Leverage = 10
        elif msg['text'] == '/Leverage = 20':
            bot.sendMessage(chat_id, 'Leverage setting complete!')
            Leverage = 20
        elif msg['text'] == '/Leverage = 40':
            bot.sendMessage(chat_id, 'Leverage setting complete!')
            Leverage = 40
        elif msg['text'] == '/Leverage = 60':
            bot.sendMessage(chat_id, 'Leverage setting complete!')
            Leverage = 60
        elif msg['text'] == '/Leverage = 100':
            bot.sendMessage(chat_id, 'Leverage setting complete!')
            Leverage = 100
        elif msg['text'] == '/help':
            bot.sendMessage(chat_id, '/start - 시작\n/stop - 중지\n/isForceStart - 일부 매매조건을 무시하고 매매합니다.\n/isNormalStart - 일부 매매조건을 무시하지 않고 매매합니다.\n/Leverage - 레버리지 설정')
MessageLoop(bot, handle).run_as_thread()
def send_message(message):
    chat_id = "5820794752"
    bot.sendMessage(chat_id, message)
# 스케줄러 실행
def job():
    usd = get_balance('USDT')
    btc = get_balance('BTC')
    multiplier = 1
    last_buy_time = None
    time_since_last_buy = None
    buy_amount = usd * buy_unit # 분할 매수 금액 계산
    start = True
    bull_market = False
    while stop == False:
        try:
            now = datetime.now()
            current_price = get_current_price(COIN)
            client.futures_change_leverage(symbol=COIN, leverage=Leverage)
            if now.hour % 3 == 0 and now.minute == 0 or start == True:
                if usd <= get_balance('USDT'):
                    usd = get_balance('USDT')
                    buy_amount = usd * buy_unit
                target_price = predict_target_price(COIN, "low")
                sell_price = predict_target_price(COIN, "high")
                PriceEase = round((sell_price - target_price) * 0.1, 1)
                hour_1 = 1-is_bull_market(COIN, '1h')
                hour_4 = 1-is_bull_market(COIN, '4h')
                hour_8 = 1-is_bull_market(COIN, '8h')
                hour_24 = 1-is_bull_market(COIN, '1d')
                if hour_1 >= 0.45 and hour_3 >= 0.45 and hour_6 >= 0.45:
                    bull_market = True
                else:
                    bull_market = False
                message = f"매수가 조회 : {target_price}\n매도가 조회 : {sell_price}\n현재가 조회 : {current_price}\n1시간뒤 크거나 같을 확률 예측 : {hour_1*100}%\n4시간뒤 크거나 같을 확률 예측 : {hour_4*100}%\n8시간뒤 크거나 같을 확률 예측 : {hour_8*100}%{bull_market}\n내일 크거나 같을 확률{hour_24*100}%\n원화잔고 : {usd}\n비트코인잔고 : {btc}\n목표가 완화 : {PriceEase}\n레버리지 : {Leverage}"
                send_message(message)
                start = False
            # 매수 조건
            if current_price <= target_price + PriceEase:
                usd = get_balance('USDT')
                if bull_market==True or isForceStart==True:
                    if usd > 10 and target_price + PriceEase < sell_price-(PriceEase*3):
                        if get_balance('USDT') < usd * buy_unit:
                            buy_amount = usd
                        client.futures_create_order(symbol=COIN, side='BUY', type='MARKET', quantity=buy_amount)
                        last_buy_time = now
                        multiplier = 1
                        print(now, "매수")
            # 매도 조건
            else:
                if current_price >= sell_price-(PriceEase*multiplier):
                    btc = get_balance('BTC')
                    if btc > 0.00008 and btc is not None:
                        client.futures_create_order(symbol=COIN, side='SELL', type='MARKET', quantity=btc)
                        print(now, "매도")
            # PriceEase 증가 조건
            if last_buy_time is not None:
                time_since_last_buy = now - last_buy_time
                if time_since_last_buy.total_seconds() >= 3600: # 1시간마다
                    multiplier += 1
                    if multiplier>5:
                        multiplier=5
                        last_buy_time = None
                    last_buy_time = now
            time.sleep(1)
        except Exception as e:
            print(e)
            time.sleep(1)
schedule.every(1).seconds.do(job)
while True:
    try:
        schedule.run_pending()
        time.sleep(1)
    except Exception as e:
        print(e)
        time.sleep(1)
