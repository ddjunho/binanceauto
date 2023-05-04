import math
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
from binance.exceptions import BinanceAPIException
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from binance_keys import api_key, api_secret
from telepot.loop import MessageLoop
tf.config.run_functions_eagerly(True)

# 로그인
client = Client(api_key, api_secret)
COIN = "BTCUSDT" #코인명
bot = telepot.Bot(token="6296102104:AAFC4ddbh7gSgkGOdysFqEBUkIoWXw0-g5A")

def get_balance(ticker):
    # 선물 거래 계좌 잔고 조회
    try:
        # Get futures account information
        info = client.futures_account()
        # Get balances
        balances = info['assets']
        # Find balance for given ticker
        for balance in balances:
            if balance['asset'] == ticker:
                return float(balance['availableBalance'])
        # If ticker not found, return 0
        return 0
    except (requests.exceptions.RequestException, simplejson.errors.JSONDecodeError) as e:
        print(f"에러 발생: {e}")
    return 0
def get_position(ticker):
    # 선물 거래 계좌 포지션 조회
    try:
        # Get futures account information
        info = client.futures_account()
        # Get positions
        positions = info['positions']
        # Find position for given ticker
        for position in positions:
            if position['symbol'] == ticker:
                return float(position['positionAmt'])
        # If ticker not found, return 0
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
def predict_target_prices(ticker):
    candles = client.futures_klines(symbol=ticker, interval='4h', limit=1000)
    df = pd.DataFrame(candles, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'])
    df = df.astype({'open' : 'float', 'high' : 'float', 'low' : 'float', 'close' : 'float', 'volume' : 'float'})
    X = df[['open', 'high', 'low', 'close', 'volume']].values
    X_scaler = StandardScaler()
    X = X_scaler.fit_transform(X)
    y_high = df['high'].values
    y_low = df['low'].values
    y_scaler_high = StandardScaler()
    y_high = y_scaler_high.fit_transform(y_high.reshape((-1, 1)))
    y_scaler_low = StandardScaler()
    y_low = y_scaler_low.fit_transform(y_low.reshape((-1, 1)))
    X_train = []
    y_train_high = []
    y_train_low = []
    data=999
    for i in range(data, len(X)):
        X_train.append(X[i - data:i, :])
        y_train_high.append(y_high[i, 0])
        y_train_low.append(y_low[i, 0])
    X_train = np.array(X_train)
    y_train_high = np.array(y_train_high)
    y_train_low = np.array(y_train_low)
    model_high = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(128, input_shape=(data, 5)),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.005)),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.005)),
        tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.005)),
        tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.005)),
        tf.keras.layers.Dense(1)
    ])
    model_high.compile(optimizer='adam', loss='mse', run_eagerly=True)
    model_high.fit(X_train, y_train_high, epochs=100, verbose=1)
    
    model_low = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(128, input_shape=(data, 5)),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.005)),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.005)),
        tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.005)),
        tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.005)),
        tf.keras.layers.Dense(1)
    ])
    model_low.compile(optimizer='adam', loss='mse', run_eagerly=True)
    model_low.fit(X_train, y_train_low, epochs=100, verbose=1)
    
    last_data = df[['open', 'high', 'low', 'close', 'volume']].iloc[-data:].values
    last_data_mean = last_data.mean(axis=0)
    last_data_std = last_data.std(axis=0)
    last_data = (last_data - last_data_mean) / last_data_std
    last_data = np.expand_dims(last_data, axis=0)
    predicted_price_high = model_high.predict(last_data)
    predicted_price_high = y_scaler_high.inverse_transform(predicted_price_high)
    predicted_price_low = model_low.predict(last_data)
    predicted_price_low = y_scaler_low.inverse_transform(predicted_price_low)
    return float(predicted_price_high.flatten()[0]), float(predicted_price_low.flatten()[0])

def is_bull_market(ticker, time):
    candles = client.futures_klines(symbol=ticker, interval=time, limit=1000)
    df = pd.DataFrame(candles, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'])
    DF = df.astype({'open' : 'float', 'high' : 'float', 'low' : 'float', 'close' : 'float', 'volume' : 'float'})
    # 기술적 지표 추가
    DF['ma5'] = DF['close'].rolling(window=5).mean()
    DF['ma10'] = DF['close'].rolling(window=10).mean()
    DF['ma20'] = DF['close'].rolling(window=20).mean()
    DF['ma60'] = DF['close'].rolling(window=60).mean()
    DF['ma120'] = DF['close'].rolling(window=120).mean()
    # RSI 계산
    delta = DF['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    DF['rsi'] = 100 - (100 / (1 + rs))
    # MACD 계산
    exp1 = DF['close'].ewm(span=12, adjust=False).mean()
    exp2 = DF['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    DF['macd'] = macd
    DF['macdsignal'] = signal
    DF['macdhist'] = hist
    # 결측값 제거
    DF = DF.dropna()
    # 입력 데이터와 출력 데이터 분리
    X = DF[['open', 'high', 'low', 'close', 'volume', 'ma5', 'ma10', 'ma20', 'ma60', 'ma120', 'rsi', 'macd', 'macdsignal', 'macdhist']]
    y = (DF['close'].shift(-1) < DF['close']).astype(int) # time 뒤의 가격이 낮을 확률 예측
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
    global Leverage
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
            bot.sendMessage(chat_id, '현재 레버리지 : Leverage\n레버리지 설정\n/Leverage_1\n/Leverage_5\n/Leverage_10\n/Leverage_20\n/Leverage_40\n/Leverage_60\n/Leverage_100')
        elif msg['text'] == '/Leverage_1':
            bot.sendMessage(chat_id, 'Leverage setting complete!')
            Leverage = 1
        elif msg['text'] == '/Leverage_5':
            bot.sendMessage(chat_id, 'Leverage setting complete!')
            Leverage = 5
        elif msg['text'] == '/Leverage_10':
            bot.sendMessage(chat_id, 'Leverage setting complete!')
            Leverage = 10
        elif msg['text'] == '/Leverage_20':
            bot.sendMessage(chat_id, 'Leverage setting complete!')
            Leverage = 20
        elif msg['text'] == '/Leverage_40':
            bot.sendMessage(chat_id, 'Leverage setting complete!')
            Leverage = 40
        elif msg['text'] == '/Leverage_60':
            bot.sendMessage(chat_id, 'Leverage setting complete!')
            Leverage = 60
        elif msg['text'] == '/Leverage_100':
            bot.sendMessage(chat_id, 'Leverage setting complete!')
            Leverage = 100
        elif msg['text'] == '/help':
            bot.sendMessage(chat_id, '/start - 시작\n/stop - 중지\n/isForceStart - 일부 매매조건을 무시하고 매매합니다.\n/isNormalStart - 일부 매매조건을 무시하지 않고 매매합니다.\n/set_Leverage - 레버리지 설정')
MessageLoop(bot, handle).run_as_thread()
def send_message(message):
    chat_id = "5820794752"
    bot.sendMessage(chat_id, message)
def buy_coin(buy_amount):
    # Get the ticker information for COIN
    ticker = client.futures_ticker(symbol=COIN)
    price = float(ticker['lastPrice'])
    # Calculate the amount of BTC
    btc_amount = buy_amount / price
    btc_amount = round(btc_amount,3)-0.001
    client.futures_create_order(symbol=COIN, side='BUY', type='MARKET', quantity=btc_amount)
# 스케줄러 실행
def job():
    usd = get_balance('USDT')
    btc = get_position(COIN)
    multiplier = 1
    last_buy_time = None
    time_since_last_buy = None
    buy_amount = usd # 매수 금액 계산
    bull_market = False
    start = True
    isForceStart = False
    while stop == False:
        try:
            now = datetime.now()
            current_price = get_current_price(COIN)
            client.futures_change_leverage(symbol=COIN, leverage=Leverage)
            if now.hour % 2 == 0 and now.minute == 0 or start == True:
                usd = get_balance('USDT')
                buy_amount = usd
                sell_price, target_price = predict_target_prices(COIN)
                PriceEase = round((sell_price - target_price) * 0.1, 1)
                hour_1 = round((1-is_bull_market(COIN, '1h'))*100,5)
                hour_2 = round((1-is_bull_market(COIN, '2h'))*100,5)
                hour_4 = round((1-is_bull_market(COIN, '4h'))*100,5)
                hour_6 = round((1-is_bull_market(COIN, '6h'))*100,5)
                hour_8 = round((1-is_bull_market(COIN, '8h'))*100,5)
                hour_24 = round((1-is_bull_market(COIN, '1d'))*100,5)
                if hour_24 >= 50 and hour_4 >= 45 and hour_8 >= 45:
                    bull_market = True
                else:
                    bull_market = False
                formatted = now.strftime('%Y-%m-%d %H:%M:%S')
                message = f"Local time : {formatted} UTC\n매수가 조회 : {target_price}\n매도가 조회 : {sell_price}\n현재가 조회 : {current_price}\n1시간뒤 크거나 같을 확률 예측 : {hour_1}%\n2시간뒤 크거나 같을 확률 예측 : {hour_2}%\n4시간뒤 크거나 같을 확률 예측 : {hour_4}%\n6시간뒤 크거나 같을 확률 예측 : {hour_6}%\n8시간뒤 크거나 같을 확률 예측 : {hour_8}%\n매매조건 : {bull_market}\n조건무시 : {isForceStart}\n내일 크거나 같을 확률{hour_24}%\n달러잔고 : {usd}\n비트코인잔고 : {btc}\n목표가 완화 : {PriceEase}\n레버리지 : {Leverage}"
                send_message(message)
                start = False
            # 매수 조건
            if current_price <= target_price + PriceEase:
                usd = get_balance('USDT')
                if bull_market==True or isForceStart==True:
                    if usd > 35 and target_price + PriceEase < sell_price-(PriceEase*5):
                        try:
                            buy_coin(buy_amount)
                            pass
                        except BinanceAPIException as e:
                            message = f"매수 실패 {e}!"
                            print(e)
                            send_message(message)
                        else:
                            message = f"매수 성공 !"
                            send_message(message)
                            last_buy_time = now
                            multiplier = 1
            # 매도 조건
            else:
                if current_price >= sell_price-(PriceEase*multiplier):
                    btc = get_balance('BTC')
                    if btc > 0.00008 and btc is not None:
                        client.futures_create_order(symbol=COIN, side='SELL', type='MARKET', quantity=btc)
                        message = f"매도 완료 !"
                        send_message(message)
                        isForceStart = False
            # PriceEase 증가 조건
            if last_buy_time is not None:
                time_since_last_buy = now - last_buy_time
                if time_since_last_buy.total_seconds() >= 3600: # 1시간마다
                    multiplier += 1
                    last_buy_time = now
                    if multiplier>5:
                        multiplier=5
                        last_buy_time = None
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
