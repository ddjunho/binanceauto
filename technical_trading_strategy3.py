from binance.client import Client
import numpy as np
import pandas as pd
from pandas import Timestamp
from binance_keys import api_key, api_secret
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import schedule
import telepot
from telepot.loop import MessageLoop
from datetime import datetime, timedelta
import math
bot = telepot.Bot(token="6296102104:AAFC4ddbh7gSgkGOdysFqEBUkIoWXw0-g5A")
client = Client(api_key, api_secret)

# 거래량이 폭증한 티커와 증가 비율을 반환하는 함수 정의
def get_surge_tickers():
  tickers = client.get_all_tickers()
  df_tickers = pd.DataFrame(tickers)
  df_tickers = df_tickers[df_tickers['symbol'].str.contains('USDT')]
  exchange_info = client.futures_exchange_info()
  symbols = [item['symbol'] for item in exchange_info['symbols']]
  df_tickers = df_tickers[df_tickers['symbol'].isin(symbols)]
  ticker_list = df_tickers['symbol'].tolist()
  surge_list = []
  for ticker in ticker_list:
    candles = client.futures_klines(symbol=ticker, interval='4h', limit=6)
    df = pd.DataFrame(candles, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'])
    df['volume'] = df['volume'].astype(float)
    mean_volume = df['volume'].mean()
    max_volume = df['volume'].max()
    if max_volume >= mean_volume * 1.5 and mean_volume > 0:
      # 거래량이 증가한 비율을 계산하고 surge_list에 티커와 함께 추가하기
      surge_ratio = max_volume / mean_volume
      surge_list.append((ticker, surge_ratio))
  surge_list.sort(key=lambda x: x[1], reverse=True)
  return surge_list[:3]


# 저가와 고가 기준의 수익률을 반환하는 함수 정의
def get_ror_by_low_and_high(k_low=0.5, k_high=0.5, coin="BTCUSDT"):
  candles = client.futures_klines(symbol=coin, interval='4h', limit=6)
  df = pd.DataFrame(candles, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'])
  df = df.astype({'open' : 'float', 'high' : 'float', 'low' : 'float', 'close' : 'float', 'volume' : 'float'})
  df['range'] = (df['high'] - df['low'])
  df['target_low'] = df['open'] + df['range'].shift(1) * k_low
  df['ror_low'] = np.where(df['high'] > df['target_low'], df['close'] / df['target_low'],1)
  low_ror = df['ror_low'].cumprod().iloc[-2]
  df['target_high'] = df['open'] - df['range'].shift(1) * k_high
  df['ror_high'] = np.where(df['low'] < df['target_high'], df['target_high'] / df['close'],1)
  high_ror = df['ror_high'].cumprod().iloc[-2]
  return low_ror, high_ror

# 가장 수익률이 높은 티커와 k값과 구분을 반환하는 함수 정의
def get_best_ticker_and_k_and_type(tickers):
  best_ticker = None
  best_k = None
  best_ror = 0
  best_type = None
  for coin, ratio in tickers:
    for k in np.arange(0.1, 1.0, 0.1):
      low_ror, _ = get_ror_by_low_and_high(k_low=k, coin=coin) 
      high_ror, _ = get_ror_by_low_and_high(k_high=k, coin=coin) 
      if low_ror > best_ror:
        best_ticker = coin
        best_k = k
        best_ror = low_ror
        best_type = "low"
      if high_ror > best_ror:
        best_ticker = coin
        best_k = k
        best_ror = high_ror
        best_type = "high"
  return best_ticker, best_k, best_type, best_ror


def predict_price_movement(ticker):
    candles = client.futures_klines(symbol=ticker, interval='4h', limit=500)
    df = pd.DataFrame(candles, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'])
    DF = df.astype({'open' : 'float', 'high' : 'float', 'low' : 'float', 'close' : 'float', 'volume' : 'float'})
    # 기술적 지표 추가
    DF['ma5'] = DF['close'].rolling(window=5).mean()
    DF['ma10'] = DF['close'].rolling(window=10).mean()
    DF['ma20'] = DF['close'].rolling(window=20).mean()
    DF['ma60'] = DF['close'].rolling(window=60).mean()
    DF['ma120'] = DF['close'].rolling(window=120).mean()
    # 결측값 제거
    DF = DF.dropna()
    # 입력 데이터와 출력 데이터 분리
    X = DF[['open', 'high', 'low', 'close', 'volume', 'ma5', 'ma10', 'ma20', 'ma60', 'ma120']]
    y = (DF['close'].shift(-1) > DF['close']).astype(int) # time 뒤의 가격이 낮을 확률 예측
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

def predict_rsi(ticker):
    candles = client.futures_klines(symbol=ticker, interval='4h', limit=500)
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
    # 결측값 제거
    DF = DF.dropna()
    # 입력 데이터와 출력 데이터 분리
    X = DF[['open', 'high', 'low', 'close', 'volume', 'ma5', 'ma10', 'ma20', 'ma60', 'ma120']]
    y = DF['rsi']
    # 학습 데이터와 검증 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    # 모델 구성
    model = RandomForestRegressor(n_estimators=100)
    # 학습
    model.fit(X_train, y_train)
    # 예측
    rsi_pred = model.predict(X_test.iloc[-1].values.reshape(1, -1))[0]
    rsi_pred = round(rsi_pred, 4)
    return rsi_pred


def predict_macd(ticker):
    candles = client.futures_klines(symbol=ticker, interval='4h', limit=500)
    df = pd.DataFrame(candles, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'])
    DF = df.astype({'open' : 'float', 'high' : 'float', 'low' : 'float', 'close' : 'float', 'volume' : 'float'})
    # 기술적 지표 추가
    DF['ma5'] = DF['close'].rolling(window=5).mean()
    DF['ma10'] = DF['close'].rolling(window=10).mean()
    DF['ma20'] = DF['close'].rolling(window=20).mean()
    DF['ma60'] = DF['close'].rolling(window=60).mean()
    DF['ma120'] = DF['close'].rolling(window=120).mean()
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
    X = DF[['open', 'high', 'low', 'close', 'volume', 'ma5', 'ma10', 'ma20', 'ma60', 'ma120']]
    y = DF['macd']
    # 학습 데이터와 검증 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    # 모델 구성
    model = RandomForestRegressor(n_estimators=100)
    # 학습
    model.fit(X_train, y_train)
    # 예측
    macd_pred = model.predict(X_test.iloc[-1].values.reshape(1, -1))[0]
    macd_pred = round(macd_pred, 4)
    return macd_pred



# 목표가격을 반환하는 함수 정의
def get_target_price(ticker, k, type):
  #변동성 돌파 전략
  candles = client.futures_klines(symbol=ticker, interval='4h', limit=6)
  df = pd.DataFrame(candles, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'])
  df = df.astype({'open' : 'float', 'high' : 'float', 'low' : 'float', 'close' : 'float', 'volume' : 'float'})
  if type == "low":
    low_price = df.iloc[0]['close'] + (df.iloc[0]['high'] - df.iloc[0]['low']) * k #저가 계산
    return low_price
  elif type == "high":
    high_price = df.iloc[0]['close'] - (df.iloc[0]['high'] - df.iloc[0]['low']) * k #고가 계산
    return high_price

# 코인개수 계산 함수 정의
def calculate_quantity(symbol, leverage):
    try:
        balance = client.futures_account_balance()
        usdt_balance = float([item["balance"] for item in balance if item["asset"] == "USDT"][0])
        market_price = float(client.futures_mark_price(symbol=symbol)["markPrice"])
        exchange_info = client.futures_exchange_info()
        symbol_info = next(item for item in exchange_info["symbols"] if item["symbol"] == symbol)
        precision = symbol_info["quantityPrecision"]
        quantity = usdt_balance * leverage / market_price
        quantity = math.floor(quantity * 10**precision) / 10**precision
        return quantity
    
    except Exception as e:
        print(f"An error occurred while calculating the quantity: {e}")
        return None
# 매수 함수 정의
def buy(symbol, quantity, leverage):
    client.futures_change_leverage(symbol=symbol, leverage=leverage)
    order = client.futures_create_order(
        symbol=symbol,
        side=SIDE_BUY,
        type=ORDER_TYPE_MARKET,
        quantity=quantity)

# 매도 함수 정의
def sell(symbol, quantity, leverage):
    client.futures_change_leverage(symbol=symbol, leverage=leverage)
    order = client.futures_create_order(
        symbol=symbol,
        side=SIDE_SELL,
        type=ORDER_TYPE_MARKET,
        quantity=quantity)
    
#포지션 종료함수 정의
def close_position(symbol):
  try:
    position = client.futures_position_information(symbol=symbol)
    long_quantity = float([item["positionAmt"] for item in position if item["positionSide"] == "LONG"][0])
    short_quantity = float([item["positionAmt"] for item in position if item["positionSide"] == "SHORT"][0])
    if long_quantity > 0:
        client.futures_create_order(symbol=symbol, side="SELL", type="MARKET", quantity=long_quantity)
        print(f"Closed long position of {long_quantity} {symbol} at market price.")
    if short_quantity < 0:
        client.futures_create_order(symbol=symbol, side="BUY", type="MARKET", quantity=-short_quantity)
        print(f"Closed short position of {-short_quantity} {symbol} at market price.")
    if long_quantity == 0 and short_quantity == 0:
        print(f"No position of {symbol} to close.")
  except IndexError:
        print(f"No position of {symbol} to close.")
  except Exception as e:
      print(f"An error occurred while closing the position: {e}")

def send_message(message):
    chat_id = "5820794752"
    bot.sendMessage(chat_id, message)


best_ticker, best_k, best_type, best_ror = get_best_ticker_and_k_and_type(get_surge_tickers())

# schedule.every(15).minutes.do(lambda: predict_price(best_ticker)) # 15분마다 예측 함수 실행

# message = f"Local time : {formatted} UTC\n매수가 조회 : {target_price}\n매도가 조회 : {sell_price}\n변동성돌파종가 조회 : {close_price}\n현재가 조회 : {current_price}\n1시간뒤 크거나 같을 확률 예측 : {hour_1}%\n2시간뒤 크거나 같을 확률 예측 : {hour_2}%\n4시간뒤 크거나 같을 확률 예측 : {hour_4}%\n6시간뒤 크거나 같을 확률 예측 : {hour_6}%\n8시간뒤 크거나 같을 확률 예측 : {hour_8}%\n매매조건 : {bull_market}\n조건무시 : {isForceStart}\n내일 크거나 같을 확률{hour_24}%\n달러잔고 : {usd}\n비트코인잔고 : {btc}\n목표가 완화 : {PriceEase}\n레버리지 : {Leverage}"
# send_message(message)
# 가장 수익률이 높은 티커와 k값과 구분을 구하고 출력하기
print("가장 수익률이 높은 티커와 k값과 구분은 다음과 같습니다.")
print("티커:", best_ticker)
print("구분:", best_type) # low와 high를 출력하기
print("k값:", best_k)
print("수익률:", best_ror)

# 목표가격을 구하고 출력하기
target_price = get_target_price(best_ticker, best_k, best_type)
print("목표가격:", target_price)
movement = predict_price_movement(best_ticker)
rsi = predict_rsi(best_ticker)
macd = predict_macd(best_ticker)
if macd>0:
   macd_indicators=True
else:macd_indicators=False
print("상승확률:", movement*100,"%")
print("RSI지표:", rsi)
print("MACD지표:",macd, macd_indicators)

quantity = calculate_quantity(best_ticker, 10)
print("코인개수:",quantity)
# buy(best_ticker, quantity, 10)
# sell(best_ticker, quantity, 10)
print("포지션:",close_position(best_ticker))
