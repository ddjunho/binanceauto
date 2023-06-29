from binance.client import Client
import numpy as np
import pandas as pd
from binance_keys import api_key, api_secret
from statsmodels.tsa.arima.model import ARIMA
import schedule
import telepot
from telepot.loop import MessageLoop

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
def get_best_ticker_and_k_and_type():
  surge_tickers = get_surge_tickers()
  best_ticker = None
  best_k = None
  best_ror = 0
  best_type = None
  for coin, ratio in surge_tickers:
    for k in np.arange(0.1, 1.0, 0.1):
      low_ror, high_ror = get_ror_by_low_and_high(k_low=k, coin=coin)
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
  return best_ticker, best_k, best_type

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
    balance = client.futures_account_balance()
    usdt_balance = float([item["balance"] for item in balance if item["asset"] == "USDT"][0])
    market_price = float(client.futures_mark_price(symbol=symbol)["markPrice"])
    quantity = usdt_balance * leverage / market_price
    precision = client.futures_exchange_info()["symbols"][symbol]["quantityPrecision"]
    quantity = math.floor(quantity * 10**precision) / 10**precision
    return quantity
  
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

def send_message(message):
    chat_id = "5820794752"
    bot.sendMessage(chat_id, message)

# 시계열 분석 함수
predicted_close_price = 0
def predict_price(ticker):
    global predicted_close_price
    candles = client.futures_klines(symbol=ticker, interval='15m', limit=96)
    df = pd.DataFrame(candles, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored']) # 데이터 프레임 생성
    df = df.astype({'open' : 'float', 'high' : 'float', 'low' : 'float', 'close' : 'float', 'volume' : 'float'})
    df = df.reset_index()
    df['ds'] = df['index']
    df['y'] = df['close']
    data = df[['ds','y']]
    model = ARIMA(data['y'], order=(1,1,1)) # arima 모델 생성
    model_fit = model.fit() # 모델 학습
    current_time = data.iloc[-1]['ds'].hour
    target_time = (current_time // 4 + 1) * 4
    diff_time = target_time - current_time
    n = diff_time * 4 + 1
    forecast = model_fit.forecast(n) 
    hour = forecast.iloc[-1]['ds'].hour
    if hour % 4 == 0:
        closeValue = forecast[0][-1]
        predicted_close_price = closeValue
predict_price(best_ticker)
schedule.every(15).minutes.do(lambda: predict_price(best_ticker)) # 15분마다 예측 함수 실행

message = f"Local time : {formatted} UTC\n매수가 조회 : {target_price}\n매도가 조회 : {sell_price}\n변동성돌파종가 조회 : {close_price}\n현재가 조회 : {current_price}\n1시간뒤 크거나 같을 확률 예측 : {hour_1}%\n2시간뒤 크거나 같을 확률 예측 : {hour_2}%\n4시간뒤 크거나 같을 확률 예측 : {hour_4}%\n6시간뒤 크거나 같을 확률 예측 : {hour_6}%\n8시간뒤 크거나 같을 확률 예측 : {hour_8}%\n매매조건 : {bull_market}\n조건무시 : {isForceStart}\n내일 크거나 같을 확률{hour_24}%\n달러잔고 : {usd}\n비트코인잔고 : {btc}\n목표가 완화 : {PriceEase}\n레버리지 : {Leverage}"
send_message(message)
# 가장 수익률이 높은 티커와 k값과 구분을 구하고 출력하기
best_ticker, best_k, best_type = get_best_ticker_and_k_and_type()
print("가장 수익률이 높은 티커와 k값과 구분은 다음과 같습니다.")
print("티커:", best_ticker)
print("구분:", best_type) # low와 high를 출력하기
print("k값:", best_k)
print("수익률:", best_ror)

# 목표가격을 구하고 출력하기
target_price = get_target_price(best_ticker, best_k, best_type)
print("목표가격:", target_price)

quantity = calculate_quantity(best_ticker, 10)
buy(best_ticker, quantity, 10)
sell(best_ticker, quantity, 10)
close_position("ETHUSDT")
