from binance.client import Client
import numpy as np
import pandas as pd
from binance_keys import api_key, api_secret

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

predicted_close_price = 0
def predict_price(ticker):
    global predicted_close_price
    candles = client.futures_klines(symbol=ticker, interval='15m', limit=108)
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
