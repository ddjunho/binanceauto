from binance.client import Client
import numpy as np
import pandas as pd
from binance_keys import api_key, api_secret

client = Client(api_key, api_secret)

# 거래량이 폭증한 티커를 반환하는 함수 정의
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
        if max_volume >= mean_volume * 1.5 and mean_volume > 0: # 거래량이 증가한 비율을 계산하고 surge_list에 티커와 함께 추가하기
            surge_ratio = max_volume / mean_volume
            surge_list.append((ticker, surge_ratio))
    surge_list.sort(key=lambda x: x[1], reverse=True)
    return surge_list[:3]
COIN_1 = get_surge_tickers()[0][0]
COIN_2 = get_surge_tickers()[1][0]
COIN_3 = get_surge_tickers()[2][0]

def get_low_and_high(k_low=0.5, k_high=0.5, coin="BTCUSDT"):
    candles = client.futures_klines(symbol=coin, interval='4h', limit=6)
    df = pd.DataFrame(candles, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'])
    df = df.astype({'open' : 'float', 'high' : 'float', 'low' : 'float', 'close' : 'float', 'volume' : 'float'})
    df['range'] = (df['high'] - df['low'])
    df['target_low'] = df['open'] + df['range'].shift(1) * k_low
    df['ror_low'] = np.where(df['high'] > df['target_low'], df['close'] / df['target_low'],1)
    low = df['ror_low'].cumprod().iloc[-2]
    df['target_high'] = df['open'] - df['range'].shift(1) * k_high
    df['ror_high'] = np.where(df['low'] < df['target_high'], df['target_high'] / df['close'],1)
    high = df['ror_high'].cumprod().iloc[-2]
    return low, high

coins = [COIN_1, COIN_2, COIN_3]

best_ticker = None
best_k = None
best_ror = 0
best_type = None

for coin in coins:
    for k in np.arange(0.1, 1.0, 0.1):
        low, high = get_low_and_high(k_low=k, coin=coin)
        if low > best_ror:
            best_ticker = coin
            best_k = k
            best_ror = low
            best_type = "low"
        if high > best_ror:
            best_ticker = coin
            best_k = k
            best_ror = high
            best_type = "high"

print(coins)
print("가장 수익률이 높은 티커와 k값은 다음과 같습니다.")
print("티커:", best_ticker)
print("구분:", best_type) # low와 high를 출력하기
print("k값:", best_k)
print("수익률:", best_ror)
def get_target_price(ticker_list):
    #변동성 돌파 전략
    low_high_dict = {} #저가와 고가를 저장할 딕셔너리
    for ticker in ticker_list: #ticker리스트를 순회하면서
        candles = client.futures_klines(symbol=ticker, interval='4h', limit=6)
        df = pd.DataFrame(candles, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'])
        df = df.astype({'open' : 'float', 'high' : 'float', 'low' : 'float', 'close' : 'float', 'volume' : 'float'})
        low_price = df.iloc[0]['close'] + (df.iloc[0]['high'] - df.iloc[0]['low']) * 0.6 #저가 계산
        high_price = df.iloc[0]['close'] + (df.iloc[0]['high'] - df.iloc[0]['low']) * 1.0 #고가 계산
        low_high_dict[ticker] = (low_price, high_price) #딕셔너리에 저장
    return low_high_dict #딕셔너리 반환
