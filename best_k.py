from binance.client import Client
import numpy as np
import pandas as pd
from binance_keys import api_key, api_secret

client = Client(api_key, api_secret)

def get_low_and_high(k_low=0.5, k_high=0.5):
    candles = client.futures_klines(symbol="BTCUSDT", interval='4h', limit=6)
    df = pd.DataFrame(candles, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'])
    df = df.astype({'open' : 'float', 'high' : 'float', 'low' : 'float', 'close' : 'float', 'volume' : 'float'})
    df['range'] = (df['high'] - df['low'])
    df['target_low'] = df['open'] + df['range'].shift(1) * k_low
    df['target_high'] = df['open'] - df['range'].shift(1) * k_high

    df['low'] = np.where(df['low'] < df['target_low'], df['close'] / df['target_low'],1)
    low = df['low'].cumprod().iloc[-2]
    df['high'] = np.where(df['high'] > df['target_high'], df['close'] / df['target_high'],1)
    high = df['high'].cumprod().iloc[-2]
    return low, high

for k in np.arange(0.1, 1.0, 0.1):
    low, high = get_low_and_high(k_low=k)
    print("%.1f %f %f" % (k, low))

for k in np.arange(1.0, 2.0, 0.1):
    low, high = get_low_and_high(k_high=k)
    print("%.1f %f %f" % (k, high))
