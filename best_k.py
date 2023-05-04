from binance.client import Client
import numpy as np
from binance_keys import api_key, api_secret

client = Client(api_key, api_secret)

def get_ror(k=0.5):
    candles = client.futures_klines(symbol="BTCUSDT", interval='4h', limit=7)
    df = pd.DataFrame(candles, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'])
    df = df.astype({'open' : 'float', 'high' : 'float', 'low' : 'float', 'close' : 'float', 'volume' : 'float'})
    df['range'] = (df['high'] - df['low']) * k
    df['target'] = df['open'] + df['range'].shift(1)

    df['ror'] = np.where(df['high'] > df['target'], df['close'] / df['target'],1)
    ror = df['ror'].cumprod()[-2]
    return ror

for k in np.arange(0.1, 1.0, 0.1):
    ror = get_ror(k)
    print("%.1f %f" % (k, ror))
