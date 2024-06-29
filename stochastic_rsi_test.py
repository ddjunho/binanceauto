import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization
import ccxt

################ 백테스트 ################

# 백테스트 기본 셋팅
from binance_keys import api_key, api_secret

exchange = ccxt.binance({
    'rateLimit': 1000,
    'enableRateLimit': True,
    'apiKey': api_key,
    'secret': api_secret,
    'options': {
        'defaultType': 'future'
    }
})
symbol = 'ETHUSDT'
timeframe = '5m'  # 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M

def get_candles(exchange, symbol, timeframe='6h', limit=100):
    candles = exchange.fetch_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        since=None,
        limit=limit
    )
    df = pd.DataFrame(data=candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def calculate_rsi(data, period=14):
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

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
    data = calculate_rsi(data, 15)
    # 스토케스틱 RSI (%K) 계산
    min_rsi = data['rsi'].rolling(window=period, center = False).min()
    max_rsi = data['rsi'].rolling(window=period, center = False).max()
    stoch = 100 * (data['rsi'] - min_rsi) / (max_rsi - min_rsi)
    stoch_rsi_k = stoch.rolling(window=smooth_k, center = False).mean()
    # 스토케스틱 RSI (%D) 계산
    stoch_rsi_d = stoch_rsi_k.rolling(window=smooth_d, center = False).mean()

    return stoch_rsi_k, stoch_rsi_d


df = get_candles(exchange, symbol, timeframe='5m', limit=50)
stoch_rsi_k, stoch_rsi_d = stochastic_rsi(df, period=14, smooth_k=3, smooth_d=3)

print(stoch_rsi_k.iloc[-1],'\t', stoch_rsi_d.iloc[-1])
print(stoch_rsi_k.iloc[-1],'\t', stoch_rsi_d.iloc[-1])
print(stoch_rsi_k.iloc[-2],'\t', stoch_rsi_d.iloc[-2])
data = calculate_rsi(df, period=14)
print(data.iloc[-1])
print(data.iloc[-2])
