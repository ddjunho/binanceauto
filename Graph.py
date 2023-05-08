import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

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
    
    # 주가 그래프
    plt.plot(DF['close'])
    plt.title('Close Price')
    plt.show()

    # 이동 평균선 그래프
    plt.plot(DF['ma5'], label='MA5')
    plt.plot(DF['ma10'], label='MA10')
    plt.plot(DF['ma20'], label='MA20')
    plt.plot(DF['ma60'], label='MA60')
    plt.plot(DF['ma120'], label='MA120')
    plt.legend()
    plt.title('Moving Averages')
    plt.show()

    # RSI 그래프
    plt.plot(DF['rsi'])
    plt.title('RSI')
    plt.show()

    # MACD 그래프
    plt.plot(DF['macd'], label='MACD')
    plt.plot(DF['macdsignal'], label='Signal')
    plt.bar(DF.index, DF['macdhist'], label='Histogram')
