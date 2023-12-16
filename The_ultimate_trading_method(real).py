import pandas as pd
import numpy as np
import random as rand
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
import ccxt
from binance_keys import api_key, api_secret
################ 백테스트 ################

# 백테스트 기본 셋팅
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
timeframe = '5m'# 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M

# 데이터 읽어오기: 전체 데이터
candles = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=None,
                limit=500
            )
        
df = pd.DataFrame(data=candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            

# 변동성 돌파 전략 위한 로직 설정

def VBS(df, config_data):
    # k값 넣기
    k = config_data['k']

    # 목표가 구하기
    df['range'] = df['high'] - df['low']  # 고저 변동폭
    df['target'] = df['open'] + df['range'].shift(1) * k  # 목표가 한칸 내려주고 (shift), 이후 k값을 곱한 목표가 계산
    # df = df.iloc[1:-2]

    # 매수 시뮬레이션
    df['ror'] = np.where(df['high'] > df['target'], df['close'] / df['target'],
                         1)  # high > target인 지점에서 close / target

    # 최종 누적 산출
    df['total'] = df['ror'].cumprod()
    final_cum_ror = (df['total'].iloc[-1].astype(float)) * 100 -100

    # 연간 수익률 (기간 수익률)
    N = ((df.index[-1] - df.index[0])) / 365
    CAGR = (final_cum_ror ** (1 / N))

    # dd값 기록 및 mdd 계산
    array_v = np.array(df['total'])
    dd_list = -(np.maximum.accumulate(array_v) - array_v) / np.maximum.accumulate(array_v)
    peak_lower = np.argmax(np.maximum.accumulate(array_v) - array_v)
        # np.maximum.accumulate(array_v)는 최고값을 계속 갱신한 array
        # 그 array에서 원래 array를 뺀 값들의 array를 새로 생성
        # 뺀 값만 남은 array 중 가장 큰 값의 index를 추출
    peak_upper = np.argmax(array_v[:peak_lower])
        # peak_lower의 index값까지의 array_v값 추출
        # 그 중에 가장 큰 놈 (MDD 계산을 위한 최고값) 의 index 추출
    mdd = round((array_v[peak_lower] - array_v[peak_upper]) / array_v[peak_upper] * 100, 3)

    return final_cum_ror, CAGR, mdd, dd_list,df['total'] # 최종 수익률, 연간 수익률, 최대손실폭 (mdd), 손실 목록, 누적 이익 목록

################ 최적화 ################

### 베이지안 최적화

# 베이지안 최적화 위한 함수
def black_box_function (k):
    config_data = {
        'k': k,
    }
    revenue = VBS(df,config_data)
    return revenue[0]

# parameter k의 범위
pbounds = {
    'k': (0.20,1.0)
}

# 베이지안 최적화 시행
optimizer = BayesianOptimization(f=black_box_function,pbounds=pbounds,random_state=1)
optimizer.maximize(init_points=5, n_iter=100)

# 결과값 저장
target_list = []
i=0

for res in optimizer.res: # res는 최적화 결과값을 리스트로 저장한 것.
    print(res)
    target_list.append([res["target"], i+1,res['params']]) # target값을 append
    i=i+1

target_list.sort(reverse=True)

target_list[0][-1].get('k')
# 값 추출

max_beyes_k = target_list[0][-1].get('k')

def VBS_opt(df, k):

    # 목표가 구하기
    df['range'] = df['high'] - df['low']  # 고저 변동폭
    df['target'] = df['open'] + df['range'].shift(1) * k  # 목표가 한칸 내려주고 (shift), 이후 k값을 곱한 목표가 계산
    # df = df.iloc[1:-2]

    # 매수 시뮬레이션
    df['ror'] = np.where(df['high'] > df['target'], df['close'] / df['target'],
                         1)  # high > target인 지점에서 close / target

    # 최종 누적 산출
    df['total'] = df['ror'].cumprod()
    final_cum_ror = (df['total'].iloc[-1].astype(float)) * 100 -100

    # 연간 수익률 (기간 수익률)
    N = ((df.index[-1] - df.index[0])) / 365
    CAGR = (final_cum_ror ** (1 / N))

    # dd값 기록 및 mdd 계산
    array_v = np.array(df['total'])
    dd_list = -(np.maximum.accumulate(array_v) - array_v) / np.maximum.accumulate(array_v)
    peak_lower = np.argmax(np.maximum.accumulate(array_v) - array_v)
        # np.maximum.accumulate(array_v)는 최고값을 계속 갱신한 array
        # 그 array에서 원래 array를 뺀 값들의 array를 새로 생성
        # 뺀 값만 남은 array 중 가장 큰 값의 index를 추출
    peak_upper = np.argmax(array_v[:peak_lower])
        # peak_lower의 index값까지의 array_v값 추출
        # 그 중에 가장 큰 놈 (MDD 계산을 위한 최고값) 의 index 추출
    mdd = round((array_v[peak_lower] - array_v[peak_upper]) / array_v[peak_upper] * 100, 3)

    return final_cum_ror, CAGR, mdd, dd_list,df['total'] # 최종 수익률, 연간 수익률, 최대손실폭 (mdd), 손실 목록, 누적 이익 목록

final_result = VBS_opt(df,max_beyes_k)

print("최종 누적 수익률: ", final_result[0] ,"%") # 최종 누적 수익률 도출
print("MDD: ", final_result[2], "%") # mdd 도출
print("최선의 K값: ",max_beyes_k)
