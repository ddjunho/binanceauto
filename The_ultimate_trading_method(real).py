import pandas as pd
import numpy as np
import random as rand
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
백테스트
# 백 테스트 기본 셋팅
symbol = "BTCUSDT" #
BAS
interval = '1d' # 1m, 3m, 5m, 15m, 30m, th, 2h, 4h. Oh, 8h. 12h. 1d, 3d, tw. Th year = 'All' # 2017~2022 중 선택 가능.
# Ell Ol Et 2104971: El Tall Ell Ol Et
df = pd.read_csv (f'C:.##Data##{symbol [:-4]}-USDT_Data##{symbol [:-4]}- USDT_Data_{interval}##{symbol [:-4]. upper()}_USDT_{interval}_{year}.csv')
# Za Fall : _Al/
# 기간: 연도 불러오면 됨
df = pd.DataFrame(df)
df = df.dropna(how='any')
# 변동성 들파 전략 위한 로직 설정
def VBS(df, config_data):
# K값 넣기
k=config_data['k']
# 목표가 구하기
df ['range'] = df['High'] - df['Low'] # 255
df ['target'] = df ['Open'] + df ['range'].shift(1) + k  # dt=df.iloc[1:-2]
# 매수 시뮬레이션
df ['ror'] = np.where(df ['High']> df ['target'],1) #high>
# 최종 누적 산출
df ['Close'] / df['target'],
1) #high> target 2) CHI H close / target
df ['total'] = df ['ror'].cumprod()
final_cum_ror = (df['total'].iloc[-1].astype(float)) 100 -100
# 연간 수익률(기간 수익률)
N = ((df.index[-1] - df.index[0])) / 365
CAGR (final_cum_ror ** (1 / N))
=
# 아이값 기록 및 mdd 계산
array_v = np.array(df ['total'])
dd_list(np. maximum.accumulate(array_v) array_v) / np.maximum.accumulate(array_v)

peak_lower = np.argmax(np.maximum.accumulate(array_v)
array_v)
#np.maximum.accumulate(array_v) — 72 #÷ 202 array
# 그 array 에서 원래 array를 뺀 값들의 array를 새로 생성
# 뺀 값만 남은 array 중 가장 큰 값의 index를 추출
peak_upper = np.argmax(array_v[:peak_lower])
# peak lower 의 index 값까지의 array_v 값 추출
# 그 중에 가장 큰 놈 (HDD 계산을 위한 최고값) 의 index 추출
mdd = round((array_v [peak_lower]
100, 3)
-
array_v[peak_upper]) / array_v[peak_upper]
+
return final_cum_ror, CAGR, mdd, dd_list,df['total'] # § 4o1⁄2. 92 ÷€. 최대손실폭 (mdd), 손실 목록, 누적 이익 목록
최적화
베이지만 최적화
# 베이지만 최적화 위한 함수
def black_box_function (k):
#
config_data = {
}
'k': k,
revenue VBS(df, config_data)
return revenue[0]
parameter k의 범위
pbounds = {
'k' (0.20,1.0)
# 베이지만 최적화 시행
optimizer BayesianOptimization(f=black_box_function, pbounds=pbounds, random_state=1)
optimizer.maximize(init_points=5, n_iter=100)
# 결과값 저장
target_list = []
i=0
for res in optimizer.res: # rost $A$ 2775 NZ^EE HDD X.
print(res)
target_list.append( [res["target"], i+1, res['params']]) # target append
i=i+1
target_list.sort(reverse=True)
target_list [0] [-1].get('k').
# 값 추출
max_beyes_k= target_list [0] [-1].get('k').
﻿

def VBS_opt(df, k):
# 목표가 구하기
df ['range'] = df ['High'] - df ['Low'] # DH df ['target'] = df ['Open'] + df['range'].shift(1) (Ehift), 이후 K값을 곱한 목표가 계산
# df = df.iloc[1:-2]
# 매수 시뮬레이션
df ['ror'] = np.where(df ['High']> df ['target'], 1) # high > target!
# 최종 누적 산출
df ['total'] = df ['ror'].cumprod()
+
k # 목표가 한칸 내려주고
df ['Close'] / df['target'], A close / target
final_cum_ror = (df['total'].iloc[-1].astype(float)) + 100 -100
# 연간 수익률 (기간 수익률)
N = ((df.index[-1] - df. index[0])) / 365
CAGR (final_cum_ror ** (1 / N))
=
# 아이값 기록 및 지미아 계산
array_v= np.array(df ['total'])
dd_list = (np.maximum.accumulate(array_v) - array_v) / np.maximum, accumulate(array_v)
peak_lower = np.argmax(np.maximum.accumulate(array_v) - array_v) #np.maximum.accumulate(array_v) — #7
H÷2NE array
# 그 array 에서 원래 array를 뺀 값들의 array를 새로 생성
# 뺀 값만 남은 array 중 가장 큰 값의 index를 추출
peak_upper = np.argmax(array_v[:peak_lower])
# peak lower 의 index 값까지의 array_v값 추출
# 그 중에 가장 큰 놈 (MDD 계산을 위한 최고값) 의 index 추출
mdd = round((array_v[peak_lower] - array_v[peak_upper]) / array_v[peak_upper] =
100, 3)
+
return final_cum_ror, CAGR, mdd, dd_list,df['total'] # # +2.92 €. 최대손실폭 (mdd), 손실 목록, 누적 이익 목록
final_result = VBS_opt (df, max_beyes_k)
print("최종 누적 수익률:
final_result[0],"%") # # + ÷ 5±
print("MDD: ", final_result[2], "%") # mdd 5
print(" Kat: ",max_beyes_k)
