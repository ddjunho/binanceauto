nohup python3 binanceauto.py > output.log &

1.  이전 캔들의 고가가 상단 밴드를 돌파하고 현재 캔들이 하락한 경우 공매도
	if data['high'].iloc[-3] >= upper_band.iloc[-3]:
                if data['close'].iloc[-2] <= data['open'].iloc[-2]

2. 이전 캔들의 저가가 하단 밴드를 돌파하고 현재 캔들이 상승한 경우 매수
	if data['low'].iloc[-3] <= lower_band.iloc[-3]:
                if data['close'].iloc[-2] >= data['open'].iloc[-2]:

 3. 이익 신호: 20 EMA에 도달했을 때 포지션 종료 (익절).
 만약 볼린저밴드의 고가와 저가의 차이의 100배가 4 이상인 경우 9EMA에 도달했을 때 포지션 종료 (익절). 

4. 만약 공매도 혹은 매수 했을때 그 전의 2개의 볼륨 오슐레이터의 값이 40%이상인경우 반대의 포지션시작
volume_oscillator.iloc[-3]
volume_oscillator.iloc[-4]
