# 거래량이 폭증한 티커를 반환하는 함수 정의
def get_surge_tickers():
    tickers = client.get_all_tickers()
    df_tickers = pd.DataFrame(tickers)
    df_tickers = df_tickers[df_tickers['symbol'].str.contains('USDT')]
    ticker_list = df_tickers['symbol'].tolist()
    surge_list = []
    for ticker in ticker_list:
        candles = client.futures_klines(symbol=ticker, interval='4h', limit=7)
        df = pd.DataFrame(candles, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'])
        df['volume'] = df['volume'].astype(float)
        mean_volume = df['volume'].mean()
        max_volume = df['volume'].max()
        # 최대값이 평균값의 2배 이상이면 거래량이 폭증한 것으로 판단하고 surge_list에 추가하기
        if max_volume >= mean_volume * 2:
            surge_list.append(ticker)
    return surge_list
  
print(get_surge_tickers())
