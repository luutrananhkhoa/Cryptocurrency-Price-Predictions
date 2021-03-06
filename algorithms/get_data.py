import yfinance as yf
import pandas as pd

pd.options.mode.chained_assignment = None


def crypto_data(ticker, period, intervals):
    df = yf.download(ticker, period=period, interval=intervals)
    i = 0
    while i < 12:
        last_date = df.iloc[[-1]].index + pd.Timedelta(hours=1) 
        df = df.append(pd.DataFrame(index=[last_date[0]]))
        i += 1
    df = df.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'])
    df.index.name = 'Datetime'
    df_pred = df
    return df,df_pred
