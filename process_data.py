import csv 
import pandas as pd
import time
from pathlib import Path
import requests 
import ta 
from ta import add_all_ta_features
import os
import yfinance as yf
from database_conn import Database_conn
from sqlalchemy import create_engine
import datetime

# api_key = 'U89XP6FG3XT2EI2N'
# his_time_interval = '30min'

db = Database_conn()

#needs to happen for all columns of l0_intraday independently. Will create "silver layer": "L1_AMZN" and "L1_GOOG" etc... :

def process_data(table_in, delta = False, add_pct_change = True): #define process_data method that adds all the ta into a new df

    if delta == False:
    
        df_ta = db.sql_to_df(table_in)
        
        print('processing table {} to silver'.format(table_in))

    elif delta == True: 
        
        df_ta = table_in
    
    import pandas as pd
    # dropping all rows that do not have a timestamp (will happen if this code is run during a trading day instead of at the end)
    df_ta = df_ta.drop(df_ta.loc[df_ta['timestamp'] == 'time'].index)

    mask = (df_ta['timestamp'].dt.time >= datetime.time(9, 30)) & (df_ta['timestamp'].dt.time <= datetime.time(16, 0))
    df_ta = df_ta.loc[mask]
    
    #ensure columns are float datatypes

    df_ta['close'] = pd.to_numeric(df_ta['close'])
    df_ta['low'] = pd.to_numeric(df_ta['low'])
    df_ta['high'] = pd.to_numeric(df_ta['high'])
    df_ta['open'] = pd.to_numeric(df_ta['open'])
    df_ta['volume'] = pd.to_numeric(df_ta['volume'])

    # create SMA, BOLL BANDS columns
    df_ta['sma30'] = df_ta['close'].rolling(30).mean()
    df_ta['sma50'] = df_ta['close'].rolling(50).mean()
    df_ta['rstd'] = df_ta['close'].rolling(30).std()
    df_ta['upper_band'] = df_ta['sma50'] + 2 * df_ta['rstd']
    df_ta['lower_band'] = df_ta['sma50'] - 2 * df_ta['rstd']

    # technical analysis: momentum indicators

    import pandas as pd
    import ta
    from ta import add_all_ta_features

    rsi_list = [12,14,16,18]

    for i in rsi_list:

        df_ta['rsi_{}'.format(i)] = ta.momentum.rsi(df_ta['close'], i)
        df_ta['srsi_{}'.format(i)] = ta.momentum.stochrsi(df_ta['close'], i, 3, 3)
        df_ta['sosc'] = ta.momentum.stoch(df_ta['close'], df_ta['high'], df_ta['low'], 14, 3)
        df_ta['trix_{}'.format(i)] = ta.trend.trix(df_ta['close'], i)

    macd_list = [24, 26, 28, 30]

    for i in macd_list:

        df_ta['macd_{}'.format(i)] = ta.trend.macd(df_ta['close'], i, int(i/2), int(i/2.5))
        df_ta['mi_{}'.format(i)] = ta.trend.mass_index(df_ta['close'], i, int(i/2.5))


    #create percent change variables for range of periods in advance to check whether the current stock price is a buy or sell etc.

    if add_pct_change == True:
    
        pct_change_lst = list(range(5,30))

        for pct in pct_change_lst: 

            df_ta['pct_change_{}'.format(pct)] = df_ta['close'].pct_change(periods = pct)
            df_ta['pct_change_{}'.format(pct)] = df_ta['pct_change_{}'.format(pct)].shift(-pct)

        df_ta.dropna(inplace = True) 

        # currently the BUY/SELL window is 5-30 periods and 2%. Change to 7%/5% BUY/SELL ratio eventually. 
        # for future reference - can do "weak buy, medium buy, strong buy" and have different % growth categories. 
        signal = []

        for row_series in df_ta.iterrows():
            pct_change = list(row_series[1][-(len(pct_change_lst)):-1]) #looping through each column element of a single row for a single ticker symbol. 
            if max(pct_change) >= 0.07:
                signal.append(2)
            elif min(pct_change) <= -0.03:
                signal.append(0)
            else:
                signal.append(1)

        df_ta['signal'] = signal
    
    elif add_pct_change == False:
        
        pass

    print('--> processing table {} to silver completed'.format(table_in))

    print('process_data(): ', df_ta)

    return df_ta
