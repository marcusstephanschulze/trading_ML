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

def extract_prices(api_key, his_time_interval, ticker):

    print('beginning extraction of ticker {}'.format(ticker))

    df = pd.DataFrame(columns = ['time','open','high','low','close','volume'])

    his_window = [i for i in range(1,13)]
    slice_interval_1 = ['year1month{}'.format(i) for i in his_window]
    slice_interval_2 = ['year2month{}'.format(i) for i in his_window]
    slice_interval = slice_interval_1 + slice_interval_2
    count = 0
    #print('extraction for time window: {}'.format(slice_interval[count]))
    
    for i in slice_interval: 
        CSV_URL = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol={}&interval={}&slice={}&apikey={}'.format(
            ticker, his_time_interval, i, api_key)
        print('---> extraction for time window: {}'.format(slice_interval[count]))
        with requests.Session() as s:
            download = s.get(CSV_URL)
            decoded_content = download.content.decode('utf-8')
            cr = csv.reader(decoded_content.splitlines(), delimiter=',')
            my_list = list(cr)
            
            #add each row of "my_list" to the df created above (new df created for each loop)

            for row in my_list:
                df_length = len(df)
                df.loc[df_length] = row

            time.sleep(15) # alphavantage has max num requests per minute - the delay makes sure it circumvents errors. 
        count += 1
        print('---> iteration status {}/{}'.format(count, len(slice_interval)))
    
    # format ticker df
    df = df.drop(df.loc[df['time'] == 'time'].index)
    df = df.sort_values(by = 'time')
    df = df.set_index('time')
    df = df.reset_index()
    df['id'] = df.index
    
    df = df.reset_index(drop=True)
    df = df.rename(columns={'index': 'id'})
    
    df['timestamp'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df = df.drop(columns=['time'])
    #df = df.set_index('timestamp').between_time('9:30', '16:00').reset_index()
    
    
    # df = df.rename(columns={'time': 'timestamp'})
    df = df.drop(index=df.index[-2:])
    cols = ['id'] + ['timestamp'] + [col for col in df.columns if col != 'id' and col!= 'timestamp']

    # create a new dataframe with columns in the desired order
    df = df.reindex(columns=cols)
    #strip df of nulls (look into this more!)
    
    df = df.dropna()
    #write df to database "l0_ticker1"
    
    db = Database_conn()
    
    table_name = "L0_{}".format(ticker)
    print('extracting data and inserting into {}'.format(table_name))
    db.write_to_db(df, table_name)
    
    #add to df and document success
    print('----> successfully completed extraction of ticker {}'.format(ticker))
    print('----> ticker data stored in table L0_{}'.format(ticker))

    print(df)
    
    
def delta_extract(api_key, his_time_interval, ticker):
    
    print('beginning extraction of ticker {}'.format(ticker))

    #df = pd.DataFrame(columns = ['time','open','high','low','close','volume'])

    CSV_URL = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={}&interval={}&apikey={}'.format(ticker, his_time_interval, api_key)

    print('---> delta extraction for recent data')
    r = requests.get(CSV_URL)
    data = r.json()
    df = pd.DataFrame(data = data['Time Series (30min)'])
    df = df.transpose()
    df = df.sort_index()
    df = df.reset_index()
    df = df.rename(columns = {'index': 'time', '1. open': 'open', '2. high': 'high','3. low': 'low', '4. close': 'close', '5. volume': 'volume'})
    df['id'] = df.index
    
    

    df['timestamp'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df['timestamp'] = df['timestamp'].astype('datetime64')
    df = df.drop('time', axis = 1)
    df = df.reindex(columns = ['id','timestamp','open','high','low','close','volume'])
    #strip df of nulls (look into this more!)
    
    df = df.dropna()
    #write df to database "l0_ticker1"
    
    return df
    