
import csv 
import pandas as pd
import time
from pathlib import Path
import requests 
import ta 
from ta import add_all_ta_features
import os
import yfinance as yf

api_key = 'U89XP6FG3XT2EI2N'
his_time_interval = '30min'
live_time_interval = '30m' #needs to be different because yfinance requires different string input to alphavantage 

def get_ticker_list():

    ticker_list = []
    with open('C:/Users/mschulze/source/repos/shares_trading_reporting/ticker_data/ticker_list.csv', newline='') as csvfile:
        ticker_csv = csv.reader(csvfile, delimiter=' ')
        for row in ticker_csv:
            ticker_list.append(row[0])


    ticker_list_short = ticker_list[2:6]
    print(ticker_list_short)
    return ticker_list_short

def extract_prices(api_key, his_time_interval):

    df_ticker_csv = {}

    for ticker in list(get_ticker_list()):

        print('beginning extraction of ticker {}'.format(ticker))

        df = pd.DataFrame(columns = ['time','open','high','low','close','volume'])

        slice_interval = ['year1month1','year1month2']
        
        count = 0

        for i in slice_interval: 
            CSV_URL = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol={}&interval={}&slice={}&apikey={}'.format(
                ticker, his_time_interval, i, api_key)

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
            print('iteration status {}/{}'.format(count, len(slice_interval)))
        
        # format ticker df

        df = df.sort_values(by = 'time')
        df = df.set_index('time')

        #add to csv file
        filepath = Path('C:/Users/mschulze/source/repos/shares_trading_reporting/ticker_data/{}_raw_alphavantage.csv'.format(ticker))  
        filepath.parent.mkdir(parents=True, exist_ok=True)  
        df.to_csv(filepath)  
        
        #add to df and document success
        df_ticker_csv[ticker] = filepath
        print('successfully completed extraction of ticker {}'.format(ticker))
        print('ticker data stored in dataframe')
        print('[ticker symbol: {}, ticker file location: {}'.format(ticker, filepath))

    return df_ticker_csv

def process_data(df_ta, add_pct_change): #define process_data method that adds all the ta into a new df
    import pandas as pd
    # dropping all rows that do not have a timestamp (will happen if this code is run during a trading day instead of at the end)
    df_ta = df_ta.drop(df_ta.loc[df_ta['time'] == 'time'].index)

    #ensure columns are float datatypes

    df_ta['close'] = pd.to_numeric(df_ta['close'])
    df_ta['low'] = pd.to_numeric(df_ta['low'])
    df_ta['high'] = pd.to_numeric(df_ta['high'])
    df_ta['open'] = pd.to_numeric(df_ta['open'])
    df_ta['volume'] = pd.to_numeric(df_ta['volume'])

    # create SMA, BOLL BANDS columns
    df_ta['SMA30'] = df_ta['close'].rolling(30).mean()
    df_ta['SMA50'] = df_ta['close'].rolling(50).mean()
    df_ta['RSTD'] = df_ta['close'].rolling(30).std()
    df_ta['upper_band'] = df_ta['SMA50'] + 2 * df_ta['RSTD']
    df_ta['lower_band'] = df_ta['SMA50'] - 2 * df_ta['RSTD']

    # technical analysis: momentum indicators

    import pandas as pd
    import ta
    from ta import add_all_ta_features

    rsi_list = [12,14,16,18]

    for i in rsi_list:

        df_ta['RSI_{}'.format(i)] = ta.momentum.rsi(df_ta['close'], i)
        df_ta['sRSI_{}'.format(i)] = ta.momentum.stochrsi(df_ta['close'], i, 3, 3)
        df_ta['sOsc'] = ta.momentum.stoch(df_ta['close'], df_ta['high'], df_ta['low'], 14, 3)
        df_ta['TRIX_{}'.format(i)] = ta.trend.trix(df_ta['close'], i)

    macd_list = [24, 26, 28, 30]

    for i in macd_list:

        df_ta['MACD_{}'.format(i)] = ta.trend.macd(df_ta['close'], i, int(i/2), int(i/2.5))
        df_ta['MI_{}'.format(i)] = ta.trend.mass_index(df_ta['close'], i, int(i/2.5))


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
            if max(pct_change) >= 0.05:
                signal.append('BUY')
            elif min(pct_change) <= -0.03:
                signal.append('SELL')
            else:
                signal.append('HOLD')

        df_ta['signal'] = signal
    
    elif add_pct_change == False:
        
        pass

    
    return df_ta

def technical_indicators(): #run at the end of every trading day. remove iterator later, it is not needed

    import pandas as pd
    import csv

    path = 'C:/Users/mschulze/source/repos/shares_trading_reporting/ticker_data/'
    dir_list = os.listdir(path)
    dir_list.remove('ticker_list.csv')
    
    # loop through all files in dir_list and publish output of process_data method to csv(filepath)
    for file in dir_list:
        df_ta = pd.read_csv(path + file)
        filepath = Path('C:/Users/mschulze/source/repos/shares_trading_reporting/ticker_data/ticker_ta_data/{}_ta_data.csv'.format(file.split('_')[0]))  
        filepath.parent.mkdir(parents=True, exist_ok=True)  
        process_data(df_ta, True).to_csv(filepath)  
        
def daily_update(live_time_interval):

    import pandas as pd
    import yfinance as yf
    from datetime import datetime


    #define empty df with correct columns:

    with open('C:/Users/mschulze/source/repos/shares_trading_reporting/ticker_data/ticker_ta_data/AAPL_ta_data.csv') as csv_file:
 
        # reading the csv file using DictReader
        csv_reader = csv.DictReader(csv_file)
    
        # converting the file to dictionary
        # by first converting to list
        # and then converting the list to dict
        dict_from_csv = dict(list(csv_reader)[0])
    
        # making a list from the keys of the dict
        columns = list(dict_from_csv.keys())
    
        # displaying the list of column names

    
    df_reporting = pd.DataFrame(columns = columns)

    pct_change_lst = list(range(5,30))
    delete_str = []
    for pct in pct_change_lst:
        delete_str.append('pct_change_{}'.format(pct))
    delete_str = delete_str + ['signal', 'time','']
    df_reporting = df_reporting.drop(delete_str, axis = 1)

    path = 'C:/Users/mschulze/source/repos/shares_trading_reporting/ticker_data/'
    dir_list = os.listdir(path)
    dir_list.remove('ticker_list.csv')
    dir_list.remove('ticker_ta_data')

    # loop through all files in dir_list and publish output of process_data method to csv(filepath)

    for file in dir_list:

        ticker = file.split('_')[0] #extract ticker name from file name
        
        data = yf.Ticker(ticker).history(period = '1d', interval = live_time_interval) # get yfinance data

        #data = data.drop(['Dividends','Stock Splits'])
        
        live_price = data.iloc[-1] # get last data point (most recent)
        live_price = live_price.to_frame()
        
        live_price_T = live_price.T   
        live_price_T = live_price_T.drop(['Dividends','Stock Splits'], axis = 1)
        live_price_T = live_price_T.reset_index()

        live_price_T = live_price_T.rename(columns = {'index': 'time', 'Open':'open', 'High':'high', 'Close':'close', 'Volume':'volume', 'Low':'low'})
        live_price_T.at[0, 'time'] = 0


        df = pd.read_csv(path + file) #extract csv from ticker_data and name as df
        df = df.tail(80) #only take last 50 data points
        
        df_ta = df.append(live_price_T)

        df = process_data(df_ta, False) #define df_recent as the processed version of df_ta
        df_update = df.tail(1)
        df_update = df_update.drop(['time'], axis = 1)
        df_update['ticker'] = [ticker]
        df_update = df_update.set_index('ticker')
        
        df_reporting = df_reporting.append(df_update)

    return df_reporting
        
def sector_analysis():

    import pandas as pd
    import matplotlib_inline
    import matplotlib.pyplot as plt
    import numpy as np
    
    """
    to print information on groups of stocks (can also just be for one individual stock)
    - movement over the day, week and month
    - graphing for different periods
    - overview of technical indicators (also print what they mean - whether they are leaning towards buy or sell etc.)
    """

    path = 'C:/Users/mschulze/source/repos/shares_trading_reporting/ticker_data/'
    dir_list = os.listdir(path)
    dir_list.remove('ticker_list.csv')
    dir_list.remove('ticker_ta_data')

    
    dir_ticker_list = []
    for file in dir_list:
        dir_ticker_list.append(format(file.split('_')[0]))
    
    df_data = pd.DataFrame(columns = dir_ticker_list)
    for file in dir_list:
        
        df = pd.read_csv('C:/Users/mschulze/source/repos/shares_trading_reporting/ticker_data/{}'.format(file))
        df = df.drop(df.loc[df['time'] == 'time'].index)
        df = df.dropna()
        df_data[format(file.split('_')[0])] = df['close']

    df_data = df_data.astype(float)
    normalized_df=(df_data-df_data.min())/(df_data.max()-df_data.min()) 
    df_corr = normalized_df.corr(method = 'pearson')
    df_corr_procc = df_corr[df_corr!= 1]
    df_corr_procc = df_corr_procc.fillna(0)
    print(normalized_df.head())
    for ticker in list(normalized_df.columns):
        normalized_df[ticker] = normalized_df[ticker].fillna(normalized_df[ticker].mean())  

        # finding stocks of similar correlation - 

        df_highcorr = df_corr_procc[ticker][df_corr_procc[ticker] > 0.5]

        #plot stocks of similar correlation
        plot_list = []
        plot_list.append(ticker)
        plot_list = plot_list + list(df_highcorr.index)

        normalized_df[plot_list].head(200).plot()
        print(normalized_df[plot_list].head(10))
        # plt.show()

        
        # print('{} has high correlation with the following: '.format(ticker))
        # print('tickers: {}'.format(list(df_highcorr.index)))
        # print(df_highcorr)

    #cross correlation to estimate the similarity of two stocks movement:

    # sig1 = normalized_df['AAPL'].to_numpy()
    
    # sig2 = normalized_df['MSFT'].to_numpy()
    # print(sig1, sig2)

    # corr = np.correlate(a=sig1, v=sig2)

    # print(corr)



#AAPL _ MSFT
    
    normalized_df['MSFT'] = normalized_df['MSFT'].fillna(normalized_df['MSFT'].mean())
    normalized_df['MU'] = normalized_df['MU'].fillna(normalized_df['MU'].mean())

    # print(normalized_df['MSFT'].isnull().sum())
    # print(normalized_df['MU'].isnull().sum())

    # x1 = normalized_df['AAPL'].to_numpy()
    # y1 = normalized_df['MSFT'].to_numpy()
    # y2 = normalized_df['MU'].to_numpy()
    # m1, b1 = np.polyfit(x1, y1, 1)
    # m2, b2 = np.polyfit(x1, y2, 1)
    # print(m2, b2)

    # plt.figure(figsize=(12,4))
    # ax = plt.axes()
    # ax.scatter(x = x1, y = y1, c = 'green', label = 'AAPL - MSFT')
    # ax.plot(x1, m1*x1 + b1, c = 'green')
    # ax.scatter(x = x1, y = y2, c = 'blue', label = 'AAPL - MU')
    # ax.plot(x1, m2*x1 + b2, c = 'blue')

    # plt.legend()
    # plt.show()

    import seaborn as sns

    sns.set_style('whitegrid')
    sns.lmplot(x = 'MU', y = 'SIRI', data = normalized_df)
    #plt.show()


sector_analysis()
                

# extract_prices(api_key, his_time_interval)
#technical_indicators()
#daily_update(live_time_interval)
# %%
