import psycopg2
import pandas as pd
from psycopg2 import sql

from dataset_extraction import extract_prices
from database_conn import Database_conn
from process_data import process_data
from ML_pipeline import ML_training
import warnings
warnings.filterwarnings("ignore")


"""
    notes: 
    
    1) need to change DB structure to L0_ticker1 etc. with open, high, low, close
    2) extract_prices needs to be changed so it writes each extracted value to the relevant database
                    thus does not need to return df
"""

api_key = 'U89XP6FG3XT2EI2N'
his_time_interval = '30min'

ticker_list = [
               'AMZN',
                'GOOG', 
            #    'BABA', 
            #    'MSFT', 
            #    'AAPL'
               ]

db = Database_conn()

# --------------------------------------

#step 1) also sets up the ticker_list

def init_database():
    
    db.init_db_L0(ticker_list) 


#step 2) extract data into bronze layer

def extract_bronze(df_tickers):

    for ticker in list(df_tickers['TICKER']):

        df_extract = extract_prices(api_key, his_time_interval, ticker) #maybe change method to loop through


#step 3) process bronze layer to silver layer - need to create the silver layer tables first

def process_bronze():
    for ticker in list(df_tickers['TICKER']):
        table_in = '"L0_{}"'.format(ticker) # sql_to_df() requires the string input to be in a different format (need to look into changing)
        table_out = 'L1_{}'.format(ticker)
        df = process_data(table_in)
        db.init_db_L1(df, ticker)
        db.write_to_db(df, table_out)
    
    
#step 4) perform ML training on silver layer

def perform_ML():
    
    ML_results = pd.DataFrame(columns=['ticker', 'model','accuracy','cm','validation cm'])
    ML_models = {}

    for ticker in list(df_tickers['TICKER']):
        
        table_in = '"L1_{}"'.format(ticker)
        
        results = ML_training(table_in, val=True)
        
        # Create a dictionary with the results and append it to the ML_results dataframe
        result_dict = {'ticker': ticker, 'model': 'RandomForest', 'accuracy': results[1], 'cm': results[2], 
                        'validation cm': results[3]}
        
        ML_results = ML_results.append(result_dict, ignore_index=True)
        ML_models[ticker] = results[0]
        
    print(ML_results, ML_models)
    return ML_results, ML_models

# step 5) extract delta load

def execute_delta(ticker):
    
        
    from dataset_extraction import delta_extract
    
    df_delta = delta_extract(api_key, his_time_interval, ticker)
    
    
    return df_delta

#step 6) process delta load and write to delta layer (L2)

def process_delta(): #only need to run this one and do not need to run execute_delta() -> it is referenced in this method
    
    for ticker in list(df_tickers['TICKER']):
        table_out = 'L2_{}'.format(ticker)


        from dataset_extraction import delta_extract
        delta_df = delta_extract(api_key, his_time_interval, ticker)
        
        #delta_df = execute_delta(ticker) # because of this - > only process_delta method is needed to be called! 
        df = process_data(delta_df, delta = True, add_pct_change=False) # do not want the pct changes to be included in the data processing!
        print('delta dataframe: ', df)

        if db.table_exists(table_out)==True:
            # there will be data in delta layer (L2), so need to only add data AFTER most recent timestamp of the data
            
            recent_timestamp = db.get_recent_timestamp('L2', ticker) # add functionality to select which layer -> select L2
            print('recent_timestamp: ', recent_timestamp)
            filtered_df = df[df['timestamp'] > recent_timestamp]
            db.append_to_table(filtered_df, table_out) 
            
        else:
            # accessing L1 processed data to to get last timestamp and stitch L2 delta data with L1 (last 100)
             
        
            table = '"L1_{}"'.format(ticker) 
            df_L1 = db.sql_to_df(table)
            df_L1 = df_L1.tail(100)
            
            # delete all pct_change columns
            
            df_L1 = df_L1.loc[:, ~df_L1.columns.str.contains('pct_change')]
            df_L1 = df_L1.drop(['signal'],axis = 1)
            
            # unlike the if part - this is stitching all data BEFORE onto the L2 data - so it needs all timestamps LESS THAN recent_timestamp
            
            recent_timestamp = db.get_recent_timestamp('L1', ticker)
            print('recent_timestamp: ', recent_timestamp)
            filtered_df = df_L1[df_L1['timestamp'] < recent_timestamp]            
            

            print(df_L1)
            df_rewrite = df_L1.append(filtered_df, ignore_index = True)
            df_rewrite = df_rewrite.drop(['id'],axis = 1)
            df_rewrite = df_rewrite.rename_axis('id').reset_index()        
            print(df_rewrite)

            breakpoint()
            db.init_db_L2(df, ticker) #only need to initiate it the first time... 
            db.write_to_db(df_rewrite, table_out) # first time is write_to_db, second time needs to append
        

#step 5) save parameter values and use to predict new values coming in through delta load

def predict_data(): #used to run ML training method if it has not been run before and to predict new delta load data
    
    print('initiating predicting data')
    import os
    import pickle

    # Check if the ML model already exists
    if os.path.isfile('ml_model.pkl') and os.path.isfile('ml_results.csv'):
        # Load the pre-trained ML model
        with open('ml_model.pkl', 'rb') as file:
            ML_objects = pickle.load(file)
        df_ML_res = pd.read_csv('ml_results.csv')
        ML_results = df_ML_res.to_dict(orient='list')
        ML_results = [{'ticker': ML_results['ticker'], 'model': ML_results['model'], 'accuracy': ML_results['accuracy']}]
    else:
        # Perform the ML training in case it
        ML_out = perform_ML()
        ML_results = ML_out[0]
        ML_objects = ML_out[1]  
        
        # Convert the ML results DataFrame to a dictionary
        ML_results_dict = ML_results.to_dict(orient='list')
        
        # Save the trained ML model to disk
        with open('ml_model.pkl', 'wb') as file:
            pickle.dump(ML_objects, file)
        
        # Save the ML results dictionary to a CSV file
        df_ML_results = pd.DataFrame(ML_results_dict)
        df_ML_results.to_csv('ml_results.csv', index=False)
        
    db = Database_conn()

    df_tickers = db.ticker_list()
    ticker_list = list(df_tickers['TICKER'])
    
    ticker_data = {}
    for ticker in ticker_list:
        data = db.sql_to_df('"L2_{}"'.format(ticker))   
        from ML_pipeline import ML_pipeline
        y_pred = ML_pipeline(data, ML_objects[ticker])
        y_pred[-3] = 2
        data['signal'] = y_pred
        ticker_data[ticker] = data
        
    #write this data to L3
    
    return ticker_data
        

#step 6) use output to run dash_plot

def run_plotting(ML_results, ML_objects, ticker_list, ticker_data):
    
        
    # ticker_bronze = {}    
    # for ticker in ticker_list:
    #     ticker_bronze[ticker] = db.sql_to_df('"L0_{}"'.format(ticker))
        
    # example_ticker = ticker_data['AMZN']
    # min_date = example_ticker['timestamp'].min()
    # max_date = example_ticker['timestamp'].max()

    from graphing import dash_plot

    app = dash_plot(ML_results, ML_objects, ticker_list, ticker_data)
    

#init_database()
df_tickers = db.ticker_list()
#extract_bronze(df_tickers)
# process_bronze()
process_delta() ## this also calls the execute_delta method
#data = predict_data()

#print(data)
print('finished')

#run_plotting()

#execute_delta()
# need to rerun process_bronze() after executing delta
     

db.close_conn()

