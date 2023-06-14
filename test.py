
import pickle
import pandas as pd
from database_conn import Database_conn
from ML_pipeline import ML_pipeline


with open('ml_model.pkl', 'rb') as file:
    ML_objects = pickle.load(file)
df_ML_res = pd.read_csv('ml_results.csv')
ML_results = df_ML_res.to_dict(orient='list')
ML_results = {'ticker': ML_results['ticker'], 'model': ML_results['model'], 'accuracy': ML_results['accuracy']}
print(ML_results)
db = Database_conn()

df_tickers = db.ticker_list()
ticker_list = list(df_tickers['TICKER'])
ticker_data = {}
for ticker in ticker_list:
    ticker_data[ticker] = db.sql_to_df('"L1_{}"'.format(ticker))
    
# ticker_bronze = {}    
# for ticker in ticker_list:
#     ticker_bronze[ticker] = db.sql_to_df('"L0_{}"'.format(ticker))
    
ticker_ML_object = ML_objects['AMZN']
df = ticker_data['AMZN']
y_pred = ML_pipeline(df, ticker_ML_object)

print(y_pred, type(y_pred))

df['signal'] = y_pred
print(df)