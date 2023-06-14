import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from database_conn import Database_conn
db = Database_conn()
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

def ML_training(table_in, val = True):

    print('initiating ML training for {}'.format(table_in))
    df_ta = db.sql_to_df(table_in)
    
    if val == True:
        df_val = df_ta.head(500).copy()
        df_ta = df_ta.iloc[500:]
        df_val = df_val.drop('id',axis = 1)
        df_val = df_val.drop('timestamp',axis = 1)
        df_val = df_val.loc[:, ~df_val.columns.str.contains('pct_change')]
        df_val_features = df_val.drop('signal', axis = 1)

        # Create a StandardScaler object
        scaler = StandardScaler()

        # Apply the scaler to the entire DataFrame
        df_val_features = scaler.fit_transform(df_val_features)
        
        X_val, y_val,  = df_val_features, df_val['signal'],  # target variable
    else: 
        pass
    

    df_ta = df_ta.drop('id',axis = 1)
    df_ta = df_ta.drop('timestamp',axis = 1)
    df_ta = df_ta.loc[:, ~df_ta.columns.str.contains('pct_change')]
    df_ta_features = df_ta.drop('signal', axis = 1)
    
    # Create a StandardScaler object
    scaler = StandardScaler()

    # Apply the scaler to the entire DataFrame
    df_ta_features = scaler.fit_transform(df_ta_features)
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        df_ta_features,  # features
        df_ta['signal'],  # target variable
        test_size=0.2,
        random_state=42,
        )   

    # Create a random forest classifier object
    rfc = RandomForestClassifier()

    # Define the hyperparameters to search over
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    print('--> beginning ML fitting')
    # Create a GridSearchCV object
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=cv)

    # Fit the GridSearchCV object to the training data
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters
    print('--> Best hyperparameters:', grid_search.best_params_)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Fit the best model to the training data
    best_model.fit(X_train, y_train)
    
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    
    # Get the predicted classes for the test data
    y_pred = best_model.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred, average = 'weighted')
    # Print the accuracy of the model
    print('--> Model Accuracy:', accuracy)
    print('--> Model f1 score:', f1score)

    # Get the predicted classes for the test data
    # y_pred = best_model.predict(X_test)

    # Create the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # validation data set ------------------
    
    if val == True:
        # Get the predicted classes for the test data
        y_val_pred = best_model.predict(X_val)

        # Create the confusion matrix
        cm_val = confusion_matrix(y_val, y_val_pred)
        
        return [best_model, accuracy, cm, cm_val]
    
    else: 
        return [best_model, accuracy, cm]
    
    
def ML_pipeline(df, ml_object):
    
    print('initiating ML predict for new data')
    
    df = df.drop('id',axis = 1)
    df = df.drop('timestamp',axis = 1)
    df_features = df.drop('signal', axis = 1)
    
    # Create a StandardScaler object
    scaler = StandardScaler()

    # Apply the scaler to the entire DataFrame
    df_features = scaler.fit_transform(df_features)
    
    y_pred = ml_object.predict(df_features)
    
    return y_pred
    
def ML_plotting(cm, label):

    # Create a heatmap for the confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

    # Set the axis labels and title
    plt.xlabel(label)
    plt.ylabel('Actual')
    plt.title('Validation Confusion Matrix')

    # Show the plot
    plt.show()
    
def graph_window(df, y_pred):
    
    df['prediction'] = list(y_pred)
    # create a figure and axis object
    fig, ax = plt.subplots(figsize=(10, 6))

    # plot the last 500 Close values
    df.iloc[-1000:, 3].plot(ax=ax, linewidth  = 0.3)

    # loop through the index of the last 500 data points and plot a vertical line for every datapoint where the signal equals 2
    for index in df.iloc[-10000:, :].index:
        if df.loc[index, 'signal'] == 2:
            ax.axvline(x=index, color='g', linestyle='-', linewidth=0.1)
        elif df.loc[index, 'signal'] == 0:
            ax.axvline(x=index, color='r', linestyle='-', linewidth=0.1)

    # set the title and labels
    ax.set_title('Close Value with Signal = 2 and 0')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Value')

    # display the plot
    plt.show()