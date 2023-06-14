# Import required libraries
import pandas as pd
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_table
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import plotly.express as px
from database_conn import Database_conn


# ML_results = ML_out[0]
#ML_results = {'AMZN': 0.98, 'GOOG': 0.97, 'AAPL': 0.95, 'BABA': 0.98}

def dash_plot(ML_results, ML_objects, ticker_list, ticker_data):
    #ML_results = {key: value.astype(str) for key, value in ML_results.items()}

    # Create a dash application
    app = dash.Dash(__name__)
    
    # Create an app layout
    app.layout = html.Div(children=[html.H1('Stock Price Visualisation',
                                            style={'textAlign': 'center', 'color': '#503D36',
                                                'font-size': 40}),

                                    dcc.Dropdown(id='selected_ticker',
                                                    options=[
                                                        {'label': ticker, 'value': ticker}
                                                        for ticker in ticker_list
                                                    ],
                                                    value='AMZN',
                                                    placeholder="Select Ticker",
                                                    searchable=True
                                                ),
                                    
                                    html.Br(),

                                    html.Div(
                                            children=[
                                                html.Div(
                                                    children=[
                                                        html.H3('ML Output Table'),
                                                        dash_table.DataTable(
                                                            id='ml-output-table',
                                                            columns=[{'name': col, 'id': col} for col in ML_results[0].keys()],
                                                            data=ML_results
                                                        )
                                                    ],
                                                    style={'width': '40%', 'display': 'inline-block', 'vertical-align': 'top'}
                                                ),
                                                html.Div(
                                                    children=[
                                                        dcc.Graph(id='ticker_price_plot')
                                                    ],
                                                    style={'width': '58%', 'display': 'inline-block', 'padding-left': '2%'}
                                                )
                                            ]
                                        )
                                    ]
                        )

    
    
    def graph_window(df, y_pred):
        # create a figure and axis object
        fig = go.Figure()

        # plot the last 500 Close values
        fig.add_trace(go.Scatter(
            x=df['id'][-1000:],  # Update this line
            y=df['close'][-1000:],
            mode='lines',
            line=dict(width=0.3),
            name='Close Value'
        ))

        # find the indices where the signal equals 2
        signal_2_indices = df.index[df['signal'] == 2]


        # add callouts for signal = 2
        for index in signal_2_indices:
            print(index, df.loc[index, 'close'])
            fig.add_annotation(
                x=index,
                y=df.loc[index, 'close'],
                text=
                '''Signal = Buy; Price = {}'''.format(df.loc[index, 'close']),
                showarrow=True,
                arrowhead=1,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor='green',
                ax=0,
                ay=-30,
                bordercolor='black',
                borderwidth=1,
                borderpad=2,
                bgcolor='lightgreen',
                opacity=0.8
            )

        # set the title and labels
        fig.update_layout(
            title='Close Value with Signal = 2',
            xaxis_title='Date',
            yaxis_title='Close Value'
        )

        return fig

    @app.callback(
        Output(component_id='ticker_price_plot', component_property='figure'),
        Input(component_id='selected_ticker', component_property='value')
    )
    def update_graph(selected_ticker):
        if not selected_ticker:
            # Return a default plot or a message indicating no stock is selected
            default_fig = go.Figure(data=[])
            default_fig.update_layout(title='No stock selected')
            return default_fig

        df = ticker_data[selected_ticker]
        df = df.drop('id', axis=1).reset_index().rename(columns={'index': 'id'})
        ticker_ML_object = ML_objects[selected_ticker]
        
        from ML_pipeline import ML_pipeline
        y_pred = ML_pipeline(df, ticker_ML_object)
        y_pred[-250] = 2
        y_pred[-200] = 2
        y_pred[-3] = 2
        df['signal'] = y_pred
        print(df)
        #df['signal'] = list(y_pred)
        # Call the graph_window function to generate the plot
        fig = graph_window(df, y_pred)
        return fig


    app.run_server()
    
    # # Run the app
    # if __name__ == '__main__':
    #     app.run_server()