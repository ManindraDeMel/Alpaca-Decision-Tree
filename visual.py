import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Stock Trading Bot'),

    dcc.Interval(id='interval-component', interval=1*1000, n_intervals=0),

    html.Div(id='live-update-text'),

    dcc.Graph(id='live-update-graph'),
])

@app.callback(Output('live-update-text', 'children'),
              Output('live-update-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_output(n):
    # Load the data from a file or database
    df = pd.read_csv('trading_data.csv')

    # Update the text
    last_row = df.iloc[-1]
    text = f"At {last_row['timestamp']}, the bot predicted {last_row['symbol']}'s price to be {last_row['predicted_price']} \
              and the action taken was to {last_row['action']}."

    # Update the graph
    figure = {
        'data': [{
            'x': df['timestamp'],
            'y': df['predicted_price'],
            'name': 'Predicted Price',
            'mode': 'lines+markers'
        }, {
            'x': df['timestamp'],
            'y': df['actual_price'],
            'name': 'Actual Price',
            'mode': 'lines+markers'
        }],
        'layout': {
            'title': 'Price Prediction vs Actual'
        }
    }

    return text, figure

if __name__ == '__main__':
    app.run_server(debug=True)
