import numpy as np
import pandas as pd
import yfinance as yf
from math import sqrt
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error
import sys
from os.path import dirname, abspath
from collections import Counter

# Path setup
current_script_path = dirname(abspath(__file__))
parent_directory = dirname(current_script_path)
sys.path.append(parent_directory)

# Function to round decimal values
def round_up(x):
    return round(x, 3) if isinstance(x, (int, float)) else x

def clean_up(dirty_file, clean_file):
    df = pd.read_csv(dirty_file)
    df_clean = df.dropna(how='all')
    df_clean = df_clean.apply(lambda x: x.map(round_up) if x.dtype == 'float' else x)
    df_clean.to_csv(clean_file, index=False)
    return clean_file

def gather_data():
    stock_name = 'TSLA'
    time = '1d'
    raw_file = 'stock_dataMining/tsla_data_raw.csv'
    start_date, end_date = "2021-06-01", "2023-12-31"
    data = yf.download(stock_name, start=start_date, end=end_date, interval=time)
    data.to_csv(raw_file)
    return raw_file

def featured_stats(df, file_name):
    # Calculate various features
    sma_period, ema_period, volatility_period, rsi_period = 20, 14, 14, 20
    df['SMA'] = df['Close'].rolling(window=sma_period).mean().fillna(0)
    df['EMA'] = df['Close'].ewm(span=ema_period).mean()
    df['Volatility'] = df['Close'].rolling(window=volatility_period).std().fillna(0)
    delta = df['Close'].diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs)).fillna(0)
    ema_26, ema_12 = df['Close'].ewm(span=26).mean(), df['Close'].ewm(span=12).mean()
    df['MACD'] = ema_12 - ema_26
    df['Signal Line'] = df['MACD'].ewm(span=9).mean()
    df['Daily_Return'] = df['Close'].pct_change().fillna(0)
    mean_close, std_close = df['Close'].mean(), df['Close'].std()
    df['Normalized_Close'] = (df['Close'] - mean_close) / std_close
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
    df['10_Day_MA_Return'] = df['Daily_Return'].rolling(window=10).mean().fillna(0)
    df['Previous_Day_Return'] = df['Daily_Return'].shift(1).fillna(0)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.to_csv(file_name)

def main():
    data_file = gather_data()
    df = pd.read_csv(data_file)
    featured_file = "stock_dataMining/tsla_data_featured.csv"
    featured_stats(df, featured_file)
    clean_file = "stock_dataMining/tsla_data_clean_featured.csv"
    clean_up(featured_file, clean_file)

    df = pd.read_csv(clean_file)
    df_for_var = df[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA', 'EMA', 'Volatility', 'RSI', 'MACD', 'Signal Line', 'Daily_Return', 'Normalized_Close', 'Log_Return', '10_Day_MA_Return', 'Previous_Day_Return']]
    df_for_var_diff = df_for_var.diff().dropna()

    # Initialize variables to store the best RMSE and corresponding parameters
    best_rmse = float('inf')
    best_forecast_matrix = None
    best_actual_values = None
    best_mse = None

    # Define ranges for validation and test periods
    validation_range = range(20, 240, 20)  # Example range, adjust as needed
    test_range = range(10, 150, 10)         # Example range, adjust as needed

    for validation_period_size in validation_range:
        for test_period_size in test_range:
            # Calculate the split points
            validation_split = len(df) - validation_period_size - test_period_size
            test_split = len(df) - test_period_size

            # Ensure there's enough data for training
            if validation_split < 1:
                continue

            # Split the data
            train_data = df_for_var_diff[:validation_split]
            validation_data = df_for_var_diff[validation_split:test_split]
            test_data = df_for_var_diff[test_split:]

            # Check if there's enough data for VAR model
            if len(train_data) < 18 or len(validation_data) < 18:
                continue

            # Train the VAR model and calculate RMSE
            try:
                model = VAR(train_data.values)
                optimal_lag_order = model.select_order()
                lag_order = optimal_lag_order.selected_orders['aic']

                # Check if lag_order is feasible
                if lag_order >= len(train_data):
                    print(f"Skipping due to large lag_order: {lag_order} for train_data size: {len(train_data)}")
                    continue

                model_fit = model.fit(lag_order)
                forecast = model_fit.forecast(validation_data.values[-lag_order:], steps=len(validation_data))
                forecasted_values = forecast[:, 3]
                actual_values = validation_data['Close'].values
                mse = mean_squared_error(actual_values, forecasted_values)
                rmse = sqrt(mse)

                # Update best RMSE and corresponding parameters
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_mse = mse

                    # Round the forecasted and actual values individually
                    forecasted_values = np.round(forecasted_values, 3)
                    actual_values = np.round(actual_values, 3)

                    # Concatenate the values to create matrices
                    if best_forecast_matrix is None:
                        best_forecast_matrix = forecasted_values
                        best_actual_values = actual_values
                    else:
                        best_forecast_matrix = np.column_stack((best_forecast_matrix, forecasted_values))
                        best_actual_values = np.column_stack((best_actual_values, actual_values))

            except Exception as e:
                print(f"Error during modeling with validation_period_size={validation_period_size}, test_period_size={test_period_size}: {e}")

    # Round the results to three decimal places
    best_rmse = round(best_rmse, 3)
    best_mse = round(best_mse, 3)

    # Write the best results to file with matrix formatting
    if best_forecast_matrix is not None and best_actual_values is not None:
        with open("stock_dataMining/prediction_evaluation_best.txt", "w") as output_file:
            output_file.write(f"Best RMSE:\n{best_rmse}\n")
            output_file.write(f"Mean Squared Error:\n{best_mse}\n")
            output_file.write(f"Forecasted Values:\n{best_forecast_matrix}\n")
            output_file.write(f"Actual Values:\n{best_actual_values}\n")

    return best_rmse

if __name__ == '__main__':
    main()
