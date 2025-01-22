from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import yfinance as yf
import pandas as pd


#Download Intraday Data from yfinance
symbol = 'AAPL'
df = yf.download(tickers=symbol, interval='1m', period='5d')

#Define Helper Functions (RSI, Bollinger, etc.)
def calculate_rsi(data, column='Close', window=14):
    delta = data[column].diff(1)
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def add_bollinger_bands(data, column='Close', window=20, num_std=2):
    rolling_mean = data[column].rolling(window=window).mean()
    rolling_std = data[column].rolling(window=window).std()
    data['BB_Middle'] = rolling_mean
    data['BB_Upper'] = rolling_mean + (rolling_std * num_std)
    data['BB_Lower'] = rolling_mean - (rolling_std * num_std)


#Technical Indicators
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [f"{col[0]}_{col[1]}" for col in df.columns]

#Identify the correct Close column name
close_col = None
for col in df.columns:
    if "Close" in col and "Adj" not in col:
        close_col = col
        break

#If we couldn't find a close column, raise an error
if close_col is None:
    raise ValueError("No suitable 'Close' column found in the DataFrame columns.")

#Moving Averages (short, medium)
df['MA_5'] = df[close_col].rolling(window=5).mean()
df['MA_15'] = df[close_col].rolling(window=15).mean()
df['MA_Crossover'] = df['MA_5'] - df['MA_15']

#Bollinger Bands
add_bollinger_bands(df, column=close_col, window=20, num_std=2)

#RSI
df['RSI'] = calculate_rsi(df, column=close_col, window=14)

#Rolling Volatility (Std)
df['Volatility_15'] = df[close_col].rolling(window=15).std()

#Price Returns
df['Return_1m'] = df[close_col].pct_change(1)
df['Return_5m'] = df[close_col].pct_change(5)

# Lagged Features (example: last close)
df['Lag_Close'] = df[close_col].shift(1)

#Time-based Features
df.index = pd.to_datetime(df.index)
df['Hour'] = df.index.hour
df['Minute'] = df.index.minute
df['DayOfWeek'] = df.index.dayofweek 

#Create Target for "Next Hour" Price Direction
df['Future_Close'] = df[close_col].shift(-60)
df['Target'] = (df['Future_Close'] > df[close_col]).astype(int)

#Clean Data (drop rows with NaNs from calculations/shifting)
df.dropna(inplace=True)

print(df.tail(10))
print("Dataframe shape:", df.shape)
print("Dataframe columns:", df.columns.tolist())

target = df['Target']

features = df.drop(columns=['Target', 'Future_Close'])

#Setup for XGBoost algorithim
Stock_Forecaster = XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.1, objective='binary:logistic')

x_train, x_test, y_train, y_test = train_test_split(features,target,test_size=0.3)

Stock_Forecaster.fit(x_train, y_train)
y_pred = Stock_Forecaster.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print(accuracy)