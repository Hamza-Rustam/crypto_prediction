import requests
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error


def fetch_kline_data(symbol, interval, limit):
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        "symbol": symbol,
        "interval": interval,  # e.g., "1m" for 1 minute
        "limit": limit  # Number of data points (candlesticks) to retrieve
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        # Convert to DataFrame
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']
        df = pd.DataFrame(data, columns=columns)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    else:
        print(f"Error: {response.status_code}")
        return None

# Fetch the last 60 minutes of BTCUSDT data
df = fetch_kline_data("BTCUSDT", "1m", 60)
#print(df.columns)



df["open"]= df["open"].astype("float")
df["high"]= df["high"].astype("float")
df["low"]= df["low"].astype("float")
df["close"]= df["close"].astype("float")
df["volume"]= df["volume"].astype("float")

#print(df.dtypes)

print(df.isnull().sum())  # This will show the count of null values in each column
df = df.dropna()  # Removes all rows with any null values


# Create lagged features
df['close_lag1'] = df['close'].shift(1)
df['close_lag2'] = df['close'].shift(2)

# Create moving averages
df['ma5'] = df['close'].rolling(window=5).mean()
df['ma10'] = df['close'].rolling(window=10).mean()

# Drop rows with NaN values created by shifting
df.dropna(inplace=True)

features =  df[['open', 'high', 'low',  'volume', 'close_lag1',
       'close_lag2', 'ma5', 'ma10']]

target = df['close']

"""print(f"feature are : {features}")
print(target)"""


x_train,x_test,y_train,y_test= train_test_split(features, target, test_size=0.2, random_state=42)



model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(x_train,y_train)

pred = model.predict(x_test)

r2score = r2_score(y_test, pred)
mse = mean_squared_error(y_test, pred)

print(f" r2 score is : {r2score}")
print(f" mean_squared_error is : {mse}")




pred = model.predict(x_test)

plt.scatter(y_test, pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)  # Diagonal line
plt.show()



feature_importances = pd.Series(model.feature_importances_, index=features.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title("Feature Importance")
plt.show()
