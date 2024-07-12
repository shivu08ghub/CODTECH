import yfinance as yf
import pandas as pd


#Step 1: Data collection
ticker='AAPL'
df=yf.download(ticker, start='2020-01-01',end='2023-01-01')


#Step 2: Data preprocessing
df=df[['Close']].copy() #explicitly create a copy of the DataFrame slice
df.dropna(inplace=True)


import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

#prepare data for LSTM model
scaler=MinMaxScaler(feature_range=(0,1))
scaled_df= scaler.fit_transform(df)

#split the data into training and testing sets
train_size = int(len(scaled_df) * 0.8)
test_size = len(scaled_df) - train_size
train_df, test_df = scaled_df[:train_size], scaled_df[train_size:]

#create datasets for LSTM
def create_dataset(dataset, look_back=60):
    X, y = [],[] #properly initialize x and y as empty lists
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)

#lenght of sequences
look_back = 60

#create sequences of training
X_train, y_train = create_dataset(train_df, look_back)

#create sequences of testing
X_test, y_test = create_dataset(test_df, look_back)

if X_train.size==0 or X_test.size==0:
    raise ValueError("Not enough data to create training and testing datasets. Increase the look_back period or the size of your dataset.")

#reshape the data to be 3D for LSTM [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#Step 3: ARIMA Model
from statsmodels.tsa.arima.model import ARIMA

#convert index to datetime
df.index = pd.to_datetime(df.index)

#define the model
model_arima = ARIMA(df['Close'], order=(5,1,0))

#fit the model
model_arima_fit = model_arima.fit()

#make predictions
forecast_arima = model_arima_fit.forecast(steps=30)
forecast_arima = scaler.inverse_transform(forecast_arima.values.reshape(-1, 1))

#Step 4: LSTM Model
#build the LSTM model
model_lstm = Sequential()
model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
model_lstm.add(LSTM(units=50))
model_lstm.add(Dense(1))

#compile the model
model_lstm.compile(optimizer='adam', loss='mean_squared_error')

#train the LSTM model
model_lstm.fit(X_train, y_train, epochs=20, batch_size=32)

#make predictions with the LSTM model
predictions_lstm = model_lstm.predict(X_test)
predictions_lstm = scaler.inverse_transform(predictions_lstm)

#step 5: Visualization
#plotting the results
import matplotlib.pyplot as plt

from matplotlib import style
style.use('Solarize_Light2')
#print(plt.style.available)

#plot the closing price
plt.figure(figsize=(12,6))
font1={'family':'serif','color':'green','size':25}
font2={'family':'serif','color':'red','size':15}
plt.plot(df.index,df['Close'], label='Closing Price')
plt.title('APPLE Closing Prices Prediction',fontdict=font1)
plt.xlabel('Date',fontdict=font2)
plt.ylabel('Price',fontdict=font2)
plt.legend()
plt.grid(axis='x',color='r',linestyle='--',linewidth=2)
plt.grid(axis='y',color='y',linestyle=':',linewidth=4)
plt.show()


#plot LSTM predictions
plt.figure(figsize=(14, 7))
font1={'family':'serif','color':'red','size':25}
font2={'family':'serif','color':'green','size':15}
plt.plot(df.index[look_back + train_size + 1:], predictions_lstm, label='LSTM Forecast', color='green')
plt.title('AAPL Stock Price Forecast Prediction using LSTM',fontdict=font1)
plt.xlabel('Date',fontdict=font2)
plt.ylabel('Stock Price',fontdict=font2)
plt.legend()
plt.grid(axis='x',color='red',linestyle='--',linewidth=2)
plt.grid(axis='y',color='purple',linestyle=':',linewidth=4)
plt.show()


#plot the ARIMA forecast predictions 
plt.figure(figsize=(10,5))
font1={'family':'serif','color':'red','size':25}
font2={'family':'serif','color':'green','size':15}
plt.plot(df.index,df['Close'],label='Historical')
plt.plot(pd.date_range(start=df.index[-1], periods=31, freq='B')[1:], forecast_arima, label='ARIMA Forecast', color='red')
plt.title('AAPL Stock Price Forecast Prediction using ARIMA',fontdict=font1)
plt.xlabel('Date',fontdict=font2)
plt.ylabel('Price',fontdict=font2)
plt.legend()
plt.grid(axis='x',color='purple',linestyle='--',linewidth=2)
plt.grid(axis='y',color='yellow',linestyle=':',linewidth=2)
plt.show()










































