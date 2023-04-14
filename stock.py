from cProfile import label
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from keras.models import load_model
import streamlit as st
import yfinance as yf
from datetime import date, timedelta
from sklearn.metrics import mean_absolute_error

start = '2010-01-01'
today = date.today()
yesterday = today - timedelta(days=1)
end = yesterday
yf.pdr_override()
st.title('Stock Trend Prediction')
user_input = st.text_input('Enter Stock Ticker', 'AAPL')

df = pdr.get_data_yahoo(user_input, start, end)

# Describing Data to User

st.subheader('Data from 2010 to yesterday ')
st.write(df.describe())

# Visualiztions

st.subheader('Closing Price v/s Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price v/s Time Chart with 50MA and 200MA')
fig2 = plt.figure(figsize = (12,6))
ma50 = df.Close.rolling(50).mean()
ma200 = df.Close.rolling(200).mean()
plt.plot(df.Close)
plt.plot(ma50, 'r')
plt.plot(ma200, 'g')
st.pyplot(fig2)

df = df.reset_index()
df = df.drop(['Date','Adj Close'], axis=1)

# Splitting data into training and testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.60)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.60):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)
x_train = []
y_train = []
for i in range(100, data_training.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])

x_train = np.array(x_train)
y_train = np.array(y_train)

# Load model
model = load_model('keras_model.h5')

# Machine Learning Model

# from keras.layers import Dense, Dropout, LSTM
# from keras.models import Sequential

# model = Sequential()
# model.add(LSTM(units = 90, activation = 'softsign', return_sequences = True, input_shape = (x_train.shape[1],1)))
# model.add(Dropout(0.2))

# model.add(LSTM(units = 100, activation = 'softsign', return_sequences = True))
# model.add(Dropout(0.3))

# model.add(LSTM(units = 120, activation = 'softsign', return_sequences = True))
# model.add(Dropout(0.4))

# model.add(LSTM(units = 160, activation = 'softsign'))
# model.add(Dropout(0.5))

# model.add(Dense(units = 1))

# model.compile(optimizer='RMSprop', loss = 'mean_squared_error')
# model.fit(x_train, y_train, epochs=50)
# model.save('keras_model.h5')

past_100_days = data_testing.tail(100)
final_df = past_100_days._append(data_testing, ignore_index= True)
input_data = scaler.fit_transform(final_df)
x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test = np.array(x_test)
y_test = np.array(y_test)

# Making Predictions

y_predicted = model.predict(x_test)

scaler = scaler.scale_

scale_factor  = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

st.subheader('Predictions v/s Original Price')
fig3 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b' , label = 'Original Price')
plt.plot(y_predicted, 'r', label= 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)

mape = mean_absolute_error(y_test, y_predicted)
st.write("Mean absolute error")
st.write(mape)