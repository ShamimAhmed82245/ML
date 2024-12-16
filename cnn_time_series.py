!pip install tensorflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Load or generate your temperature data
# Assuming you have a DataFrame 'df' with 'Timestamp' as index and 'Temperature' as column

# Preprocess the data
def preprocess_data(df, look_back=24):
    X, y = [], []
    for i in range(len(df) - look_back):
        X.append(df.iloc[i:(i+look_back), 0].values)
        y.append(df.iloc[i+look_back, 0])
    return np.array(X), np.array(y)

# Reshape the data for CNN input
look_back = 24
X, y = preprocess_data(df, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
X_train, X_test = X[0:train_size], X[train_size:len(X)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

# Create the CNN model
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(look_back, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=2)

# Make predictions
trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)

# Evaluate the model
# Calculate RMSE and MAPE
print('Train RMSE:', np.sqrt(np.mean((trainPredict[:,0] - y_train)**2)))
print('Test RMSE:', np.sqrt(np.mean((testPredict[:,0] - y_test)**2)))
# ...other evaluation metrics
