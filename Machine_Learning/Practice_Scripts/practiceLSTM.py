import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dc

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

data = pd.read_csv('Machine_Learning/AMZN.csv')

data = data[['Date', 'Close']]

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

data['Date'] = pd.to_datetime(data['Date'])

plt.plot(data['Date'].values, data['Close'].values)
plt.show()

def prepare_dataframe_for_lstm(df, n_steps):
  df = dc(df) # make a deepcopy

  df.set_index('Date', inplace=True) # set date is index

  # applies the shifting of the dataframe
  for i in range(1, n_steps + 1):
    df[f'Close(t-{i})'] = df['Close'].shift(i)
  
  df.dropna(inplace=True)

  return df

lookback = 7
shifted_df = prepare_dataframe_for_lstm(data, lookback)
shifted_df_as_np = shifted_df.to_numpy()

from sklearn.preprocessing import MinMaxScaler

# scales data to all be in between -1 and 1
scaler = MinMaxScaler(feature_range=(-1, 1))
shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)

X = shifted_df_as_np[:, 1:]
y = shifted_df_as_np[:, 0]

X = dc(np.flip(X, axis=1))

# determining at which index to split training and testing data
split_index = int(len(X) * 0.95)

# divide X and y sets into training and testing sets
X_train = X[:split_index] 
X_test = X[split_index:]

y_train = y[:split_index] 
y_test = y[split_index:]

X_train = X_train.reshape((-1, lookback, 1))
X_test = X_test.reshape((-1, lookback, 1))

y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

# makes sets into tensors
X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).float()
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).float()

class TimeSeriesDataset(Dataset):
  def __init__(self, X, y):
    self.X = X
    self.y = y
  
  def __len__(self):
    return len(self.X)
  
  def __getitem__(self, i):
    return self.X[i], self.y[i]

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

# this is what will be used to iterate over, get batches, and make updates to our model
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for _, batch in enumerate(train_loader):
  x_batch, y_batch = batch[0].to(device), batch[1].to(device)
  print(x_batch.shape, y_batch.shape)
  break

class LSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_stacked_layers):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_stacked_layers = num_stacked_layers
    
    self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)

    self.fc = nn.Linear(hidden_size, 1)
  
  def forward(self, x):
    batch_size = x.size(0)
    h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
    c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
    out, _ = self.lstm(x, (h0, c0))
    out = self.fc(out[:, -1, :])
    return out

model = LSTM(1, 4, 1)
model.to(device)

def train_one_epoch():
  model.train(True)
  print(f'Epoch: {epoch + 1}')
  running_loss = 0.0

  for batch_index, batch in enumerate(train_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)

    output = model(x_batch)
    loss = loss_function(output, y_batch)
    running_loss += loss.item()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch_index % 100 == 99: # print every 100 batches
      avg_loss_across_batches = running_loss / 100
      print('Batch {0}, Loss: {1:.3f}'.format(batch_index + 1, avg_loss_across_batches))
      running_loss = 0.0

  print()

def validate_one_epoch():
  model.train(False)
  running_loss = 0.0

  for batch_index, batch in enumerate(test_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)

    with torch.no_grad():
      output = model(x_batch)
      loss = loss_function(output, y_batch)
      running_loss += loss.item()
  
  avg_loss_across_batches = running_loss / len(test_loader)

  print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
  print('***************************************************')
  print()

  
learning_rate = 0.1
num_epochs = 10
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
  train_one_epoch()
  validate_one_epoch()

with torch.no_grad():
  predicted = model(X_train.to(device)).to('cpu').numpy()

# scale values back
train_predictions = predicted.flatten()

dummies = np.zeros((X_train.shape[0], lookback+1))
dummies[:, 0] = train_predictions
dummies = scaler.inverse_transform(dummies)

train_predictions = dc(dummies[:, 0])

dummies = np.zeros((X_train.shape[0], lookback+1))
dummies[:, 0] = y_train.flatten()
dummies = scaler.inverse_transform(dummies)

new_y_train = dc(dummies[:, 0])

# prediction results on input it has seen
plt.plot(new_y_train, label="Actual Close")
plt.plot(train_predictions, label="Predicted Close")
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()

test_predictions = model(X_test.to(device)).detach().cpu().numpy().flatten()
dummies = np.zeros((X_test.shape[0], lookback+1))
dummies[:, 0] = test_predictions
dummies = scaler.inverse_transform(dummies)

test_predictions = dc(dummies[:, 0])

dummies = np.zeros((X_test.shape[0], lookback+1))
dummies[:, 0] = y_test.flatten()
dummies = scaler.inverse_transform(dummies)

new_y_test = dc(dummies[:, 0])

# prediction results on input the model has not seen
plt.plot(new_y_test, label="Actual Close")
plt.plot(test_predictions, label="Predicted Close")
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()

# predicting closing prices for February
def predict_next_month(model, recent_data, device, scaler):
    model.eval()
    predictions = []

    for _ in range(29):  # Predict next 29 days
        # Scale the data
        recent_data_scaled = scaler.transform(recent_data.reshape(-1, recent_data.shape[-1])).reshape(recent_data.shape)
        print(recent_data_scaled.shape)
        # Convert to PyTorch tensor and add batch dimension
        input_data = torch.tensor(recent_data_scaled, dtype=torch.float32).to(device)
        print(input_data.shape)
        # Make a prediction
        with torch.no_grad():
            predicted_scaled = model(input_data)
        
        # Inverse scale the prediction
        predicted = scaler.inverse_transform(predicted_scaled.cpu().numpy().reshape(-1, 1))
        
        # Append the prediction to our list of predictions
        predictions.append(predicted.item())

        recent_features = recent_data[:, -1, 1:]
        
        # Append the predicted value to recent_data and remove the oldest value
        recent_data = np.append(recent_data[:, 1:, :], np.hstack([predicted, recent_features]).reshape(1, 1, -1), axis=1)

    return predictions

close_scaler = MinMaxScaler(feature_range=(-1, 1))
close_scaler.fit(shifted_df[['Close']])

# get last week of closing prices
# Get the last 'lookback' days of data with all features
recent_data = shifted_df.tail(lookback).values

# Reshape 'recent_data' to be 3D (add batch size dimension)
recent_data = recent_data.reshape(1, lookback, -1)

# Generate predictions for February
feb_predictions = predict_next_month(model, recent_data, device, scaler)

# Create 'dummies' with the same number of rows as 'feb_predictions'
dummies = np.zeros((len(feb_predictions), lookback+1))

# Assign 'feb_predictions' to the first column of 'dummies'
dummies[:, 0] = feb_predictions

# Apply the inverse transformation of the 'MinMaxScaler' to 'dummies'
dummies = scaler.inverse_transform(dummies)

# Extract the first column of 'dummies' (which contains the inverse transformed predictions)
feb_predictions = dc(dummies[:, 0])

# Plot the predictions
plt.plot(feb_predictions, label="Predicted Close")
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()












# import datetime as dt
# import time as time
# input_dates = []
# for i in range(1, 30):
#   if (i < 10):
#     date_str = "2024-02-0" + str(i)
#   else:
#     date_str = "2024-02-" + str(i)
#   # Convert the date string to a datetime object
#   date_obj = dt.datetime.strptime(date_str, '%Y-%m-%d')
#   # # Convert the datetime object to a Unix timestamp and append it to the list
#   # input_dates.append(time.mktime(date_obj.timetuple()))
#   input_dates.append(date_obj)

# # scales data to all be in between -1 and 1
# input_dates = np.array(input_dates)
# input_dates = input_dates.reshape(-1, 1)
# input_dates = scaler.fit_transform(input_dates)

# input_dates = np.array([input_dates[i:i+lookback] for i in range(len(input_dates) - lookback + 1)])
# input_dates = torch.tensor(input_dates).float()

# predictions = model(input_dates.to(device)).detach().cpu().numpy().flatten()

# dummies = np.zeros((input_dates.shape[0], lookback+1))
# dummies[:, 0] = predictions
# dummies = scaler.inverse_transform(dummies)
# predictions = dc(dummies[:, 0])

# plt.plot(predictions, label="Predicted Close")
# plt.xlabel('Day')
# plt.ylabel('Close')
# plt.legend()
# plt.show()
