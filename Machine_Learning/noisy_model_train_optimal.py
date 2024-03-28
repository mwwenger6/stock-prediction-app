import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dc

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler

import joblib
import json
import requests
import random

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def process_json_data(json_data):
    data = pd.json_normalize(json_data)
    data = data[['time', 'price']]
    data['time'] = pd.to_datetime(data['time'])
    data = data.rename(columns={'time': 'Date', 'price': 'Close'})
    return data

def prepare_dataframe_for_lstm(df, n_steps):
  df = dc(df) # make a deepcopy

  df.set_index('Date', inplace=True) # set date is index

  # applies the shifting of the dataframe
  for i in range(1, n_steps + 1):
    df[f'Close(t-{i})'] = df['Close'].shift(i)

  df.dropna(inplace=True)

  return df

class TimeSeriesDataset(Dataset):
  def __init__(self, X, y):
    self.X = X
    self.y = y

  def __len__(self):
    return len(self.X)

  def __getitem__(self, i):
    return self.X[i], self.y[i]

class LSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_stacked_layers):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_stacked_layers = num_stacked_layers

    self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)

    self.fc = nn.Linear(hidden_size, 1)

    self.activation = 'tanh'

  def forward(self, x):
    batch_size = x.size(0)
    h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
    c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
    out, _ = self.lstm(x, (h0, c0))
    out = self.fc(out[:, -1, :])
    return out

def train_one_epoch():
  model.train(True)
  # print(f'Epoch: {epoch + 1}')
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
      # print('Batch {0}, Loss: {1:.3f}'.format(batch_index + 1, avg_loss_across_batches))
      running_loss = 0.0

  model.train(False)

  # print('***************************************************')
  # print()

def mean_squared_error(y_true, y_pred):
    n = len(y_true)
    squared_errors = [(y_true[i] - y_pred[i]) ** 2 for i in range(n)]
    mse = sum(squared_errors) / n
    return mse

# to introduce random noise into predictions
def noise(price, percent_volatility):
    fluctuation_range = price * percent_volatility
    noise = random.uniform(-fluctuation_range, fluctuation_range)
    return noise

# def get_model_error(model, device, X_pred, validation_data, scaler, pred_range, percent_volatility):
#     loss_values = []

#     for i in range(100):

#         with torch.no_grad():
#             predictions = []
#             x = X_pred.to(device)
#             for _ in range(pred_range):
#                 prediction = model(x).to('cpu').numpy().flatten()
#                 prediction = prediction.reshape(-1, 1)
#                 scaled_prediction = scaler.inverse_transform(prediction)
#                 scaled_prediction = scaled_prediction + noise(scaled_prediction, percent_volatility)
#                 predictions.append(scaled_prediction[0][0])
#                 noisyPred = scaler.transform(scaled_prediction)
#                 x = torch.cat((x[:, 1:, :], torch.tensor(noisyPred).reshape(1, 1, 1)), dim=1)

#         # Calculate error
#         error = mean_squared_error(validation_data, predictions)
#         loss_values.append(error)

#     return sum(loss_values) / len(loss_values)

# get tickers
ticker_api_endpoint = 'https://stockrequests.azurewebsites.net/Stock/GetStocks'
ticker_response = requests.get(ticker_api_endpoint)
ticker_json_data = ticker_response.json()
tickers = [item['ticker'] for item in ticker_json_data]

# for each ticker, train a model and save it
for ticker in tickers:
  if (ticker == "AAPL"): # skip these tickers
    continue

  api_endpoint = 'https://stockgenieapi.azurewebsites.net/Stock/GetHistoricalStockData/' + ticker
  
  # get stock data
  response = requests.get(api_endpoint)
  json_data = response.json()
  data = process_json_data(json_data)

  # get validation data
  validation_data = data[:21]
  validation_data = validation_data[::-1]

  # scale the data, and save the scaler
  scaler = StandardScaler()
  scaler_input = data['Close'].to_numpy().reshape(-1, 1)
  scaler.fit(scaler_input)
  joblib.dump(scaler, 'Machine_Learning/Optimal_Scalers/' + ticker + 'scaler.pkl')

  scaled_prices = scaler.transform(scaler_input)
  data['Close'] = scaled_prices
  data = data[21:]
  validation_data = validation_data['Close'].to_numpy()
  data = data[::-1]

  lookback = 50
  shifted_df = prepare_dataframe_for_lstm(data, lookback)

  # for predictions / validation
  input_data = shifted_df.tail(1)
  input_data = np.array(input_data)
  X_pred = input_data[:, 1:]
  y_pred = input_data[:, 0]

  X_pred = dc(np.flip(X_pred, axis=1))

  X_pred = X_pred.reshape((-1, lookback, 1))
  y_pred = y_pred.reshape((-1, 1))

  X_pred = torch.tensor(X_pred).float()
  y_pred = torch.tensor(y_pred).float()

  # for training
  shifted_df_as_np = shifted_df.to_numpy()

  X_train = shifted_df_as_np[:, 1:]
  y_train = shifted_df_as_np[:, 0]

  X_train = dc(np.flip(X_train, axis=1))

  X_train = X_train.reshape((-1, lookback, 1))
  y_train = y_train.reshape((-1, 1))

  # makes sets into tensors
  X_train = torch.tensor(X_train).float()
  y_train = torch.tensor(y_train.copy()).float()

  train_dataset = TimeSeriesDataset(X_train, y_train)

  train_dataset = TimeSeriesDataset(X_train, y_train)

  # this is what will be used to iterate over, get batches, and make updates to our model
  batch_size = 16
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  for _, batch in enumerate(train_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)
    break

  # get stock volatility for the past month worth of prices (21 market days)
  api_endpoint = 'https://stockrequests.azurewebsites.net/Stock/GetTechnicalStockInfoForStock/90/true/' + ticker

  response = requests.get(api_endpoint)
  json_data = response.json()
  percent_volatility = json_data['percentVolatility']
  percent_volatility = percent_volatility * .01

  count = 0
  isOptimal = False
  lowest_model_error = 1000000
  while (isOptimal == False):

    count = count + 1
    model = LSTM(1, 4, 1)
    model.to(device)

    learning_rate = 0.1
    num_epochs = 10
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_one_epoch()

    loss_values = []

    for i in range(100):

        with torch.no_grad():
            predictions = []
            x = X_pred.to(device)
            for _ in range(21):
                prediction = model(x).to('cpu').numpy().flatten()
                prediction = prediction.reshape(-1, 1)
                scaled_prediction = scaler.inverse_transform(prediction)
                scaled_prediction = scaled_prediction + noise(scaled_prediction, percent_volatility)
                predictions.append(scaled_prediction[0][0])
                noisyPred = scaler.transform(scaled_prediction)
                x = torch.cat((x[:, 1:, :], torch.tensor(noisyPred).reshape(1, 1, 1)), dim=1)

        # Calculate error
        error = mean_squared_error(validation_data, predictions)
        loss_values.append(error)

    model_error = sum(loss_values) / len(loss_values)

    if (model_error < lowest_model_error):
        lowest_model_error = model_error

    print(f'{ticker} model error: {model_error}')
    if (model_error < 30 or (count > 10 and model_error < lowest_model_error + 20)):
        isOptimal = True
        # save the model parameters for future use with predicting
        PATH = "Machine_Learning/Optimal_Models/" + ticker + "model.pth"
        torch.save(model.state_dict(), PATH)

  print(f'{ticker} model trained and saved')