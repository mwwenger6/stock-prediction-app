import pandas as pd
import numpy as np
from copy import deepcopy as dc

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler

import argparse
import joblib
import json
import requests
import random

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

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='Ticker and pred range')
parser.add_argument('--ticker', type=str, help='ticker name')
parser.add_argument('--range', type=int, help='number of predicted days into the future')
args = parser.parse_args()
ticker = args.ticker

api_endpoint = 'https://stockgenieapi.azurewebsites.net/Stock/GetHistoricalStockData/' + ticker

response = requests.get(api_endpoint)
json_data = response.json()
data = process_json_data(json_data)

# load the scaler we used when training (to scale the data back)
scaler = joblib.load('wwwroot/Scalers/' + ticker + 'scaler.pkl')

# scale the data values
scaler_input = data['Close'].to_numpy().reshape(-1, 1)
scaled_prices = scaler.transform(scaler_input)
data['Close'] = scaled_prices

# reverse order of data
data = data[::-1]

lookback = 100
shifted_df = prepare_dataframe_for_lstm(data, lookback)

input_data = shifted_df.tail(1)
input_data = np.array(input_data)

X_pred = input_data[:, 1:]
y_pred = input_data[:, 0]

X_pred = dc(np.flip(X_pred, axis=1))

X_pred = X_pred.reshape((-1, lookback, 1))
y_pred = y_pred.reshape((-1, 1))

# makes sets into tensors
X_pred = torch.tensor(X_pred).float()
y_pred = torch.tensor(y_pred).float()

pred_dataset = TimeSeriesDataset(X_pred, y_pred)
batch_size = 1
pred_loader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=False)

# load model for predictions
PATH = "wwwroot/Models/" + ticker + "model.pth"
model = LSTM(1,4,1)
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))

pred_range = 21

# get stock volatility for the past month worth of prices (21 market days)
api_endpoint = 'https://stockrequests.azurewebsites.net/Stock/GetTechnicalStockInfoForStock/90/true/' + ticker

response = requests.get(api_endpoint)
json_data = response.json()
percent_volatility = json_data['percentVolatility']
percent_volatility = percent_volatility * .01

# to introduce random noise into predictions
def noise(price, percent_volatility):
    fluctuation_range = price * percent_volatility
    noise = random.uniform(-fluctuation_range, fluctuation_range)
    return noise

with torch.no_grad():
  predictions = []
  x = X_pred.to(device)
  for _ in range(pred_range):
    prediction = model(x).to('cpu').numpy().flatten()
    prediction = prediction.reshape(-1, 1)
    scaled_prediction = scaler.inverse_transform(prediction)
    scaled_prediction = scaled_prediction + noise(scaled_prediction, percent_volatility)
    predictions.append(scaled_prediction[0][0])
    noisyPred = scaler.transform(scaled_prediction)
    x = torch.cat((x[:, 1:, :], torch.tensor(noisyPred).reshape(1, 1, 1)), dim=1)

for prediction in predictions:
    print(str(prediction) + " ", end="")

