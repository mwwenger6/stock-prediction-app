import pandas as pd
import numpy as np
from copy import deepcopy as dc
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler

import joblib
import json
import argparse
import requests

#ticker to test
ticker = 'AAPL'

# our model's class
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

def process_json_data(json_data):
    data = pd.json_normalize(json_data)
    data = data[['time', 'price']]
    data['time'] = pd.to_datetime(data['time'])
    data = data.rename(columns={'time': 'Date', 'price': 'Close'})
    return data

device = 'cpu'

api_endpoint = 'https://stockgenieapi.azurewebsites.net/Home/GetHistoricalStockData/' + ticker

response = requests.get(api_endpoint)
json_data = response.json()
data = process_json_data(json_data)

validation_data = data['Close'][:60]
validation_data = np.array(validation_data)
validation_data = np.flip(validation_data)

inp_d = data['Close'][:120]
inp_d = inp_d[:60]
inp_d = np.array(inp_d)
inp_d = np.flip(inp_d)

# load model for predictions
PATH = ticker + "model.pth"
model = LSTM(1,4,1)
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))

# load the scaler we used when training (to scale the data back)
scaler = joblib.load(ticker + 'scaler.pkl')

# prepare input data
def prepare_dataframe_for_lstm(df, n_steps):
  df = dc(df) # make a deepcopy

  df.set_index('Date', inplace=True) # set date is index

  # applies the shifting of the dataframe
  for i in range(1, n_steps + 1):
    df['Close(t-' + str(i) + ')'] = df['Close'].shift(i)

  df.dropna(inplace=True)

  return df

lookback = 7
shifted_df = prepare_dataframe_for_lstm(data, lookback)

pred_range = 60

input_data = shifted_df[:(pred_range*2)]
input_data = input_data[60:]

input_data = np.array(input_data)

input_data = scaler.transform(input_data)

X_pred = input_data[:, 1:]
y_pred = input_data[:, 0]

X_pred = dc(np.flip(X_pred, axis=1))

X_pred = X_pred.reshape((-1, lookback, 1))
y_pred = y_pred.reshape((-1, 1))

# makes sets into tensors
X_pred = torch.tensor(X_pred).float()
y_pred = torch.tensor(y_pred).float()

class TimeSeriesDataset(Dataset):
  def __init__(self, X, y):
    self.X = X
    self.y = y

  def __len__(self):
    return len(self.X)

  def __getitem__(self, i):
    return self.X[i], self.y[i]

pred_dataset = TimeSeriesDataset(X_pred, y_pred)
batch_size = 16
pred_loader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=False)

with torch.no_grad():
  predicted = model(X_pred.to(device)).to('cpu').numpy()

# scale values back
predictions = predicted.flatten()

dummies = np.zeros((X_pred.shape[0], lookback+1))
dummies[:, 0] = predictions
dummies = scaler.inverse_transform(dummies)

predictions = dc(dummies[:, 0])

# prediction results
plt.plot(predictions, label="Predicted Close")
plt.plot(validation_data, label="Actual Close")
plt.plot(inp_d, label="Actual Prev Close")
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()