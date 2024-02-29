import pandas as pd
import numpy as np
from copy import deepcopy as dc

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler

import joblib
import json
import argparse
import requests

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

device = 'cpu'

def process_json_data(json_data):
    data = pd.json_normalize(json_data)
    data = data[['time', 'price']]
    data['time'] = pd.to_datetime(data['time'])
    data = data.rename(columns={'time': 'Date', 'price': 'Close'})
    return data

parser = argparse.ArgumentParser(description='Ticker and new data range')
parser.add_argument('--ticker', type=str, help='ticker name')
parser.add_argument('--range', type=int, help='number of new data days')
args = parser.parse_args()
ticker = args.ticker

api_endpoint = 'https://stockgenieapi.azurewebsites.net/Home/GetHistoricalStockData/' + ticker

response = requests.get(api_endpoint)
json_data = response.json()
data = process_json_data(json_data)

# load model for updates
PATH = "wwwroot/Models/" + ticker + "model.pth"
model = LSTM(1,4,1)
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))

# load the scaler we used when training to scale the data going in
scaler = joblib.load('wwwroot/Scalers/' + ticker + 'scaler.pkl')

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

new_data_range = int(args.range)

X_new = input_data[:, 1:]
y_new = input_data[:, 0]

X_new = dc(np.flip(X_new, axis=1))

X_new = X_new.reshape((-1, lookback, 1))
y_new = y_new.reshape((-1, 1))

# makes sets into tensors
X_new = torch.tensor(X_new).float()
y_new = torch.tensor(y_new).float()

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]

new_dataset = TimeSeriesDataset(X_new, y_new)
batch_size = 16
new_loader = DataLoader(new_dataset, batch_size=batch_size, shuffle=False)

# make updates to our model
for inputs, targets in new_loader:
    inputs = inputs.to(device)
    targets = targets.to(device)

    # Forward pass
    outputs = model(inputs)

    # Compute loss
    loss = criterion(outputs, targets)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

