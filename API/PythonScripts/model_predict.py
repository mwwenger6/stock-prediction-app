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
import argparse

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

def process_json_file(json_file_path):
  with open(json_file_path, 'r') as file:
    data = json.load(file)
  data = pd.json_normalize(data)
  data = data[['time', 'price']]
  data['time'] = pd.to_datetime(data['time'])
  data = data.rename(columns={ 'time' : 'Date', 'price' : 'Close' })
  return data

parser = argparse.ArgumentParser(description='Process a JSON file.')
parser.add_argument('json_file', type=str, help='Path to the JSON file')
parser.add_argument('ticker', type=str, help='ticker name')
parser.add_argument('pred_range', type=str, help='number of predicted days into the future')
args = parser.parse_args()
json_file_path = args.json_file
data = process_json_file(json_file_path)

ticker = args.ticker

# load model for predictions
PATH = "Model/" + ticker + "model.pth"
model = LSTM(1,4,1)
model.load_state_dict(torch.load(PATH))

# load the scaler we used when training (to scale the data back)
scaler = joblib.load('Model/' + ticker + 'scaler.pkl')

# prepare input data
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

pred_range = int(args.pred_range)

input_data = shifted_df[:pred_range]
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
predictions = predictions.tolist()
json_results = json.dumps(predictions)

return json_results