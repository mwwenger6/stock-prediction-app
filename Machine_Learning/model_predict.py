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


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# load model for predictions
PATH = "Machine_Learning/model.pth"
model = LSTM(1,4,1)
model.load_state_dict(torch.load(PATH))

# load the scaler we used when training (to scale the data back)
scaler = joblib.load('Machine_Learning/scaler.pkl')

# get input data needed to make a prediction for the next month (month of feb with 29 days)
data = pd.read_csv('Machine_Learning/AMZN.csv')
data = data[['Date', 'Close']]
data['Date'] = pd.to_datetime(data['Date'])
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

input_data = shifted_df.tail(29)
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
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()
