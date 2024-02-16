import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dc

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('Machine_Learning/AMZN.csv')

data = data[['Date', 'Close']]

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

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
shifted_df_as_np = shifted_df.to_numpy()

scaler = MinMaxScaler(feature_range=(-1, 1))
shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)

X_train = shifted_df_as_np[:, 1:]
y_train = shifted_df_as_np[:, 0]

X_train = dc(np.flip(X_train, axis=1))

X_train = X_train.reshape((-1, lookback, 1))
y_train = y_train.reshape((-1, 1))

# makes sets into tensors
X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).float()

class TimeSeriesDataset(Dataset):
  def __init__(self, X, y):
    self.X = X
    self.y = y
  
  def __len__(self):
    return len(self.X)
  
  def __getitem__(self, i):
    return self.X[i], self.y[i]

train_dataset = TimeSeriesDataset(X_train, y_train)

# this is what will be used to iterate over, get batches, and make updates to our model
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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

learning_rate = 0.1
num_epochs = 10
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
  train_one_epoch()
