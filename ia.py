import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pickle
import csv
from sklearn.preprocessing import MinMaxScaler
import os

lr = 0.000001

# Définir le modèle LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1) # unique sortie (prédiction)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Prendre la dernière sortie temporelle
        return out.view(-1)

# Dataset pour les données temporelles
class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_columns, output_column, seq_length):
        self.inputs = data[input_columns].values
        self.outputs = data[output_column].values
        self.seq_length = seq_length

    def __len__(self):
        return len(self.inputs) - self.seq_length

    def __getitem__(self, idx):
        x = self.inputs[idx:idx + self.seq_length]
        y = self.outputs[idx + self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def mean_absolute_percentage_error(y_true, y_pred):
    return torch.mean(torch.abs((y_true - y_pred) / y_true))

# Fonction d'entraînement
def train_model(dataframe, input_columns, output_column, seq_length, model, epochs=10, batch_size=32, lr=0.000001):
    
    dataframe = dataframe.dropna(subset=input_columns + [output_column])

    dataset = TimeSeriesDataset(dataframe, input_columns, output_column, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # model.train()
    total_loss = 0

    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()

            predictions = model.forward(x_batch).view(-1) # On dégage la deuxième dimension de con

            loss = criterion(y_batch, predictions)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        total_loss += avg_loss
    
    return total_loss / epochs


def gives_data(model, input_columns, output_column, seq_length):
    for filename in os.listdir("data"):
        filename = os.path.join("data", filename)
        
        with open(filename, 'rb') as file:
            hist = pickle.load(file)

        if hist.empty:
            continue
        
        for column in input_columns:
            hist["SR_" + column] = hist[column].pct_change(fill_method=None)
            # scaler = MinMaxScaler()
            # hist["SR_" + column] = scaler.fit_transform(hist[["SR_" + column]])
        
        loss = train_model(hist, ["SR_"+column for column in input_columns], output_column, seq_length, model)
    
    return loss

def predict(model, dataframe, input_columns, output_column, seq_length):
    model.eval()
    dataset = TimeSeriesDataset(dataframe, input_columns, output_column, seq_length)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    predictions = []

    for x_batch, y_batch in dataloader:
        prediction = model.forward(x_batch)
        predictions.append(prediction.item())

    return predictions

def display_predictions(hist, seq_length, prediction_window, model):
    # Rescale the data
    # hist["SR_Close"] = hist["Close"].pct_change()
    # scaler = MinMaxScaler()
    # hist["Stock_return"] = scaler.fit_transform(hist[["Stock_return"]])

    # Predict the stock return
    hist['Prediction_SR'] = len(hist["Close"]) * [np.nan]
    stock_returns_close = list(hist["SR_Close"].values)

    for i in range(seq_length+prediction_window, len(hist["Close"])):
        rolling_pred = stock_returns_close[i-seq_length-prediction_window:i-prediction_window] # file d'attente
        for _ in range(prediction_window):
            x = torch.FloatTensor(rolling_pred).view(1, seq_length, 1)
            pred = model.forward(x).item()
            rolling_pred.pop()
            rolling_pred.append(pred)
        
        hist.loc[hist.index[i], 'Prediction_SR'] = pred

    print(f"std : {hist['Prediction_SR'].std(skipna=True):.6f}, Expected std : {hist['SR_Close'].std(skipna=True):.6f}")

    # Rescale the data
    # hist["Stock_return"] = scaler.inverse_transform(hist[["Stock_return"]])
    # hist["Prediction_SR"] = scaler.inverse_transform(hist[["Prediction_SR"]])

    # Inverse pct_change
    pred = list(hist["Prediction_SR"].values)
    prices = list(hist["Close"].values)
    hist['Prediction'] = len(hist["Close"]) * [np.nan]

    for i, ret in enumerate(hist["Prediction_SR"]):
        if i > seq_length+prediction_window:
            hist.loc[hist.index[i], 'Prediction'] = pred[i] * prices[i-prediction_window] + prices[i-prediction_window]
    
    return hist
