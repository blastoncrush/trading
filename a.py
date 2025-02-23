import pickle
import ia
from patterns import patterns
import torch
import time
import pandas as pd

SEQ_LENGTH = 5
PREDICTION_WINDOW = 3

torch.set_num_threads(4)
start_time = time.time()

title = "Dow30"
print("Indice : ", title)
file = open("./" + title, 'rb')
hist = pickle.load(file)
file.close()

hist = hist.reset_index()
start = pd.Timestamp("2024-01-01 00:00:00+00:00", tz = "UTC")
end = pd.Timestamp.utcnow()
hist = hist[hist["Date"] > start]
hist = hist[hist["Date"] < end]
hist = hist.set_index("Date")

hist, add_plots, labels = patterns(hist)

# Train the LSTM model
for j in range(1, 11):
    for k in range(1, 4):
        print("=== Modèle à", j, k, "neurones ===")
        model = ia.LSTMModel(1, j, k)
        
        for i in range(50000):
            loss = ia.train_model(hist, ["Close"], "SR_Close", 5, model)
            if i % 100 == 0:
                print(i, ":", loss)
        
        file = open(f"model{k}.pkl", 'wb')
        pickle.dump(model, file)
        file.close()

        hist = ia.display_predictions(hist, SEQ_LENGTH, PREDICTION_WINDOW, model)

        # Calcul de perf
        pred = hist['Prediction'].values
        prices = hist['Close'].values
        scoret = 0
        for i in range(len(pred)):
            if i > SEQ_LENGTH + PREDICTION_WINDOW:
                if (pred[i] > prices[i - PREDICTION_WINDOW] and prices[i] > prices[i - PREDICTION_WINDOW]) or \
                (pred[i] < prices[i - PREDICTION_WINDOW] and prices[i] < prices[i - PREDICTION_WINDOW]):
                    scoret += 1
        scoret /= len(hist['Prediction'].dropna())
        print(f"Score de tendance : {100 * scoret:.6f}%")

        scoreg = 0
        for i in range(len(pred)):
            if i > SEQ_LENGTH + PREDICTION_WINDOW:
                scoreg += abs(prices[i] - pred[i]) / prices[i - PREDICTION_WINDOW]
        scoreg /= len(hist['Prediction'].dropna())
        print(f"Score d'écart : {scoreg:.6f}")

        print("--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()