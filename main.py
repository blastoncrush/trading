import matplotlib.collections
import yfinance as yf
import mplfinance as mpf
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import pickle
from patterns import patterns
import interface
import pandas_ta as ta
from backtesting import Strategy, Backtest
import seaborn as sns

"""
ive got a 6 gb csv file with the last 5000 days of price info for around 4800 companies merged with quarterly financial data like assets revenue etc merged with macro indicators like interest rates and all of that interpolated and cleaned up before passing to an lstm with many many layers with tuned hyper params. instead of using a standard loss value you should use mean absolute percentage error because if a stock is trading at 1$ and you have a loss of 1$ you would be off by 100% but if its trading at 1000$ and yoru loss is 10 then you would be off by 1%.
also try to use time distributed lstm layers at the input and to use leaky relu not just relu so that parts of the model that arnt use often dont just become dead weight . Also use a deeper model to capture the complex relationship between the columns in the training data and MOST IMPORTANTLY..... Kill any over fitting with lots of dropout layers and batch norm
"""

SEQ_LENGTH = 5
PREDICTION_WINDOW = 3

selected_index = interface.select_index()
#selected_index = "Dow 30"

title = selected_index
print("Indice : ", title)
file = open("./data/" + title, 'rb')
hist = pickle.load(file)
file.close()

# Sélectionner une certaine période
hist = hist.reset_index()
start = pd.Timestamp("2024-06-01 00:00:00+00:00", tz = "UTC")
end = pd.Timestamp.utcnow()
hist = hist[hist["Date"] > start]
hist = hist[hist["Date"] < end]
hist = hist.set_index("Date")

hist, add_plots, labels = patterns(hist)

for ap in add_plots:
    ap['data'] = ap['data'].to_numpy()

"""
# Train the LSTM model
model = ia.LSTMModel(1, 3, 1)
loss = ia.train_model(hist, ["Close"], "SR_Close", 5, model)

# ia.gives_data(model, ["Close"], "SR_Close", SEQ_LENGTH)
file = open("model.pkl", 'wb')
pickle.dump(model, file)
file.close()

file = open("model.pkl", 'rb')
model = pickle.load(file)
file.close()

hist = ia.display_predictions(hist, SEQ_LENGTH, PREDICTION_WINDOW, model)

add_plots.append(mpf.make_addplot(hist['Prediction'], label="Prediction", color='gray', panel=0))

hist = pd.concat([hist, pd.DataFrame(np.nan, index=hist.index, columns=[f'future_{i+1}' for i in range(PREDICTION_WINDOW)])], axis=1)

# Calcul de perf
pred = hist['Prediction'].values
prices = hist['Close'].values
scoret = 0
for i in range(len(pred)):
    if i > SEQ_LENGTH+PREDICTION_WINDOW:
        if pred[i] > prices[i-PREDICTION_WINDOW] and prices[i] > prices[i-PREDICTION_WINDOW] or pred[i] < prices[i-PREDICTION_WINDOW] and prices[i] < prices[i-PREDICTION_WINDOW]:
            scoret += 1
scoret /= len(hist['Prediction'].dropna())
print(f"Score de tendance : {100*scoret:.6f}%")

scoreg = 0
for i in range(len(pred)):
    if i > SEQ_LENGTH+PREDICTION_WINDOW:
        scoreg += abs(prices[i] - pred[i]) / prices[i-PREDICTION_WINDOW]
scoreg /= len(hist['Prediction'].dropna())
print(f"Score d'écart : {scoreg:.6f}")
"""

# Plot the candlestick chart with moving averages and Doji patterns
fig, axlist = mpf.plot(hist, type='candle', style='yahoo', title=f'{title}',
                       ylabel='Price (EUR)', volume=True, addplot=add_plots, returnfig=True,
                       main_panel=0, num_panels=5, panel_ratios=(.2, .05, .05, .05, .01)) # volume_panel = 4


# Create interactive check buttons for each layer
rax = plt.axes([0.0, 0.0, 0.1, 1.])
visibility = [False] * len(labels)
check = CheckButtons(rax, labels, visibility)

# Hidden by default
for i, ax in enumerate(axlist):
    for line in ax.lines:
        if line.get_label() in labels:
            line.set_visible(not line.get_visible())
    for collection in ax.collections:
        if isinstance(collection, matplotlib.collections.PathCollection): #on ne touche pas aux chandeliers
            collection.set_visible(not collection.get_visible())

# Define update function to toggle visibility
def func(label):
    for ax in axlist:
        for line in ax.lines:
            if line.get_label() == label: # plus propre en faisant la discjonction sur ax
                line.set_visible(not line.get_visible())
        for collection in ax.collections:
            #if isinstance(collection, matplotlib.collections.PathCollection): #on ne touche pas aux chandeliers
                if collection.get_label() == label:
                    collection.set_visible(not collection.get_visible())
    plt.draw()

# Connect check buttons with update function
check.on_clicked(func)

plt.show()