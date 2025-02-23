import mplfinance as mpf
import numpy as np
from scipy.signal import argrelextrema
import pandas_ta as ta

def patterns(hist):
    
    # Moving Averages
    hist['MA20'] = hist['Close'].rolling(window=20).mean()
    hist['MA50'] = hist['Close'].rolling(window=50).mean()
    hist['SR_Close'] = hist['Close'].pct_change()
    hist['SR_Open'] = hist['Open'].pct_change()
    hist["EMA_50"]=ta.ema(hist.Close, length=50) # Moyenne mobile exponentielle sur 50 périodes
    hist["EMA_30"]=ta.ema(hist.Close, length=30) # Moyenne mobile exponentielle sur 30 périodes

    # RSI
    def calculate_rsi(hist, window=14):
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        hist['RSI'] = rsi
        return hist

    hist = calculate_rsi(hist)
    assert 'RSI' in hist.columns

    # Stochastic
    def calculate_stochastic(hist, k_period=14, d_period=3):
        hist['L14'] = hist['Low'].rolling(window=k_period).min()
        hist['H14'] = hist['High'].rolling(window=k_period).max()
        hist['%K'] = 100 * ((hist['Close'] - hist['L14']) / (hist['H14'] - hist['L14']))
        hist['%D'] = hist['%K'].rolling(window=d_period).mean()
        return hist

    hist = calculate_stochastic(hist)

    # MACD
    def calculate_macd(hist, short_window=12, long_window=26, signal_window=9):
        # Calculate the short-term and long-term exponential moving averages (EMAs)
        hist['EMA_short'] = hist['Close'].ewm(span=short_window, adjust=False).mean()
        hist['EMA_long'] = hist['Close'].ewm(span=long_window, adjust=False).mean()
        
        # Calculate the MACD line
        hist['MACD'] = hist['EMA_short'] - hist['EMA_long']
        
        # Calculate the Signal line
        hist['Signal'] = hist['MACD'].ewm(span=signal_window, adjust=False).mean()
        
        return hist

    #hist = calculate_macd(hist)

    """   Exploitation of indicators   """
    # RSI
    def find_rsi_supports_resistances(hist, order=5):
        # Points pivot (minima et maxima locaux)
        rsi = hist['RSI']
        local_min = argrelextrema(rsi.values, np.less, order=order)[0]
        local_max = argrelextrema(rsi.values, np.greater, order=order)[0]
        local_min = hist.index[local_min]
        local_max = hist.index[local_max]

        # Ajout des supports et résistances dans le DataFrame
        hist['RSI Support'] = np.nan
        hist['RSI Resistance'] = np.nan
        hist.loc[local_min, 'RSI Support'] = rsi[local_min]
        hist.loc[local_max, 'RSI Resistance'] = rsi[local_max]

        return hist
    
    hist = find_rsi_supports_resistances(hist)

    def rsi_buy(hist):
        return (hist['RSI'] > 50) & (hist['RSI'].shift(1) <= 50) & (hist['RSI'].shift(3) <= 40)

    hist['RSI Buy'] = np.where(rsi_buy(hist), hist["RSI"], np.nan)

    def rsi_sell(hist):
        return (hist['RSI'] < 50) & (hist['RSI'].shift(1) >= 50) & (hist['RSI'].shift(3) >= 60)

    hist['RSI Sell'] = np.where(rsi_sell(hist), hist["RSI"], np.nan)

    # Stochastic
    def stochastic_buy(hist):
        return (
            (hist['%K'].shift(1) < 20) & (hist['%D'].shift(1) < 20) &  # Zone de survente
            (hist['%K'] > 20) &
            (hist['%K'] > hist['%D'])                                 # %K dépasse %D
        )
    
    hist['Stochastic Buy'] = np.where(stochastic_buy(hist), hist["%K"], np.nan)

    def stochastic_sell(hist):
        return (
            (hist['%K'].shift(2) > 80) & (hist['%D'].shift(2) > 80) &
            (hist['%K'].shift(1) < 80) & (hist['%K'].shift(1) < hist['%D'].shift(1)) &
            (hist['%K'] < 75) & (hist['%K'] > 60)
        )
    
    hist['Stochastic Sell'] = np.where(stochastic_sell(hist), hist["%K"], np.nan)

    # MACD
    def macd_buy(hist):
        return (
            (hist['MACD'] < 0) & (hist['Signal'] < 0) &               # MACD et Signal négatifs
            (hist['MACD'] > hist['Signal']) &                        # Croisement MACD > Signal
            (hist['MACD'].shift(1) <= hist['Signal'].shift(1))       # Croisement récent
        )
    
    #hist["MACD Buy"] = np.where(macd_buy(hist), hist["MACD"], np.nan)
    
    """   Candlestick pattern   """
    # Doji
    def is_doji(hist):
        threshold = 0.0005  # 0.05% threshold for identifying a Doji
        return np.abs(hist['Close'] - hist['Open']) / hist['Close'] < threshold

    hist['Doji'] = np.where(is_doji(hist), np.abs(hist['Close'] + hist['Open'])/2, np.nan)

    # Hammer
    def is_hammer(hist):
        prop = 1.5
        threshold = 0.1
        return ((hist['High'] - hist['Low']) / np.abs(hist['Close'] - hist['Open']) > prop) & \
        ((hist['High'] - hist['Close']) < threshold)


    hist['Hammer'] = np.where(is_hammer(hist), np.abs(hist['Close'] + hist['Open'])/2, np.nan)

    # Shooting star
    def is_shooting_star(hist):
        body_size = np.abs(hist['Close'] - hist['Open'])
        is_small_body = body_size / (hist['High'] - hist['Low']) < 0.1
        low = np.minimum(hist['Open'], hist['Close'])
        return is_small_body & \
               (hist['Close'].shift(1) < low) & (low > hist['Open'].shift(-1)) & \
               (hist['Close'].shift(1) > hist['Open'].shift(1)) & (hist['Close'].shift(-1) < hist['Open'].shift(-1))

    hist['Shooting star'] = np.where(is_shooting_star(hist), np.abs(hist['Close'] + hist['Open'])/2, np.nan)

    # Morning star
    def is_morning_star(hist):
        body_size = np.abs(hist['Close'] - hist['Open'])
        is_small_body = body_size / (hist['High'] - hist['Low']) < 0.1
        low = np.minimum(hist['Open'], hist['Close'])
        return is_small_body & \
               (hist['Close'].shift(1) > low) & (low < hist['Open'].shift(-1)) & \
               (hist['Close'].shift(1) < hist['Open'].shift(1)) & (hist['Close'].shift(-1) > hist['Open'].shift(-1))

    hist['Morning star'] = np.where(is_morning_star(hist), np.abs(hist['Close'] + hist['Open'])/2, np.nan)

    # Bullish engulfing
    def is_bullish_engulfing(hist):
        return (hist['Close'].shift(1) > hist['Open']) & (hist['Close'] > hist['Open'].shift(1)) & \
               (hist['Close'].shift(1) < hist['Open'].shift(1)) & (hist['Close'] > hist['Open'])

    hist['Bullish engulfing'] = np.where(is_bullish_engulfing(hist), np.abs(hist['Close'] + hist['Open'])/2, np.nan)

    # Bearish engulfing
    def is_bearish_engulfing(hist):
        return (hist['Close'].shift(1) < hist['Open']) & (hist['Close'] < hist['Open'].shift(1)) & \
               (hist['Close'].shift(1) > hist['Open'].shift(1)) & (hist['Close'] < hist['Open'])

    hist['Bearish engulfing'] = np.where(is_bearish_engulfing(hist), np.abs(hist['Close'] + hist['Open'])/2, np.nan)

    # Drop columns with all NaN values to handle empty patterns
    if 'Adj Close' in hist.columns:
        print(hist.columns)
    hist.dropna(axis=1, how='all', inplace=True)
    
    labels = ["MA20", "MA50", "EMA_50", "EMA_30", "Doji", "Hammer", "Shooting star", "Morning star", "Bullish engulfing", "Bearish engulfing", "RSI Buy", "RSI Sell", "RSI Support", "RSI Resistance", "Stochastic Buy", "Stochastic Sell"] # "MACD Buy"
    # Create additional plots
    add_plots = [mpf.make_addplot(hist['RSI'], color='green', panel=1, ylabel='RSI', type='line', ylim=(0, 100)),
                 mpf.make_addplot(hist['%K'], color='orange', panel=2, ylabel='Stochastic', type='line', ylim=(0, 100)),
                 mpf.make_addplot(hist['%D'], color='purple', panel=2, ylabel='Stochastic', type='line', ylim=(0, 100)),
                 ] # mpf.make_addplot(hist['MACD'], color='cyan', panel=3, ylabel='MACD', type='line', ylim=(-10, 10)),
                   # mpf.make_addplot(hist['Signal'], color='brown', panel=3, ylabel='MACD', type='line', ylim=(-10, 10))

    # Add candlestick patterns to add_plots only if they exist in hist
    if 'MA20' in hist.columns:
        add_plots.append(mpf.make_addplot(hist['MA20'], label='MA20', color='blue', panel = 0))
    if 'MA50' in hist.columns:
        add_plots.append(mpf.make_addplot(hist['MA50'], label='MA50', color='red', panel = 0))
    if 'EMA_50' in hist.columns:
        add_plots.append(mpf.make_addplot(hist['EMA_50'], label='EMA_50', color='green', panel = 0))
    if 'EMA_30' in hist.columns:
        add_plots.append(mpf.make_addplot(hist['EMA_30'], label='EMA_30', color='purple', panel = 0))
    if 'Doji' in hist.columns:
        add_plots.append(mpf.make_addplot(hist['Doji'], label='Doji', type='scatter', markersize=40, marker='o', color='white', edgecolors='black', panel = 0))
    if 'Hammer' in hist.columns:
        add_plots.append(mpf.make_addplot(hist['Hammer'], label='Hammer', type='scatter', markersize=100, marker='X', color='green', edgecolors='black', panel = 0))
    if 'Shooting star' in hist.columns:
        add_plots.append(mpf.make_addplot(hist['Shooting star'], label='Shooting star', type='scatter', markersize=100, marker='*', color='red', edgecolors='black', panel = 0))
    if 'Morning star' in hist.columns:
        add_plots.append(mpf.make_addplot(hist['Morning star'], label='Morning star', type='scatter', markersize=100, marker='*', color='green', edgecolors='black', panel = 0))
    if 'Bullish engulfing' in hist.columns:
        add_plots.append(mpf.make_addplot(hist['Bullish engulfing'], label='Bullish engulfing', type='scatter', markersize=100, marker='^', color='green', edgecolors='black', panel = 0))
    if 'Bearish engulfing' in hist.columns:
        add_plots.append(mpf.make_addplot(hist['Bearish engulfing'], label='Bearish engulfing', type='scatter', markersize=100, marker='v', color='red', edgecolors='black', panel = 0))
    if 'RSI Buy' in hist.columns:
        add_plots.append(mpf.make_addplot(hist['RSI Buy'], label='RSI Buy', type='scatter', markersize=100, marker='^', color='green', edgecolors='black', panel = 1))
    if 'RSI Sell' in hist.columns:
        add_plots.append(mpf.make_addplot(hist['RSI Sell'], label='RSI Buy', type='scatter', markersize=100, marker='^', color='green', edgecolors='black', panel = 1))
    if 'RSI Support' in hist.columns:
        add_plots.append(mpf.make_addplot(hist['RSI Support'], label='RSI Support', type='scatter', markersize=100, marker='_', color='green', panel = 1))
    if 'RSI Resistance' in hist.columns:
        add_plots.append(mpf.make_addplot(hist['RSI Resistance'], label='RSI Resistance', type='scatter', markersize=100, marker='_', color='red', panel = 1))
    if 'Stochastic Buy' in hist.columns:
        add_plots.append(mpf.make_addplot(hist['Stochastic Buy'], label='Stochastic Buy', type='scatter', markersize=100, marker='^', color='green', edgecolors='black', panel = 2))
    if 'Stochastic Sell' in hist.columns:
        add_plots.append(mpf.make_addplot(hist['Stochastic Sell'], label='Stochastic Sell', type='scatter', markersize=100, marker='v', color='red', edgecolors='black', panel = 2))
    #if 'MACD Buy' in hist.columns:
    #    add_plots.append(mpf.make_addplot(hist['MACD Buy'], label='MACD Buy', type='scatter', markersize=100, marker='^', color='green', edgecolors='black', panel = 3))
    
    return hist, add_plots, labels
