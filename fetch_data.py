import os
import csv
import yfinance as yf
import pickle
import textwrap
from datetime import datetime

def fetch_data():
    with open('indicateurs.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
                ind, title = row
                ticker = yf.Ticker(ind)
                try:
                    hist = ticker.history(period="max")
                    file = open("./data/" + title, 'wb')
                    pickle.dump(hist, file)
                    file.close()
                except ValueError:
                    continue

def fetch_mass_data():
    run = False
    with open('YTS/Stock.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            row = row[0].split(';')
            ind, title = row[0], row[1]
            print(ind, title)
            if ind == "BUD":
                run = True
            if not run:
                continue
            ticker = yf.Ticker(ind)
            try:
                hist = ticker.history(period="max")
                file = open("./data/" + title, 'wb')
                pickle.dump(hist, file)
                file.close()
            except ValueError:
                continue

def clean_data():
    # Delete empty files in the data folder
    cleant = 0
    with open('YTS/Stock.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            row = row[0].split(';')
            ind, title = row[0], row[1]
            try:
                with open("./data/" + title, 'rb') as file:
                    hist = pickle.load(file)
            except EOFError:
                print(f"Erreur de lecture pour {ind}: {title}")
                continue
            except IsADirectoryError:
                print(f"Erreur de lecture pour {ind}: {title}")
                continue
            except FileNotFoundError:
                continue
            if hist.empty:
                os.remove("./data/" + title)
                cleant += 1
            elif len(hist) < 100:
                os.remove("./data/" + title)
                cleant += 1
    print(f"Files cleaned: {cleant}")

def print_news(symbol, title):
    # Fetch the news for the given ticker
    ticker = yf.Ticker(symbol)
    news = ticker.news

    # Check if news data is available
    if not news:
        print(f"No news found for {title}")
        return

    # Print the news in a fancy way
    for article in news:
        title = article.get('title', 'No title')
        link = article.get('link', 'No link')
        publisher = article.get('publisher', 'Unknown publisher')
        summary = article.get('summary', 'No summary')
        date = article.get('providerPublishTime', 'Unknown date')

        print("=" * 80)
        print(f"Title: {title}")
        print(f"Publisher: {publisher}")
        date = datetime.utcfromtimestamp(date)
        print(f"Date: {date.strftime('%d/%m/%Y %H:%M:%S UTC')}")
        print(f"Link: {link}")
        print("Summary:")
        print(textwrap.fill(summary, width=80))
        print("=" * 80)
        print("\n")

clean_data()