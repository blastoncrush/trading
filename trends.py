from pytrends.request import TrendReq
import pandas as pd
from googleapiclient.discovery import build
from datetime import datetime, timedelta
import math
import time
    
def get_trend(keywords):
    # Initialiser la requête
    pytrends = TrendReq(hl='fr-FR', tz=360)

    # Définir le mot-clé et la période
    kw_list = keywords
    pytrends.build_payload(kw_list, cat=0, timeframe='today 5-y', geo='', gprop='')

    # Obtenir les tendances de recherche
    trends_data = pytrends.interest_over_time()
    for kw in kw_list:
        res = trends_data[kw].to_list()
        del res[-1]
        #log_diff = [math.log(res[i+1]) - math.log(res[i]) for i in range(len(res)-1)]
        print(kw, ":", res)
    return res
    
def get_views(keyword):
    # Configuration de l'API
    api_key = "AIzaSyC6pukndxjanzIEkRTudPfQ5dE6q0CuITE"
    youtube = build("youtube", "v3", developerKey=api_key)

    # last_week = (datetime.now() - timedelta(days=7)).isoformat("T") + "Z"
    video_ids = []
    next_page_token = None
    
    for _ in range(1000):
        # Recherche de vidéos publiées cette semaine avec le mot-clé
        response = youtube.videos().list(
            part="id,snippet",
            chart="mostPopular",
            regionCode = "US",
            pageToken=next_page_token
        ).execute()
        next_page_token = response.get('nextPageToken')
        # Récupération des IDs de vidéos et statistiques
        
        for item in response['items']:
            title = item['snippet']['title'].lower()
            description = item['snippet']['description'].lower()
            
            if keyword.lower() in title or keyword.lower() in description:
                video_ids.append(item['id'])
            
    # Calcul du nombre total de vues
    # total_views = sum(int(video['statistics']['viewCount']) for video in video_response['items'])
    return video_ids

fear_factors = ["gold price", "gold prices", "price of gold", "gold", "gold price", "gold prices", "depression", "crisis", "gdp", "unemployment", "inflation rate", "bankruptcy", "charity", "frugal", "price of gold", "economy", "great depression", "stock market crash", "vix"]
neutral_factos = ["dow jones", "s&p 500", "interest rate", "oil price", "live stock market", "stock market index", "stock market news", "stock market", "stock markets", "asian stock markets"]

for elem in neutral_factos:
    with open(f"keyword/{elem}.txt", "w") as f:
        f.write(" ".join([str(val) for val in get_trend([elem])]))
# get_views("ethereum")


