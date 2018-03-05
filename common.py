import pandas as pd
import numpy as np

deck    = pd.Series([(x,y) for x in ['♠','♥','♦','♣'] for y in range(2,15)])

N       = 2
holes   = []

for i in range(N):
    cards = deck.sample(2)
    deck.drop(cards.index,'rows',inplace=True)
    holes.append(cards.tolist())
