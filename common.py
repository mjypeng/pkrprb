import sys
import pandas as pd
import numpy as np

suitmap   = {'s':'♠','h':'♥','d':'♦','c':'♣'}
rankmap   = {'2':'2','3':'3','4':'4','5':'5','6':'6','7':'7','8':'8','9':'9','t':'10','j':'J','q':'Q','k':'K','a':'A'}
ordermap  = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'J':11,'Q':12,'K':13,'A':14}
ordermap2 = {'A':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'J':11,'Q':12,'K':13}
orderinvmap = {1:'A',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'10',11:'J',12:'Q',13:'K',14:'A'}

def str_to_cards(x):
    cards  = []
    for i in range(0,len(x),2):
        cards.append(suitmap[x[i]]+rankmap[x[i+1]])
    return cards

def straight(orders):
    orders.sort_values(ascending=False,inplace=True)
    for i in range(len(orders)-4):
        if (orders.iloc[i:i+5].diff().iloc[1:]==-1).all():
            return orders.iloc[i:i+5].index
    return None

def straight_flush(cards):
    print('Input:\n',cards,'\n')
    cards   = pd.Series(cards)
    suits   = cards.str[0]
    orders  = cards.str[1:].apply(lambda x:ordermap[x])
    c       = suits.value_counts()
    if c.max() >= 5:
        mask    = suits==c.idxmax()
        cards   = cards[mask]
        orders  = orders[mask]
        s       = straight(orders)
        if s is not None:
            return 8 + orders.loc[s].max()/100,cards.loc[s].tolist() # Straight Flush
        if 14 in orders.values:
            orders2  = cards.str[1:].apply(lambda x:ordermap2[x])
            s2       = straight(orders2)
            if s2 is not None:
                return 8 + orders2.loc[s2].max()/100,cards.loc[s2].tolist() # Straight Flush
        return 5 + orders.max()/100,cards.loc[orders.index[:5]].tolist() # Flush
    else:
        return 0 + orders.max()/100,None # No pairs

def four_of_a_kind(cards):
    cards   = pd.Series(cards)
    orders  = cards.str[1:].apply(lambda x:ordermap[x])
    c       = orders.value_counts().reset_index().sort_values([0,'index'],ascending=False).set_index('index')[0]
    if c.max() >= 4:
        return 7 + c.idxmax()/100,cards[orders==c.idxmax()].tolist() # Four of a kind
    # elif c.max() == 3:
    else:
        return 0 + orders.max()/100,None # No pairs




N     = int(sys.argv[1]) if len(sys.argv)>1 else 9
hole  = sys.argv[2].lower() if len(sys.argv)>2 else 'saha'
hole  = [
    suitmap[hole[0]]+rankmap[hole[1]],
    suitmap[hole[2]]+rankmap[hole[3]]]

deck  = pd.Series([x+y for x in ['♠','♥','♦','♣'] for y in ['2','3','4','5','6','7','8','9','10','J','Q','K','A']])
deck  = deck[~deck.isin(hole)]

# Simulation

holes_op = []
for i in range(N-1):
    cards = deck.sample(2)
    deck.drop(cards.index,'rows',inplace=True)
    holes_op.append(cards.tolist())

cards = deck.sample(3)
deck.drop(cards.index,'rows',inplace=True)
flop  = cards.tolist()

cards = deck.sample(1)
deck.drop(cards.index,'rows',inplace=True)
turn  = cards.tolist()

cards = deck.sample(1)
deck.drop(cards.index,'rows',inplace=True)
river = cards.tolist()
