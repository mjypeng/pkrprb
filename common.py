import sys,time
import pandas as pd
import numpy as np

suitmap   = {'s':'♠','h':'♥','d':'♦','c':'♣'}
rankmap   = {'2':'2','3':'3','4':'4','5':'5','6':'6','7':'7','8':'8','9':'9','t':'10','j':'J','q':'Q','k':'K','a':'A'}

ordermap  = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'J':11,'Q':12,'K':13,'A':14}
ordermap2 = {'A':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'J':11,'Q':12,'K':13}
orderinvmap = {1:'A',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'10',11:'J',12:'Q',13:'K',14:'A'}

def new_deck():
    return pd.Series([x+y for x in ['♠','♥','♦','♣'] for y in ['2','3','4','5','6','7','8','9','10','J','Q','K','A']])

def draw(deck,n=1):
    cards = deck.sample(n)
    deck.drop(cards.index,'rows',inplace=True)
    return cards.tolist()

def str_to_cards(x):
    cards  = []
    for i in range(0,len(x),2):
        cards.append(suitmap[x[i]]+rankmap[x[i+1]])
    return cards

def straight(orders):
    o  = orders.sort_values(ascending=False)
    for i in range(len(o)-4):
        if (o.iloc[i:i+5].diff().iloc[1:]==-1).all():
            return o.iloc[i:i+5].index
    return None

def straight_flush(cards):
    c  = cards.s.value_counts()
    if c.max() >= 5:
        cards   = cards[cards.s==c.idxmax()]
        s       = straight(cards.o)
        if s is None and 14 in cards.o.values:
            cards.loc[cards.o==14,'o'] = 1
            cards.sort_values('o',ascending=False,inplace=True)
            s      = straight(cards.o)
        if s is not None:
            hand   = cards.loc[s]
            score  = 8 + hand.iloc[0].o/100 # Straight Flush
            hand   = hand.c.tolist()
        else:
            if 1 in cards.o.values:
                cards.loc[cards.o==1,'o'] = 14
                cards.sort_values('o',ascending=False,inplace=True)
            hand   = cards.iloc[:5]
            score  = 5 + hand.iloc[0].o/100 + hand.iloc[1].o/(100**2) + hand.iloc[2].o/(100**3) + hand.iloc[3].o/(100**4) + hand.iloc[4].o/(100**5) # Flush
            hand   = hand.c.tolist()
    else:
        s  = straight(cards.o)
        if s is None and 14 in cards.o.values:
            cards.loc[cards.o==14,'o'] = 1
            cards.sort_values('o',ascending=False,inplace=True)
            s      = straight(cards.o)
        if s is not None:
            hand   = cards.loc[s]
            score  = 4 + hand.iloc[0].o/100 # Straight
            hand   = hand.c.tolist()
        else:
            score  = 0
            hand   = None
    #
    return score,hand

def four_of_a_kind(cards):
    c  = cards.o.value_counts().reset_index().sort_values(['o','index'],ascending=False).set_index('index').o
    if c.iloc[0] == 4:
        kicker = cards[cards.o!=c.index[0]].iloc[0]
        score  = 7 + c.index[0]/100 + kicker.o/(100**2) # Four of a kind
        hand   = cards[cards.o==c.index[0]].c.tolist() + [kicker.c]
    elif c.iloc[0] == 3:
        if c.iloc[1] >= 2:
            score = 6 + c.index[0]/100 + c.index[1]/(100**2) # Full house
            hand  = cards[cards.o==c.index[0]].c.tolist() + cards[cards.o==c.index[1]].c.tolist()
        else:
            kickers = cards[cards.o!=c.index[0]].iloc[:2]
            score   = 3 + c.index[0]/100 + kickers.iloc[0].o/(100**2) + kickers.iloc[1].o/(100**3) # Three of a kind
            hand    = cards[cards.o==c.index[0]].c.tolist() + kickers.c.tolist()
    elif c.iloc[0] == 2:
        if c.iloc[1] == 2:
            kicker  = cards[~cards.o.isin(c.index[:2])].iloc[0]
            score   = 2 + c.index[0]/100 + c.index[1]/(100**2) + kicker.o/(100**3) # Two pairs
            hand    = cards[cards.o==c.index[0]].c.tolist() + cards[cards.o==c.index[1]].c.tolist() + [kicker.c]
        else:
            kickers = cards[cards.o!=c.index[0]].iloc[:3]
            score   = 1 + c.index[0]/100 + kickers.iloc[0].o/(100**2) + kickers.iloc[1].o/(100**3) + kickers.iloc[2].o/(100**4) # One pairs
            hand    = cards[cards.o==c.index[0]].c.tolist() + kickers.c.tolist()
    else:
        score  = 0 + cards.iloc[0].o/100 + cards.iloc[1].o/(100**2) + cards.iloc[2].o/(100**3) + cards.iloc[3].o/(100**4) + cards.iloc[4].o/(100**5) # No pairs
        hand   = cards.iloc[:5].c.tolist()
    #
    return score,hand

def score_hand(cards):
    cards      = pd.DataFrame(cards,columns=('c',))
    cards['s'] = cards.c.str[0]
    cards['o'] = cards.c.str[1:].apply(lambda x:ordermap[x])
    cards.sort_values('o',ascending=False,inplace=True)
    score1,hand1 = straight_flush(cards)
    score2,hand2 = four_of_a_kind(cards)
    return (score1,hand1) if score1 > score2 else (score2,hand2)

N     = int(sys.argv[1]) if len(sys.argv)>1 else 9
cond  = str_to_cards(sys.argv[2].lower() if len(sys.argv)>2 else '')

t0      = time.clock()
Nsamp   = 10000
results = pd.DataFrame(columns=('score','rank','pot'))
for j in range(Nsamp):
    deck  = new_deck()
    #
    if len(cond) < 2:
        hole  = draw(deck,2)
    else:
        hole  = cond[:2]
        deck  = deck[~deck.isin(hole)]
    #
    if len(cond) < 5:
        flop  = draw(deck,3)
    else:
        flop  = cond[2:5]
        deck  = deck[~deck.isin(flop)]
    #
    if len(cond) < 6:
        turn  = draw(deck)
    else:
        turn  = cond[5:6]
        deck  = deck[~deck.isin(turn)]
    #
    if len(cond) < 7:
        river = draw(deck)
    else:
        river = cond[6:7]
        deck  = deck[~deck.isin(river)]
    #
    holes_op = []
    for i in range(N-1):
        cards = draw(deck,2)
        holes_op.append(cards)
    #
    score  = score_hand(hole + flop + turn + river)
    resj   = pd.DataFrame(columns=('score','hand'))
    resj.loc['you','score'] = score[0]
    resj.loc['you','hand']  = score[1]
    for i in range(N-1):
        scoresi = score_hand(holes_op[i] + flop + turn + river)
        resj.loc[i,'score'] = scoresi[0]
        resj.loc[i,'hand']  = scoresi[1]
    #
    results.loc[j,'score'] = resj.score['you']
    results.loc[j,'rank']  = (resj.score>resj.score['you']).sum() + 1
    if resj.score['you'] == resj.score.max():
        Nrank1  = (resj.score==resj.score.max()).sum()
        results.loc[j,'pot'] = 1/Nrank1
    else:
        results.loc[j,'pot'] = 0
    #
    if np.any(resj.score>=8):
        print("Game %d" % j)
        print('Community: ' + ' '.join(flop+turn+river))
        print("You:  [%s] ==> %.10f, [%s]" % (' '.join(hole),resj.score['you'],' '.join(resj.hand['you'])))
        for i in range(N-1):
            print("Op %d: [%s] ==> %.10f, [%s]" % (i+1,' '.join(holes_op[i]),resj.score[i],' '.join(resj.hand[i])))
        print()

print(time.clock() - t0)


# ['♠A', '♥A'], Nsamp = 10000
#            mean       std
# score  2.388511  1.521516
# rank   1.530200  0.715639
# pot    0.581532  0.492212

