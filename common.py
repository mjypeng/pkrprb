import sys,time
import pandas as pd
import numpy as np

suitmap   = {'s':'♠','h':'♥','d':'♦','c':'♣'}
rankmap   = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'t':10,'j':11,'q':12,'k':13,'a':14}

ordermap  = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'J':11,'Q':12,'K':13,'A':14}
ordermap2 = {'A':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'J':11,'Q':12,'K':13}
orderinvmap = {1:'A',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'10',11:'J',12:'Q',13:'K',14:'A'}

def new_deck():
    return pd.DataFrame([((x,y),x,y) for x in ['♠','♥','♦','♣'] for y in range(2,15)],columns=('c','s','o'))

def draw(deck,n=1):
    cards = deck.sample(n)
    deck.drop(cards.index,'rows',inplace=True)
    return cards

def str_to_cards(x):
    cards  = []
    for i in range(0,len(x),2):
        cards.append(((suitmap[x[i]],rankmap[x[i+1]]),suitmap[x[i]],rankmap[x[i+1]]))
    return pd.DataFrame(cards,columns=('c','s','o'))

def cards_to_str(cards):
    return ' '.join([x+orderinvmap[y] for x,y in cards])

def straight(orders):
    o  = orders.sort_values(ascending=False).diff()
    for i in range(len(o)-4):
        if (o.iloc[i:i+5].iloc[1:]==-1).all():
            return o.iloc[i:i+5].index
    return None

def straight_flush(cards):
    cards = cards.copy()
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
            score  = (8,hand.iloc[0].o) # Straight Flush
            hand   = hand.c.tolist()
        else:
            if 1 in cards.o.values:
                cards.loc[cards.o==1,'o'] = 14
                cards.sort_values('o',ascending=False,inplace=True)
            hand   = cards.iloc[:5]
            score  = (5,hand.iloc[0].o,hand.iloc[1].o,hand.iloc[2].o,hand.iloc[3].o,hand.iloc[4].o) # Flush
            hand   = hand.c.tolist()
    else:
        s  = straight(cards.o)
        if s is None and 14 in cards.o.values:
            cards.loc[cards.o==14,'o'] = 1
            cards.sort_values('o',ascending=False,inplace=True)
            s      = straight(cards.o)
        if s is not None:
            hand   = cards.loc[s]
            score  = (4,hand.iloc[0].o) # Straight
            hand   = hand.c.tolist()
        else:
            score  = (0,)
            hand   = None
    #
    return score,hand

def four_of_a_kind(cards):
    c  = cards.o.value_counts().reset_index().sort_values(['o','index'],ascending=False).set_index('index').o
    if c.iloc[0] == 4:
        kicker = cards[cards.o!=c.index[0]].iloc[0]
        score  = (7,c.index[0],kicker.o) # Four of a kind
        hand   = cards[cards.o==c.index[0]].c.tolist() + [kicker.c]
    elif c.iloc[0] == 3:
        if c.iloc[1] >= 2:
            score = (6,c.index[0],c.index[1]) # Full house
            hand  = cards[cards.o==c.index[0]].c.tolist() + cards[cards.o==c.index[1]].c.tolist()
        else:
            kickers = cards[cards.o!=c.index[0]].iloc[:2]
            score   = (3,c.index[0],kickers.iloc[0].o,kickers.iloc[1].o) # Three of a kind
            hand    = cards[cards.o==c.index[0]].c.tolist() + kickers.c.tolist()
    elif c.iloc[0] == 2:
        if c.iloc[1] == 2:
            kicker  = cards[~cards.o.isin(c.index[:2])].iloc[0]
            score   = (2,c.index[0],c.index[1],kicker.o) # Two pairs
            hand    = cards[cards.o==c.index[0]].c.tolist() + cards[cards.o==c.index[1]].c.tolist() + [kicker.c]
        else:
            kickers = cards[cards.o!=c.index[0]].iloc[:3]
            score   = (1,c.index[0],kickers.iloc[0].o,kickers.iloc[1].o,kickers.iloc[2].o) # One pairs
            hand    = cards[cards.o==c.index[0]].c.tolist() + kickers.c.tolist()
    else:
        score  = (0,cards.iloc[0].o,cards.iloc[1].o,cards.iloc[2].o,cards.iloc[3].o,cards.iloc[4].o) # No pairs
        hand   = cards.iloc[:5].c.tolist()
    #
    return score,hand

def score_hand(cards):
    cards.sort_values('o',ascending=False,inplace=True)
    score2,hand2 = four_of_a_kind(cards)
    score1,hand1 = straight_flush(cards)
    return (score1,hand1) if score1 > score2 else (score2,hand2)

N     = int(sys.argv[1]) if len(sys.argv)>1 else 9
cond  = str_to_cards(sys.argv[2].lower() if len(sys.argv)>2 else '')

draw_hole  = len(cond) < 2
draw_flop  = len(cond) < 5
draw_turn  = len(cond) < 6
draw_river = len(cond) < 7

deck0  = new_deck()
if not draw_hole:
    hole  = cond.iloc[:2]
    deck0  = deck0[~deck0.c.isin(hole.c)]
if not draw_flop:
    flop  = cond.iloc[2:5]
    deck0  = deck0[~deck0.c.isin(flop.c)]
if not draw_turn:
    turn  = cond.iloc[5:6]
    deck0  = deck0[~deck0.c.isin(turn.c)]
if not draw_river:
    river = cond.iloc[6:7]
    deck0  = deck0[~deck0.c.isin(river.c)]

t0      = time.clock()
Nsamp   = 10000
results = pd.DataFrame(columns=('score','rank','pot'))
for j in range(Nsamp):
    deck  = deck0.copy()
    #
    if draw_hole:
        hole  = draw(deck,2)
    #
    if draw_flop:
        flop  = draw(deck,3)
    #
    if draw_turn:
        turn  = draw(deck)
    #
    if draw_river:
        river = draw(deck)
    #
    holes_op = []
    for i in range(N-1):
        cards = draw(deck,2)
        holes_op.append(cards)
    #
    score  = score_hand(pd.concat([hole,flop,turn,river]))
    resj   = pd.DataFrame(columns=('score','hand'))
    resj.loc['you','score'] = score[0]
    resj.loc['you','hand']  = score[1]
    for i in range(N-1):
        scoresi = score_hand(pd.concat([holes_op[i],flop,turn,river]))
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
    if np.any(resj.score.str[0]>=8):
        print("Game %d" % j)
        print('Community: ' + cards_to_str(pd.concat([flop.c,turn.c,river.c])))
        print("You:  [%s] ==> %s, [%s]" % (cards_to_str(hole.c),str(resj.score['you']),cards_to_str(resj.hand['you'])))
        for i in range(N-1):
            print("Op %d: [%s] ==> %s, [%s]" % (i+1,cards_to_str(holes_op[i].c),str(resj.score[i]),cards_to_str(resj.hand[i])))
        print()

print(time.clock() - t0)
print()

print("N = %d, [%s], Nsamp = %d" % (N,cards_to_str(hole.c),Nsamp))
print(results.agg(['mean','std']).T)

# N = 5, ['♠A', '♥A'], Nsamp = 10000
#            mean       std
# score  2.388511  1.521516
# rank   1.530200  0.715639
# pot    0.581532  0.492212

# N = 5, ['♠A', '♠Q'], Nsamp = 10000
#           mean       std
# score  1.72137  1.469653
# rank   2.29050  1.238654
# pot    0.34463  0.469437
