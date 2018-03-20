import sys,time
import pandas as pd
import numpy as np
import scipy as sp

suitmap   = {'s':'♠','h':'♥','d':'♦','c':'♣'}
rankmap   = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'t':10,'j':11,'q':12,'k':13,'a':14}

ordermap  = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'J':11,'Q':12,'K':13,'A':14}
ordermap2 = {'A':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'J':11,'Q':12,'K':13}
orderinvmap = {1:'A',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'T',11:'J',12:'Q',13:'K',14:'A'}
hand_scores = pd.read_csv('hand_scores.csv',index_col=list(range(0,6))).score.apply(eval)
hand_scores.index = [orderinvmap[x[0]]+orderinvmap[x[1]]+orderinvmap[x[2]]+orderinvmap[x[3]]+orderinvmap[x[4]]+str(x[5]) for x in hand_scores.index]

def new_deck():
    return pd.DataFrame([((x,y),x,y) for x in ['♠','♥','♦','♣'] for y in range(2,15)],columns=('c','s','o'))

def str_to_cards(x):
    cards  = []
    for i in range(0,len(x),2):
        cards.append(((suitmap[x[i]],rankmap[x[i+1]]),suitmap[x[i]],rankmap[x[i+1]]))
    return pd.DataFrame(cards,columns=('c','s','o'))

def cards_to_str(cards):
    if type(cards)==pd.DataFrame: cards = cards.c
    return ' '.join([x+orderinvmap[y] for x,y in cards])

def straight(orders):
    o  = orders.drop_duplicates().copy()
    temp = o[o==14].copy()
    temp[:] = 1
    o  = pd.concat([o,temp])
    o.sort_values(ascending=False,inplace=True)
    od = o.diff()
    for i in range(len(o)-4):
        if (od.iloc[i+1:i+5]==-1).all():
            return o.iloc[i:i+5].index,o.iloc[i]
    return None

def straight_flush(cards):
    c  = cards.s.value_counts()
    if c.max() >= 5:
        cards   = cards[cards.s==c.idxmax()]
        s       = straight(cards.o)
        if s is not None:
            hand   = cards.loc[s[0]]
            score  = (8,s[1]) # Straight Flush
            hand   = hand.c.tolist()
        else:
            hand   = cards.iloc[:5]
            score  = (5,hand.iloc[0].o,hand.iloc[1].o,hand.iloc[2].o,hand.iloc[3].o,hand.iloc[4].o) # Flush
            hand   = hand.c.tolist()
    else:
        s  = straight(cards.o)
        if s is not None:
            hand   = cards.loc[s[0]]
            score  = (4,s[1]) # Straight
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

def cards_to_hash(cards):
    # 5 card hand
    return orderinvmap[cards.o.iloc[0]] + orderinvmap[cards.o.iloc[1]] + orderinvmap[cards.o.iloc[2]] + orderinvmap[cards.o.iloc[3]] + orderinvmap[cards.o.iloc[4]] + str(int(cards.s.nunique()==1))

def score_hand(cards):
    cards.sort_values('o',ascending=False,inplace=True)
    score2,hand2 = four_of_a_kind(cards)
    score1,hand1 = straight_flush(cards)
    return (score1,hand1) if score1 > score2 else (score2,hand2)

def score_hand2(cards): # 7 cards
    global hand_scores
    cards  = cards.sort_values('o',ascending=False)
    score  = (0,)
    idx_list = [
        [0,1,2,3,4],
        [0,1,2,3,5],
        [0,1,2,3,6],
        [0,1,2,4,5],
        [0,1,2,4,6],
        [0,1,2,5,6],
        [0,1,3,4,5],
        [0,1,3,4,6],
        [0,1,3,5,6],
        [0,1,4,5,6],
        [0,2,3,4,5],
        [0,2,3,4,6],
        [0,2,3,5,6],
        [0,2,4,5,6],
        [0,3,4,5,6],
        [1,2,3,4,5],
        [1,2,3,4,6],
        [1,2,3,5,6],
        [1,2,4,5,6],
        [1,3,4,5,6],
        [2,3,4,5,6],
        ]
    for idx in idx_list:
        chash  = cards_to_hash(cards.iloc[idx])
        if score < hand_scores.loc[chash]:
            score = hand_scores.loc[chash]
    return score

def compare_hands(score0,cards1):
    # return 1: score0 > score1
    # return 0: score0 = score1
    # return -1: score0 < score1
    suit  = cards1.s.value_counts()
    ordr  = cards1.o.value_counts()
    cards1.sort_values('o',ascending=False,inplace=True)
    if score0[0] == 0:
        # score0 is High card
        if suit.iloc[0] >= 5: return -1 # cards1 has flush
        elif straight(cards1.o) is not None: return -1 # cards1 has straight
        elif ordr.iloc[0] >= 2: return -1 # cards1 has pair
        else:
            # cards1 is High card
            score1  = (0,cards1.iloc[0].o,cards1.iloc[1].o,cards1.iloc[2].o,cards1.iloc[3].o,cards1.iloc[4].o)
    #
    elif score0[0] == 1:
        # score0 is one pair
        if suit.iloc[0] >= 5: return -1 # cards1 has flush
        elif straight(cards1.o) is not None: return -1 # cards1 has straight
        elif ordr.iloc[:2].sum() >= 4: return -1 # cards1 has two pairs or three of a kind
        elif ordr.iloc[0] == 2:
            # cards1 is One pair
            kickers1 = cards1[cards1.o!=ordr.index[0]].iloc[:3]
            score1   = (1,ordr.index[0],kickers1.iloc[0].o,kickers1.iloc[1].o,kickers1.iloc[2].o)
        else:
            return 1
    #
    elif score0[0] == 2:
        # score0 is two pairs
        if suit.iloc[0] >= 5: return -1 # cards1 has flush
        elif straight(cards1.o) is not None: return -1 # cards1 has straight
        elif ordr.iloc[0] >= 3: return -1 # cards1 has three of a kind
        elif ordr.iloc[:2].sum() >= 4:
            # cards1 is Two pairs
            kicker1  = cards1[~cards1.o.isin(ordr.index[:2])].iloc[0]
            score1   = (2,ordr.index[0],ordr.index[1],kicker1.o)
        else:
            return 1
    #
    elif score0[0] == 3:
        # score0 is three of a kind
        if suit.iloc[0] >= 5: return -1 # cards1 has flush
        elif straight(cards1.o) is not None: return -1 # cards1 has straight
        elif ordr.iloc[0] >= 4: return -1 # cards1 has four of a kind
        elif ordr.iloc[0] >= 3:
            if ordr.iloc[1] >= 2: return -1 # cards1 has full house
            else:
                kickers1 = cards1[cards1.o!=ordr.index[0]].iloc[:2]
                score1   = (3,ordr.index[0],kickers1.iloc[0].o,kickers1.iloc[1].o)
        else:
            return 1
    #
    elif score0[0] == 4:
        # score0 is straight
        if suit.iloc[0] >= 5: return -1 # cards1 has flush
        elif ordr.iloc[:2].sum() >= 5: return -1 # cards1 has four of a kind or full house
        else:
            s  = straight(cards1.o)
            if s is not None:
                score1  = (4,s[1])
            else:
                return 1
    #
    elif score0[0] == 5:
        # score0 is flush
        if ordr.iloc[:2].sum() >= 5: return -1 # cards1 has four of a kind or full house
        elif suit.iloc[0] >= 5:
            cards1  = cards1[cards1.s==suit.index[0]]
            s       = straight(cards1.o)
            if s is not None: return -1 # cards1 has straight flush
            else:
                # cards1 has flush
                score1  = (5,cards1.iloc[0].o,cards1.iloc[1].o,cards1.iloc[2].o,cards1.iloc[3].o,cards1.iloc[4].o)
        else:
            return 1
    #
    elif score0[0] == 6:
        # score0 is full house
        if ordr.iloc[0] >= 4: return -1 # cards1 has four of a kind
        elif suit.iloc[0] >= 5:
            cards1  = cards1[cards1.s==suit.index[0]]
            s       = straight(cards1.o)
            if s is not None: return -1 # cards1 has straight flush
            else:
                return 1 # cards1 has flush, so it's impossible to have full house
        elif ordr.iloc[:2].sum() >= 5:
            # cards1 has full house
            score1 = (6,ordr.index[0],ordr.index[1])
        else:
            return 1
    #
    elif score0[0] == 7:
        # score0 is four of a kind
        if suit.iloc[0] >= 5:
            cards1  = cards1[cards1.s==suit.index[0]]
            s       = straight(cards1.o)
            if s is not None: return -1 # cards1 has straight flush
            else:
                return 1 # cards1 has flush, so it's impossible to have four of a kind
        elif ordr.iloc[0] >= 4:
            # cards1 has four of a kind
            kicker1 = cards1[cards1.o!=ordr.index[0]].iloc[0]
            score1  = (7,ordr.index[0],kicker1.o)
        else:
            return 1
    #
    elif score0[0] == 8:
        # score0 is straight flush
        if suit.iloc[0] >= 5:
            cards1  = cards1[cards1.s==suit.index[0]]
            s       = straight(cards1.o)
            if s is not None:
                score1  = (8,s[1])
            else:
                return 1
        else:
            return 1
    #
    if score0 > score1: return 1
    elif score0 < score1: return -1
    else: return 0

def read_win_prob(N,hole):
    # "Normalize" two card combinations
    hole.sort_values('o',ascending=False,inplace=True)
    if hole.s.iloc[0] == hole.s.iloc[1]:
        hole['s'] = '♠'
    else:
        hole['s'] = ['♠','♥']
    hole['c'] = [(x,y) for x,y in hole[['s','o']].values]
    #
    res  = pd.concat([pd.read_csv("sim_prob/sim2_N10_h[%s].csv.gz" % cards_to_str(hole).replace(' ','')),pd.read_csv("sim_prob/sim3_N10_h[%s].csv.gz" % cards_to_str(hole).replace(' ',''))],ignore_index=True)
    #
    if N < 10:
        res['prWin'] = 0
        mask = res['rank'] <= 11 - N
        res.loc[mask,'prWin'] = 1
        for i in range(N-1):
            res.loc[mask,'prWin'] *= (10 - res.loc[mask,'rank'] - i)/(9 - i)
        return res.prWin.mean(),res.prWin.std()
    elif N == 10:
        return res.pot.mean(),res.pot.std()

def calculate_river_showdown_prob(N,hole,board):
    # There are 45 cards left to assign to opponent's holes, 990 combinations of 2 hole cards
    score,hand  = score_hand(pd.concat([hole,board]))
    if score[0] == 8:
        # We have Straight Flush
        board  = board[board.c.isin(hand)]
        if len(board) == 3:
            # hole**  board|||
            # large<-->small
            # 10 possible combinations
            # 1.  **||| ==> Win Pot
            # 2.  *|*|| ==> Win Pot
            # 3.  *||*| ==> Win Pot
            # 4.  *|||* ==> Win Pot
            # 5.  |**|| ==> Win Pot
            # 6.  |*|*| ==> Win Pot
            # 7.  |*||* ==> Win Pot
            # 8.  ||**| ==> Win Pot
            # 9.  ||*|* ==> Win Pot
            # 10. |||**
            #   10.1. Highest card >= K ==> Win Pot
            #   10.2. Highest card <= Q ==> Beaten by one combination 1/990
            return ((990 - (N-1))/990) if hole.o.max()<board.o.min() and board.o.max()<=12 else 1
        elif len(board) == 4:
            # hole*  board||||
            # large<-->small
            # 5 possible Combinations
            # 1. *|||| ==> Win Pot
            # 2. |*||| ==> Win Pot
            # 3. ||*|| ==> Win Pot
            # 4. |||*|
            #   4.1. Highest card >= K ==> Win Pot
            #   4.2. Highest card <= Q ==> Beaten by one combination 1/990
            # 5. ||||*
            #   5.1. Highest card >= A ==> Win Pot
            #   5.2. Highest card <= K ==> Beaten by 44 combinations 44/990
            board_max  = board.o.max()
            comb = sum(board.o>hole.o.iloc[0]) + 1
            if comb == 4 and board_max <= 12:
                return (990 - (N-1))/990
            elif comb == 5 and board_max <= 13:
                return (990 - (N-1)*44)/990
            else:
                return 1
        else: # len(board) == 5
            # ||||| Highest card >= A ==> Win Pot
            # ||||| Highest card <= K ==> Beaten by 44 combinations 44/990
            return (990 - (N-1)*44)/990 if board.o.max()<=13 else 1
    elif score[0] == 7:
        # We have Four of a Kind
        # hole   = hole[hole.c.isin(hand)]
        # s      = board.s.value_counts()
        # if len(board) == 3:

        #     # Check for Straight Flush
        #     board  = board[~board.c.isin(hand) | board.s==s.index[0]]
        #     if s.iloc[0] == 4:
        #     elif s.iloc[0] == 3:
        #     else:
        # elif len(board) == 4:
        # else: # len(board) == 5
        return 0
    elif score[0] == 6:
        return 0
    elif score[0] == 5:
        return 0
    elif score[0] == 4:
        return 0
    elif score[0] == 3:
        return 0
    elif score[0] == 2:
        return 0
    elif score[0] == 1:
        return 0
    else: # score[0] == 0
        return 0

if __name__ == '__main__':
    N     = int(sys.argv[1]) if len(sys.argv)>1 else 9
    cond  = str_to_cards(sys.argv[2].lower() if len(sys.argv)>2 else '')
    #
    pre_deal  = len(cond) < 2
    pre_flop  = len(cond) < 5
    pre_turn  = len(cond) < 6
    pre_river = len(cond) < 7
    #
    deck  = new_deck()
    deck  = deck[~deck.c.isin(cond.c)]
    if not pre_deal:  hole  = cond[:2]
    if not pre_flop:  flop  = cond[2:5]
    if not pre_turn:  turn  = cond[5:6]
    if not pre_river: river = cond[6:7]
    #
    t0      = time.clock()
    Nsamp   = 20000
    results = pd.DataFrame(columns=('score','rank','pot','winner','board'))
    for j in range(Nsamp):
        if pre_deal:
            cards  = deck.sample(2*N + 5)
            hole   = cards[:2]
            flop   = cards[2:5]
            turn   = cards[5:6]
            river  = cards[6:7]
            holes_op = cards[7:]
        elif pre_flop:
            cards  = deck.sample(5 + (N-1)*2)
            flop   = cards[:3]
            turn   = cards[3:4]
            river  = cards[4:5]
            holes_op = cards[5:]
        elif pre_turn:
            cards = deck.sample(2 + (N-1)*2)
            turn  = cards[:1]
            river = cards[1:2]
            holes_op = cards[2:]
        elif pre_river:
            cards = deck.sample(1 + (N-1)*2)
            river = cards[:1]
            holes_op = cards[1:]
        else:
            holes_op = deck.sample((N-1)*2)
        #
        score  = score_hand(pd.concat([hole,flop,turn,river]))
        resj   = pd.DataFrame(columns=('score','hand'))
        resj.loc['you','score'] = score[0]
        resj.loc['you','hand']  = score[1]
        for i in range(N-1):
            scoresi = score_hand(pd.concat([holes_op[(2*i):(2*i+2)],flop,turn,river]))
            resj.loc[i,'score'] = scoresi[0]
            resj.loc[i,'hand']  = scoresi[1]
        #
        results.loc[j,'score'] = resj.score['you']
        results.loc[j,'rank']  = (resj.score>resj.score['you']).sum() + 1
        if resj.score['you'] == resj.score.max():
            Nrank1  = (resj.score==resj.score.max()).sum()
            results.loc[j,'pot'] = (1/Nrank1) if Nrank1>1 else 1
        else:
            results.loc[j,'pot'] = 0
        results.loc[j,'winner'] = resj.score.max()
        results.loc[j,'board']  = cards_to_str(pd.concat([flop,turn,river])) #score_hand(pd.concat([flop,turn,river]))[0]
        #
        if np.any(resj.score.str[0]>=7):
            print("Game %d" % j)
            print('Community: ' + cards_to_str(pd.concat([flop.c,turn.c,river.c])))
            print("You:  [%s] ==> %s, [%s]" % (cards_to_str(hole.c),str(resj.score['you']),cards_to_str(resj.hand['you'])))
            for i in range(N-1):
                print("Op %d: [%s] ==> %s, [%s]" % (i+1,cards_to_str(holes_op.iloc[(2*i):(2*i+2)].c),str(resj.score[i]),cards_to_str(resj.hand[i])))
            print()
    #
    print(time.clock() - t0)
    print()
    #
    print("N = %d, [%s], Nsamp = %d" % (N,cards_to_str(hole.c),Nsamp))
    print(results.agg(['mean','std']).T)
    #
    results.to_csv("sim4_N%d_h[%s].csv.gz" % (N,cards_to_str(hole.c).replace(' ','')),index=False,encoding='utf-8-sig',compression='gzip')
