import sys,time
import pandas as pd
import numpy as np
import scipy as sp
import multiprocessing as mp

def takeAction(x):
    # x[0]: prob to fold
    # x[1]: prob to check/call
    # x[2]: prob to bet/raise
    # x[3]: amount to bet/raise, use string 'raise' to raise double last raise, use 0 to bet minimum amount
    # 1 - x[0] - x[1] - x[2] is prob to go allin
    samp  = np.random.random()
    if samp < x[0]:
        return ('fold',0)
    elif samp < x[0] + x[1]:
        return ('check',0)
    elif samp < x[0] + x[1] + x[2]:
        if x[3] == 'raise':
            return ('raise',0)
        else:
            return ('bet',int(x[3]))
    else:
        return ('allin',0)

#-------------------------#
#-- Game State Features --#
#-------------------------#
def position_feature(N,pos):
    if pos in (1,2,):
        return 'E'
    elif pos in (N-3,'D',):
        return 'L'
    elif pos in ('SB','BB',):
        return 'B'
    else:
        return 'M'

def position_feature_batch(N,pos):
    pos  = pos.copy()
    mask = ~pos.isin(('D','SB','BB'))
    pos[mask] = pos[mask].astype(int)
    pos[pos.isin((1,2,))]      = 'E'
    pos[(pos==N-3)|(pos=='D')] = 'L'
    pos[pos.isin(('SB','BB'))] = 'B'
    mask = ~pos.isin(('E','L','B'))
    pos[mask] = 'M'
    return pos

def opponent_response_code(x):
    # Assume x is a pandas.Series
    if x.prev_action == 'bet/raise/allin':
        return 'any_reraised'
    elif x.NRraise > 0: return 'any_raised'
    elif x.NRcall > 0:  return 'any_called'
    elif x.NRfold > 0:  return 'all_folded'
    else: return 'none'

def opponent_response_code_batch(X):
    # Assume X is a pandas.DataFrame
    y  = pd.Series('none',index=X.index)
    y[X.NRfold > 0] = 'all_folded'
    y[X.NRcall > 0] = 'any_called'
    y[X.NRraise > 0] = 'any_raised'
    y[X.prev_action == 'bet/raise/allin'] = 'any_reraised'
    return y

#-----------------------#
#-- Utility Functions --#
#-----------------------#
suitmap   = {'s':'♠','h':'♥','d':'♦','c':'♣'}
suitcolor = {'s':'\033[37m','h':'\033[91m','d':'\033[91m','c':'\033[37m'}
rankmap   = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'t':10,'j':11,'q':12,'k':13,'a':14}

ordermap  = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'J':11,'Q':12,'K':13,'A':14}
ordermap2 = {'A':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'J':11,'Q':12,'K':13}
orderinvmap = {1:'A',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'10',11:'J',12:'Q',13:'K',14:'A'}

suitinvmap   = {'♠':'S','♥':'H','♦':'D','♣':'C'}
orderinvmap2 = {1:'A',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'T',11:'J',12:'Q',13:'K',14:'A'}

def str_to_cards(x):
    cards  = []
    for i in range(0,len(x),2):
        cards.append(((suitmap[x[i]],rankmap[x[i+1]]),suitmap[x[i]],rankmap[x[i+1]]))
    return pd.DataFrame(cards,columns=('c','s','o'))

def cards_to_str(cards):
    if type(cards)==pd.DataFrame: cards = cards.c
    elif len(cards) == 0: return ''
    return ' '.join([x+orderinvmap[y] for x,y in cards])

def cards_to_pkr(cards):
    if type(cards)==pd.DataFrame: cards = cards.c
    elif len(cards) == 0: return ''
    return [orderinvmap2[y]+suitinvmap[x] for x,y in cards]

def pkr_to_str(pkr,color=False):
    # Trend micro poker platform format to string
    if color:
        return ' '.join([suitcolor[x[1].lower()]+suitmap[x[1].lower()]+(x[0] if x[0]!='T' else '10')+'\033[0m' for x in pkr])
    else:
        return ' '.join([suitmap[x[1].lower()]+(x[0] if x[0]!='T' else '10') for x in pkr])

def pkr_to_cards(pkr):
    # Trend micro poker platform format to pkrprb format
    cards  = [((suitmap[x[1].lower()],rankmap[x[0].lower()]),suitmap[x[1].lower()],rankmap[x[0].lower()]) for x in pkr]
    return pd.DataFrame(cards,columns=('c','s','o'))

def pkr_to_hash(N,cards,board):
    hole   = ''.join(sorted(cards.split()))
    board  = board.split()
    board  = ''.join(sorted(board[:3]) + board[3:])
    return "%d_%s_%s" % (N,hole,board)

#---------------------------#
#-- Hand Texture Features --#
#---------------------------#
# Hole Cards Category
# Category 1: AA, KK
# Category 2: QQ, AKs, AKo, JJ
# Category 3: AQs, AQo, TT, 99
# Category 4: AJs, KQs, 88, 77
# Category 5: AJo, ATs, ATo, KQo, KJs, 66, 55
# Category 6: A9s-A2s, KJo, KTs, QJs, QTs, JTs, 44, 33, 22
# Category 7: A9-A2, KTo, QJo, QTo, JTo, T9s, 98s, 87s, 76s, 65s, 54s
# Category 8: K9s, K9o, K8s, K8o, Q9s, Q8s, J9s, T8s, T9o, 97s, 98o, 86s, 87o, 75s, 76o, 64s

def hole_texture(hole):
    c  = hole.lower().split()
    o  = [rankmap[cc[0]] for cc in c]
    o_max = max(o)
    o_min = min(o)
    s  = [cc[1] for cc in c]
    #
    X  = pd.Series()
    X['cards_rank1'] = o_max
    X['cards_rank2'] = o_min
    X['cards_rank_sum']  = o_max + o_min
    X['cards_aces']  = o_max == 14
    X['cards_faces'] = o_min >= 10
    X['cards_pair']  = o_max == o_min
    X['cards_suit']  = s[0] == s[1]
    X['cards_conn']  = ((o_max-o_min)==1) & (o_max<=12) & (o_min>=4)
    X['cards_conn2'] = ((o_max-o_min)==2) & (o_max<=12) & (o_min>=4)
    #
    return X

def hole_texture_batch(cards):
    X  = pd.DataFrame(index=cards.index)
    c  = cards.str.lower().str.split()
    c1 = c.str[0]
    c2 = c.str[1]
    o  = np.c_[c1.str[0].apply(lambda x:rankmap[x]),c2.str[0].apply(lambda x:rankmap[x])]
    s1 = c1.str[1]
    s2 = c2.str[1]
    o_max = o.max(1)
    o_min = o.min(1)
    #
    X['cards_rank1'] = o_max
    X['cards_rank2'] = o_min
    X['cards_rank_sum']  = o_max + o_min
    X['cards_aces']  = o_max == 14
    X['cards_faces'] = o_min >= 10
    X['cards_pair']  = o_max == o_min
    X['cards_suit']  = s1 == s2
    X['cards_conn']  = ((o_max-o_min)==1) & (o_max<=12) & (o_min>=4)
    X['cards_conn2'] = ((o_max-o_min)==2) & (o_max<=12) & (o_min>=4)
    #
    return X

def hole_texture_to_category(x):
    # Assume x is a pandas.Series
    if x.cards_pair and x.cards_rank2>=13:
        return 1
    elif (x.cards_pair and x.cards_rank2>=11) or x.cards_rank2>=13:
        return 2
    elif (x.cards_pair and x.cards_rank2>=9) or x.cards_rank2>=12:
        return 3
    elif (x.cards_pair and x.cards_rank2>=7) or (x.cards_suit and x.cards_rank_sum>=25):
        return 4
    elif (x.cards_pair and x.cards_rank2>=5) or x.cards_rank_sum + (x.cards_aces or x.cards_suit) >= 25:
        return 5
    elif x.cards_pair or x.cards_rank_sum>=24 or (x.cards_suit and (x.cards_aces or x.cards_faces)):
        return 6
    elif x.cards_aces or x.cards_faces or (x.cards_conn and x.cards_suit):
        return 7
    elif ((x.cards_rank1 + x.cards_suit)>=13 and x.cards_rank2>=8) or (x.cards_conn and x.cards_rank2>=6) or (x.cards_suit and x.cards_conn2 and x.cards_rank2>=4):
        return 8
    else:
        return 9

def hole_texture_to_category_batch(X):
    # Assume X is a pandas.DataFrame
    y  = pd.Series(9,index=X.index)
    #
    mask    = (((X.cards_rank1+X.cards_suit)>=13) & (X.cards_rank2>=8)) | (X.cards_conn & (X.cards_rank2>=6)) | (X.cards_suit & X.cards_conn2 & (X.cards_rank2>=4))
    y[mask] = 8
    #
    mask    = X.cards_aces | X.cards_faces | (X.cards_conn & X.cards_suit)
    y[mask] = 7
    #
    mask    = X.cards_pair | (X.cards_rank_sum>=24) | (X.cards_suit & (X.cards_aces | X.cards_faces))
    y[mask] = 6
    #
    mask    = (X.cards_pair & (X.cards_rank2>=5)) | ((X.cards_rank_sum + (X.cards_aces | X.cards_suit)) >= 25)
    y[mask] = 5
    #
    mask    = (X.cards_pair & (X.cards_rank2>=7)) | (X.cards_suit & (X.cards_rank_sum>=25))
    y[mask] = 4
    #
    mask    = (X.cards_pair & (X.cards_rank2>=9)) | (X.cards_rank2>=12)
    y[mask] = 3
    #
    mask    = (X.cards_pair & (X.cards_rank2>=11)) | (X.cards_rank2>=13)
    y[mask] = 2
    #
    mask    =  X.cards_pair & (X.cards_rank2>=13)
    y[mask] = 1
    #
    return y

def board_texture(board):
    c    = board.lower().split()
    o,s  = zip(*[(rankmap[cc[0]],cc[1]) for cc in c])
    o,s  = np.asarray(o),np.asarray(s)
    #
    X    = pd.Series()
    X['board_rank1']     = max(o)
    X['board_rank2']     = min(o)
    X['board_aces']      = sum(o==14)
    X['board_faces']     = sum(o>=10)
    #
    ou,oc  = np.unique(o,return_counts=True)
    ou,oc  = ou[::-1],oc[::-1]
    idx    = oc.argmax()
    X['board_kind']      = oc[idx]
    X['board_kind_rank'] = ou[idx]
    #
    su,sc  = np.unique(s,return_counts=True)
    idx    = sc.argmax()
    X['board_suit']      = sc[idx]
    X['board_suit_rank'] = o[s==su[idx]].max()
    #
    X['board_conn']      = 0
    X['board_conn_rank'] = 0
    #
    if 14 in ou: ou = np.r_[ou,1] # Aces can also serve as 1
    dou  = np.diff(ou)==-1 # Mask for if the adjacent cards are connected
    if dou.any():
        if dou.all():
            X['board_conn']      = len(ou)
            X['board_conn_rank'] = ou[0]
        elif len(dou) == 2:
            X['board_conn']      = 2
            X['board_conn_rank'] = ou[0] if dou[0] else ou[1]
        elif len(dou) == 3:
            if dou[0]:
                X['board_conn']      = 3 if dou[1] else 2
                X['board_conn_rank'] = ou[0]
            elif dou[1]:
                X['board_conn']      = 3 if dou[2] else 2
                X['board_conn_rank'] = ou[1]
            else:
                X['board_conn']      = 2
                X['board_conn_rank'] = ou[2]
        else: # if len(dou) == 4:
            if dou[0]:
                if dou[1]:
                    X['board_conn']      = 4 if dou[2] else 3
                    X['board_conn_rank'] = ou[0]
                elif dou[2] and dou[3]:
                    X['board_conn']      = 3
                    X['board_conn_rank'] = ou[2]
                else:
                    X['board_conn']      = 2
                    X['board_conn_rank'] = ou[0]
            elif dou[1]:
                X['board_conn']      = (4 if dou[3] else 3) if dou[2] else 2
                X['board_conn_rank'] = ou[1]
            elif dou[2]:
                X['board_conn']      = 3 if dou[3] else 2
                X['board_conn_rank'] = ou[2]
            else:
                X['board_conn']      = 2
                X['board_conn_rank'] = ou[3]
    #
    return X

#----------------------------------------------------#
#-- Monte Carlo Simulation of Hand Win Probability --#
#----------------------------------------------------#
def new_deck():
    return pd.DataFrame([((x,y),x,y) for x in ['♠','♥','♦','♣'] for y in range(2,15)],columns=('c','s','o'))

def straight(orders):
    temp  = np.sort(np.unique(orders))
    if 14 in temp: temp = np.r_[1,temp]
    dtemp = np.diff(temp)
    for i in range(len(dtemp)-4,-1,-1):
        if (dtemp[i:i+4]==1).all():
            return temp[i+4]
    return None

def straight_flush(cards):
    suit,counts  = np.unique(cards.s,return_counts=True)
    flush        = counts.max() >= 5
    if flush:
        cards   = cards[cards.s==suit[counts.argmax()]]
        s       = straight(cards.o)
        if s is not None:
            score  = (8,s) # Straight Flush
        else:
            hand   = np.sort(cards.o)[::-1]
            score  = (5,hand[0],hand[1],hand[2],hand[3],hand[4]) # Flush
    else:
        s  = straight(cards.o)
        if s is not None:
            score  = (4,s) # Straight
        else:
            hand   = np.sort(cards.o)[::-1]
            score  = (0,hand[0],hand[1],hand[2],hand[3],hand[4]) # High card
    #
    return score

def four_of_a_kind(cards):
    c  = list(reversed(sorted(list(zip(*reversed(np.unique(cards.o,return_counts=True)))))))
    if c[0][0] == 4:
        kicker = cards[cards.o!=c[0][1]].o.max()
        score  = (7,c[0][1],kicker) # Four of a kind
    elif c[0][0] == 3:
        if c[1][0] >= 2:
            score = (6,c[0][1],c[1][1]) # Full house
        else:
            kickers = np.sort(cards[cards.o!=c[0][1]].o)[:-3:-1]
            score   = (3,c[0][1],kickers[0],kickers[1]) # Three of a kind
    elif c[0][0] == 2:
        if c[1][0] == 2:
            kicker  = cards[~cards.o.isin((c[0][1],c[1][1]))].o.max()
            score   = (2,c[0][1],c[1][1],kicker) # Two pairs
        else:
            kickers = np.sort(cards[cards.o!=c[0][1]].o)[:-4:-1]
            score   = (1,c[0][1],kickers[0],kickers[1],kickers[2]) # One pairs
    else:
        score  = (0,)
    #
    return score

def score_hand(cards):
    score1  = straight_flush(cards)
    score2  = four_of_a_kind(cards)
    return score1 if score1 > score2 else score2

def cards_to_hash(cards):
    # For 5 cards
    return tuple(sorted(cards.o)) + (int(np.unique(cards.s).shape[0]==1),)

hand_scores5  = pd.read_csv('hand_scores.csv',index_col=list(range(6))).score.apply(eval)
def score_hand5(cards):
    global hand_scores5
    return hand_scores5[cards_to_hash(cards)]

def compare_hands(score0,cards1):
    # return 1: score0 > score1
    # return 0: score0 = score1
    # return -1: score0 < score1
    suit1,counts  = np.unique(cards1.s,return_counts=True)
    flush1        = counts.max() >= 5
    if flush1 and score0[0] < 5: return -1
    straight1     = straight(cards1.o)
    if straight1 is not None and score0[0] < 4: return -1
    #
    ordr  = list(reversed(sorted(list(zip(*reversed(np.unique(cards1.o,return_counts=True)))))))
    if score0[0] == 0:
        # score0 is High card
        if ordr[0][0] >= 2: return -1 # cards1 has pair
        else:
            # cards1 is High card
            hand1   = np.sort(cards1.o)[::-1]
            score1  = (0,hand1[0],hand1[1],hand1[2],hand1[3],hand1[4])
    #
    elif score0[0] == 1:
        # score0 is one pair
        if ordr[0][0] + ordr[1][0] >= 4: return -1 # cards1 has two pairs or three of a kind
        elif ordr[0][0] == 2:
            # cards1 is One pair
            kickers1 = np.sort(cards1[cards1.o!=ordr[0][1]].o)[:-4:-1]
            score1   = (1,ordr[0][1],kickers1[0],kickers1[1],kickers1[2])
        else:
            return 1
    #
    elif score0[0] == 2:
        # score0 is two pairs
        if ordr[0][0] >= 3: return -1 # cards1 has three of a kind
        elif ordr[0][0] + ordr[1][0] >= 4:
            # cards1 is Two pairs
            kicker1  = cards1[~cards1.o.isin((ordr[0][1],ordr[1][1]))].o.max()
            score1   = (2,ordr[0][1],ordr[1][1],kicker1)
        else:
            return 1
    #
    elif score0[0] == 3:
        # score0 is three of a kind
        if ordr[0][0] >= 4: return -1 # cards1 has four of a kind
        elif ordr[0][0] >= 3:
            if ordr[1][0] >= 2: return -1 # cards1 has full house
            else:
                kickers1 = np.sort(cards1[cards1.o!=ordr[0][1]].o)[:-3:-1]
                score1   = (3,ordr[0][1],kickers1[0],kickers1[1])
        else:
            return 1
    #
    elif score0[0] == 4:
        # score0 is straight
        if ordr[0][0] + ordr[1][0] >= 5: return -1 # cards1 has four of a kind or full house
        else:
            s  = straight(cards1.o)
            if s is not None:
                score1  = (4,s)
            else:
                return 1
    #
    elif score0[0] == 5:
        # score0 is flush
        if ordr[0][0] + ordr[1][0] >= 5: return -1 # cards1 has four of a kind or full house
        elif flush1:
            cards1  = cards1[cards1.s==suit1[counts.argmax()]]
            s       = straight(cards1.o)
            if s is not None: return -1 # cards1 has straight flush
            else:
                # cards1 has flush
                hand1   = np.sort(cards1.o)[::-1]
                score1  = (5,hand1[0],hand1[1],hand1[2],hand1[3],hand1[4]) # High card
        else:
            return 1
    #
    elif score0[0] == 6:
        # score0 is full house
        if ordr[0][0] >= 4: return -1 # cards1 has four of a kind
        elif flush1:
            cards1  = cards1[cards1.s==suit1[counts.argmax()]]
            s       = straight(cards1.o)
            if s is not None: return -1 # cards1 has straight flush
            else:
                return 1 # cards1 has flush, so it's impossible to have full house
        elif ordr[0][0] + ordr[1][0] >= 5:
            # cards1 has full house
            score1 = (6,ordr[0][1],ordr[1][1])
        else:
            return 1
    #
    elif score0[0] == 7:
        # score0 is four of a kind
        if flush1:
            cards1  = cards1[cards1.s==suit1[counts.argmax()]]
            s       = straight(cards1.o)
            if s is not None: return -1 # cards1 has straight flush
            else:
                return 1 # cards1 has flush, so it's impossible to have four of a kind
        elif ordr[0][0] >= 4:
            # cards1 has four of a kind
            kicker1 = cards1[cards1.o!=ordr[0][1]].o.max()
            score1  = (7,ordr[0][1],kicker1)
        else:
            return 1
    #
    elif score0[0] == 8:
        # score0 is straight flush
        if flush1:
            cards1  = cards1[cards1.s==suit1[counts.argmax()]]
            s       = straight(cards1.o)
            if s is not None:
                score1  = (8,s)
            else:
                return 1
        else:
            return 1
    #
    if score0 > score1: return 1
    elif score0 < score1: return -1
    else: return 0

deal_win_prob = pd.read_csv('deal_win_prob.csv',index_col='hole')
def read_win_prob_(N,hole):
    # "Normalize" two card combinations
    hole.sort_values('o',ascending=False,inplace=True)
    if hole.s.iloc[0] == hole.s.iloc[1]:
        hole['s'] = '♠'
    else:
        hole['s'] = ['♠','♥']
    hole['c'] = [(x,y) for x,y in hole[['s','o']].values]
    #
    res  = pd.read_csv("sim_prob/sim_N10_h[%s].csv.gz" % cards_to_str(hole).replace(' ',''))
    #
    if N < 10:
        res['prWin'] = 0
        mask = res['rank'] <= 11 - N
        res.loc[mask,'prWin'] = 1
        for i in range(N-1):
            res.loc[mask,'prWin'] *= (10 - res.loc[mask,'rank'] - i)/(9 - i)
        return len(res),res.prWin.mean(),res.prWin.std()
    elif N == 10:
        return len(res),res.pot.mean(),res.pot.std()

def read_win_prob(N,hole):
    # "Normalize" two card combinations
    hole.sort_values('o',ascending=False,inplace=True)
    if hole.s.iloc[0] == hole.s.iloc[1]:
        hole['s'] = '♠'
    else:
        hole['s'] = ['♠','♥']
    hole['c'] = [(x,y) for x,y in hole[['s','o']].values]
    #
    res   = deal_win_prob.loc[cards_to_str(hole)]
    prWin = res["%d"%N]
    #
    return res.Nsim,prWin,np.sqrt(prWin*(1-prWin))

def calculate_win_prob(N,hole,board=(),Nsamp=100):
    deck  = new_deck()
    deck  = deck[~deck.c.isin(hole.c)]
    if len(board) > 0:
        deck  = deck[~deck.c.isin(board.c)]
    #
    pre_flop  = len(board) < 3
    pre_turn  = len(board) < 4
    pre_river = len(board) < 5
    #
    if not pre_flop:  flop  = board.iloc[:3]
    if not pre_turn:  turn  = board.iloc[3:4]
    if not pre_river:
        river  = board.iloc[4:5]
        score  = score_hand(pd.concat([hole,flop,turn,river]))
    #
    t0      = time.clock()
    pot_hat = np.zeros(Nsamp)
    for j in range(Nsamp):
        if pre_flop:
            cards = deck.sample(5 + (N-1)*2)
            flop  = cards[:3]
            turn  = cards[3:4]
            river = cards[4:5]
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
        if pre_river:
            score  = score_hand(pd.concat([hole,flop,turn,river]))
        #
        Nrank1     = 1
        pot_hat[j] = 1
        for i in range(N-1):
            resi  = compare_hands(score,pd.concat([holes_op[(2*i):(2*i+2)],flop,turn,river]))
            if resi < 0: # score[0] < scorei
                pot_hat[j] = 0
                break
            elif resi == 0: # score[0] == scorei
                Nrank1 += 1
            # scoresi = score_hand(pd.concat([holes_op[(2*i):(2*i+2)],flop,turn,river]))
            # if score[0] < scoresi[0]:
            #     pot_hat[j] = 0
            #     break
            # elif score[0] == scoresi[0]:
            #     Nrank1 += 1
        #
        if pot_hat[j] > 0:
            pot_hat[j] = (1/Nrank1) if Nrank1>1 else 1
    #
    print(time.clock() - t0)
    return pot_hat.mean(),pot_hat.std()

#----------------------------------------------#
#-- Multiprocess Win Probability Calculation --#
#----------------------------------------------#
pc  = []
pq  = None
prWin_samples = []

def calculate_win_prob_mp(q,N,hole,board=()):
    deck  = new_deck()
    deck  = deck[~deck.c.isin(hole.c)]
    if len(board) > 0:
        deck  = deck[~deck.c.isin(board.c)]
    #
    pre_flop  = len(board) < 3
    pre_turn  = len(board) < 4
    pre_river = len(board) < 5
    #
    if not pre_flop:  flop  = board.iloc[:3]
    if not pre_turn:  turn  = board.iloc[3:4]
    if not pre_river:
        river  = board.iloc[4:5]
        score  = score_hand(pd.concat([hole,flop,turn,river]))
    #
    hole_str  = cards_to_str(hole)
    board_str = cards_to_str(board)
    while not q.full():
        if pre_flop:
            cards = deck.sample(5 + (N-1)*2)
            flop  = cards[:3]
            turn  = cards[3:4]
            river = cards[4:5]
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
        if pre_river:
            score  = score_hand(pd.concat([hole,flop,turn,river]))
        #
        Nrank1   = 1
        pot_hatj = 1
        for i in range(N-1):
            resi  = compare_hands(score,pd.concat([holes_op[(2*i):(2*i+2)],flop,turn,river]))
            if resi < 0: # score[0] < scorei
                pot_hatj = 0
                break
            elif resi == 0: # score[0] == scorei
                Nrank1 += 1
        #
        if pot_hatj > 0:
            pot_hatj = (1/Nrank1) if Nrank1>1 else 1
        #
        q.put({
            'N':     N,
            'hole':  hole_str,
            'board': board_str,
            'prWin': pot_hatj,
            },block=True,timeout=None)

def calculate_win_prob_mp_start(N,hole,board=(),n_jobs=1):
    global pc
    global pq
    global prWin_samples
    #
    for pcc in pc:
        if pcc.is_alive(): pcc.terminate()
    #
    prWin_samples = [] #pd.DataFrame(columns=('N','hole','board','prWin'))
    pq  = mp.Queue(maxsize=0)
    pc  = []
    for _ in range(n_jobs):
        pc.append(mp.Process(target=calculate_win_prob_mp,args=(pq,N,hole,board)))
        pc[-1].start()

def calculate_win_prob_mp_get():
    global pc
    global pq
    global prWin_samples
    #
    while not pq.empty(): prWin_samples.append(pq.get_nowait())
    return prWin_samples

def calculate_win_prob_mp_stop():
    global pc
    global pq
    global prWin_samples
    #
    for pcc in pc:
        if pcc.is_alive(): pcc.terminate()
    if pq is not None:
        try:
            while not pq.empty(): prWin_samples.append(pq.get_nowait())
        except:
            pass

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
    Nsamp   = 40000
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
        resj.loc['you','score'] = score
        resj.loc['you','hand']  = None
        for i in range(N-1):
            scoresi = score_hand(pd.concat([holes_op[(2*i):(2*i+2)],flop,turn,river]))
            resj.loc[i,'score'] = scoresi
            resj.loc[i,'hand']  = None
        #
        results.loc[j,'score'] = resj.score['you']
        results.loc[j,'rank']  = (resj.score>resj.score['you']).sum() + 1
        if resj.score['you'] == resj.score.max():
            Nrank1  = (resj.score==resj.score.max()).sum()
            results.loc[j,'pot'] = (1/Nrank1) if Nrank1>1 else 1
        else:
            results.loc[j,'pot'] = 0
        results.loc[j,'winner'] = resj.score.max()
        results.loc[j,'board']  = cards_to_str(pd.concat([flop,turn,river]))
        if np.any(resj.score.str[0]>=7):
            print("Game %d" % j)
            print('Community: ' + cards_to_str(pd.concat([flop.c,turn.c,river.c])))
            print("You:  [%s] ==> %s" % (cards_to_str(hole.c),str(resj.score['you'])))
            for i in range(N-1):
                print("Op %d: [%s] ==> %s" % (i+1,cards_to_str(holes_op.iloc[(2*i):(2*i+2)].c),str(resj.score[i])))
            print()
    #
    print(time.clock() - t0)
    print()
    #
    print("N = %d, [%s], Nsamp = %d" % (N,cards_to_str(hole.c),Nsamp))
    print(results.agg(['mean','std']).T)
    #
    results.to_csv("sim2_N%d_h[%s].csv.gz" % (N,cards_to_str(hole.c).replace(' ','')),index=False,encoding='utf-8-sig',compression='gzip')
