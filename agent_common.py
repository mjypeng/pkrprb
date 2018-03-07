from common import *
import json,hashlib
from websocket import create_connection

def calculate_win_prob(N,hole,board=(),Nsamp=100):
    deck0  = new_deck()
    deck0  = deck0[~deck0.c.isin(hole.c)]
    #
    draw_flop  = len(board) < 3
    draw_turn  = len(board) < 4
    draw_river = len(board) < 5
    #
    if not draw_flop:
        flop   = board.iloc[:3]
        deck0  = deck0[~deck0.c.isin(flop.c)]
    if not draw_turn:
        turn   = board.iloc[3:4]
        deck0  = deck0[~deck0.c.isin(turn.c)]
    if not draw_river:
        river  = board.iloc[4:5]
        deck0  = deck0[~deck0.c.isin(river.c)]
    #
    t0   = time.clock()
    pot_hat = []
    for j in range(Nsamp):
        deck  = deck0.copy()
        #
        if draw_flop:  flop  = draw(deck,3)
        if draw_turn:  turn  = draw(deck)
        if draw_river: river = draw(deck)
        holes_op = [draw(deck,2) for _ in range(N-1)]
        #
        score  = score_hand(pd.concat([hole,flop,turn,river]))
        resj   = pd.Series() #pd.DataFrame(columns=('score','hand'))
        resj.loc['you'] = score[0]
        for i in range(N-1):
            scoresi = score_hand(pd.concat([holes_op[i],flop,turn,river]))
            resj.loc[i] = scoresi[0]
        #
        if resj.loc['you'] == resj.max():
            Nrank1  = (resj==resj.max()).sum()
            pot_hat.append(1/Nrank1)
        else:
            pot_hat.append(0)
    #
    pot_hat  = np.mean(pot_hat)
    print(time.clock() - t0)
    return pot_hat

def pkr_to_str(pkr):
    # Trend micro poker platform format to string
    return [suitmap[x[1].lower()]+(x[0] if x[0]!='T' else '10') for x in pkr]

def pkr_to_cards(pkr):
    # Trend micro poker platform format to pkrprb format
    cards  = [((suitmap[x[1].lower()],rankmap[x[0].lower()]),suitmap[x[1].lower()],rankmap[x[0].lower()]) for x in pkr]
    return pd.DataFrame(cards,columns=('c','s','o'))

def doListen(url,name,action):
    # try:
    ws  = create_connection(url)
    ws.send(json.dumps({
        'eventName': '__join',
        'data': {'playerName': name,},
        }))
    while True:
        msg  = ws.recv()
        msg  = json.loads(msg)
        event_name = msg['eventName']
        data       = msg['data']
        if event_name == '__game_prepare':
            print("Table %s: Game starts in %d sec(s)"%(data['tableNumber'],data['countDown']))
        elif event_name == '__game_start':
            print("Table %s: Game start!!!"%data['tableNumber'])
            print()
        # elif event_name in ('__new_round','_join','__deal','__round_end'):
        #     print("Table %s: Round %s:"%(data['table']['tableNumber'],data['table']['roundName']))
        #     print("Board [%s]" % ' '.join(cards_to_str(data['table']['board'])))
        #     SB_name = data['table']['smallBlind']['playerName']
        #     BB_name = data['table']['bigBlind']['playerName']
        #     for x in data['players']:
        #         print("%32s %6d %6d %s [%s]" % (
        #             'Me' if x['playerName']==name_md5 else x['playerName'],
        #             x['chips'],
        #             x['bet'],
        #             'SB' if SB_name==x['playerName'] else ('BB' if BB_name==x['playerName'] else '  '),
        #             ' '.join(cards_to_str(x['cards'])) if not x['folded'] else 'Folded',
        #             ))
        #     print()
        elif event_name == '__show_action':
            print("Table %s: Round %s: Player %32s (%6d) %s%s" % (data['table']['tableNumber'],data['table']['roundName'],data['action']['playerName'],data['action']['chips'],data['action']['action']," with %d" % data['action']['amount'] if 'amount' in data['action'] else ''))
            print()
        elif event_name == '__game_over':
            print("Table %s Game Over" % (data['table']['tableNumber']))
        # elif event_name not in ('__start_reload',): #'__bet','__action',
        #     print(event_name,'\n',json.dumps(data,indent=4))
        #
        action(ws,event_name,data)
    # except Exception as e:
    #     print("Exception: ",e)
    #     doListen(url,name,action)
