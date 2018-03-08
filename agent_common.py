from common import *
import json,hashlib
from websocket import create_connection

GLOBAL_GAME  = None

# GLOBAL_HAND  = 
# GLOBAL_BOARD = 
# PR_WIN       = 
# PR_WIN_STD   =

def build_game_info(data,table,name_md5=None):
    global GLOBAL_GAME
    GLOBAL_GAME  = pd.DataFrame(columns=('chips','pot','reld','bet','pos','cards','act','amt'))
    for x in data:
        idx  = 'Me' if x['playerName']==name_md5 else x['playerName']
        GLOBAL_GAME.loc[idx,'chips']  = x['chips']
        GLOBAL_GAME.loc[idx,'pot']    = x['roundBet']
        GLOBAL_GAME.loc[idx,'reld']   = x['reloadCount']
        GLOBAL_GAME.loc[idx,'bet']    = x['bet']
        GLOBAL_GAME.loc[idx,'pos']    = 'SB' if x['playerName']==table['smallBlind']['playerName'] else ('BB' if x['playerName']==table['bigBlind']['playerName'] else '')
        GLOBAL_GAME.loc[idx,'cards']  = "[%s]" % (' '.join(pkr_to_str(x['cards']) if 'cards' in x else []) if not x['folded'] else 'Folded')
    GLOBAL_GAME['act'] = ''
    GLOBAL_GAME['amt'] = ''

def update_game_info(data,table,action=None,name_md5=None):
    global GLOBAL_GAME
    for x in data:
        idx  = 'Me' if x['playerName']==name_md5 else x['playerName']
        GLOBAL_GAME.loc[idx,'chips']  = x['chips']
        GLOBAL_GAME.loc[idx,'pot']    = x['roundBet']
        GLOBAL_GAME.loc[idx,'reld']   = x['reloadCount']
        GLOBAL_GAME.loc[idx,'bet']    = x['bet']
        GLOBAL_GAME.loc[idx,'pos']    = 'SB' if x['playerName']==table['smallBlind']['playerName'] else ('BB' if x['playerName']==table['bigBlind']['playerName'] else '')
        GLOBAL_GAME.loc[idx,'cards']  = "[%s]" % (' '.join(pkr_to_str(x['cards']) if 'cards' in x else []) if not x['folded'] else 'Folded')
        if action is not None and x['playerName']==action['playerName']:
            GLOBAL_GAME.loc[idx,'act'] = action['action']
            GLOBAL_GAME.loc[idx,'amt'] = action['amount'] if 'amount' in action else ''

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
    pot_std_hat = np.std(pot_hat)
    pot_hat     = np.mean(pot_hat)
    print(time.clock() - t0)
    return pot_hat,pot_std_hat

def pkr_to_str(pkr):
    # Trend micro poker platform format to string
    return [suitmap[x[1].lower()]+(x[0] if x[0]!='T' else '10') for x in pkr]

def pkr_to_cards(pkr):
    # Trend micro poker platform format to pkrprb format
    cards  = [((suitmap[x[1].lower()],rankmap[x[0].lower()]),suitmap[x[1].lower()],rankmap[x[0].lower()]) for x in pkr]
    return pd.DataFrame(cards,columns=('c','s','o'))

def doListen(url,name,action):
    # try:
    global GLOBAL_GAME
    ws  = create_connection(url)
    ws.send(json.dumps({
        'eventName': '__join',
        'data': {'playerName': name,},
        }))
    m    = hashlib.md5()
    m.update(name.encode('utf8'))
    name_md5 = m.hexdigest()
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
        elif event_name in ('__new_round','__deal','__round_end','__show_action'):
            print("Table %s: Round %s: Board [%s]: Event %s" % (data['table']['tableNumber'],data['table']['roundName'],' '.join(pkr_to_str(data['table']['board'])),event_name))
            if GLOBAL_GAME is None:
                build_game_info(data['players'],data['table'],name_md5)
            else:
                update_game_info(data['players'],data['table'],data['action'] if 'action' in data else None,name_md5)
            print(GLOBAL_GAME)
            print()
        elif event_name == '__game_over':
            print("Table %s: Game Over" % data['table']['tableNumber'])
        # elif event_name not in ('__start_reload',): #'__bet','__action',
        #     print(event_name,'\n',json.dumps(data,indent=4))
        #
        action(ws,event_name,data)
    # except Exception as e:
    #     print("Exception: ",e)
    #     doListen(url,name,action)
