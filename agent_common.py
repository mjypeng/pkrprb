from common import *
import json,hashlib
from websocket import create_connection
from datetime import datetime

ws  = None

def init_game_state(players,table,name_md5=None):
    state  = pd.DataFrame(players)
    state['cards']  = state.cards.apply(pkr_to_str)
    state.set_index('playerName',inplace=True)
    state.loc[table['smallBlind']['playerName'],'position'] = 'SB'
    state.loc[table['bigBlind']['playerName'],'position']   = 'BB'
    state['action'] = np.nan
    state['amount'] = np.nan
    state['me']     = state.index==name_md5
    print(state)
    return state

def update_game_state(state,players,table,action=None):
    for x in players:
        idx  = x['playerName']
        for col in ('allIn','bet','chips','folded','isHuman','isOnline','isSurvive','reloadCount','roundBet'):
            state.loc[idx,col]  = x[col]
        state.loc[idx,'cards'] = pkr_to_str(x['cards']) if 'cards' in x else ''
    if action is not None:
        idx  = action['playerName']
        state.loc[idx,'action'] = action['action']
        state.loc[idx,'amount'] = action['amount'] if 'amount' in action else np.nan

def record_round_results(state,players):
    result  = state.copy()
    result.drop(['action','amount'],'columns',inplace=True)
    for x in players:
        idx  = x['playerName']
        if 'hand' in x:
            result.loc[idx,'hand']     = pkr_to_str(x['hand']['cards'])
            result.loc[idx,'rank']     = x['hand']['rank']
            result.loc[idx,'message']  = x['hand']['message']
        result.loc[idx,'winMoney'] = x['winMoney']
    return result

def record_game_results(state,winners):
    result  = state.copy()
    result.drop(['action','amount'],'columns',inplace=True)
    for x in winners:
        idx  = x['playerName']
        result.loc[idx,'win']   = True
        result.loc[idx,'score'] = x['chips']
    return result

def pkr_to_str(pkr):
    # Trend micro poker platform format to string
    return ' '.join([suitmap[x[1].lower()]+(x[0] if x[0]!='T' else '10') for x in pkr])

def pkr_to_cards(pkr):
    # Trend micro poker platform format to pkrprb format
    cards  = [((suitmap[x[1].lower()],rankmap[x[0].lower()]),suitmap[x[1].lower()],rankmap[x[0].lower()]) for x in pkr]
    return pd.DataFrame(cards,columns=('c','s','o'))

def read_win_prob(N,hole):
    res  = pd.read_csv("sim_N10_h[%s].csv" % cards_to_str(hole))
    return res.pot.mean(),res.pot.std()

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
    if not pre_river: river = board.iloc[4:5]
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
        score  = score_hand(pd.concat([hole,flop,turn,river]))
        resj   = pd.Series() #pd.DataFrame(columns=('score','hand'))
        resj.loc['you'] = score[0]
        for i in range(N-1):
            scoresi = score_hand(pd.concat([holes_op[(2*i):(2*i+2)],flop,turn,river]))
            resj.loc[i] = scoresi[0]
        #
        if resj.loc['you'] == resj.max():
            Nrank1  = (resj==resj.max()).sum()
            pot_hat[j] = (1/Nrank1) if Nrank1>1 else 1
        else:
            pot_hat[j] = 0
    #
    print(time.clock() - t0)
    return pot_hat.mean(),pot_hat.std()

#-- Agent Event Loop --#
def doListen(url,name,action,record=False):
    global ws
    ws  = create_connection(url)
    ws.send(json.dumps({
        'eventName': '__join',
        'data': {'playerName': name,},
        }))
    #
    if record:
        m   = hashlib.md5()
        m.update(name.encode('utf8'))
        name_md5   = m.hexdigest()
        game_id    = None
        round_id   = 0
        game_board = None
        game_state = None
    while True:
        msg  = ws.recv()
        #
        t0         = time.time()
        msg        = json.loads(msg)
        event_name = msg['eventName']
        data       = msg['data']
        if event_name in ('__action','__bet'):
            resp,out  = action(event_name,data)
            ws.send(json.dumps({
                'eventName': '__action',
                'data': {
                    'playerName': name,
                    'action': resp[0],
                    'amount': resp[1],
                    }
                }))
            out['cputime'] = time.time() - t0
            #
            print("Action:")
            print(out)
            print()
            #
            if record:
                if game_board is None: game_board = data['game']['board']
                if game_state is None:
                    game_state = init_game_state(data['game']['players'],data['game'],name_md5=name_md5)
                else:
                    update_game_state(game_state,data['game']['players'],data['game'])
            #
        elif event_name == '__game_prepare':
            print("Table %s: Game starts in %d sec(s)"%(data['tableNumber'],data['countDown']))
        elif event_name == '__game_start':
            if record:
                game_id   = datetime.now().strftime('%Y%m%d%H%M%S')
                round_id  = 0
                print("Table %s: Game %s start!!!\n"%(data['tableNumber'],game_id))
            else:
                print("Table %s: Game start!!!\n"%data['tableNumber'])
        elif event_name == '__new_round':
            if record:
                round_id   += 1
                game_board  = data['table']['board']
                game_state  = init_game_state(data['players'],data['table'],name_md5=name_md5)
        elif event_name == '__deal':
            # Deal hole cards
            if record:
                game_board  = data['table']['board']
                if game_state is None:
                    game_state  = init_game_state(data['players'],data['table'],name_md5=name_md5)
                else:
                    update_game_state(game_state,data['players'],data['table'])
        elif event_name == '__show_action':
            # Player action
            if record:
                if game_board is None: game_board = data['table']['board']
                if game_state is None:
                    game_state  = init_game_state(data['players'],data['table'],name_md5=name_md5)
                update_game_state(game_state,data['players'],data['table'],data['action'])
        elif event_name == '__round_end':
            if record:
                if game_board is None: game_board = data['table']['board']
                if game_state is None:
                    game_state  = init_game_state(data['players'],data['table'],name_md5=name_md5)
                else:
                    update_game_state(game_state,data['players'],data['table'])
                result  = record_round_results(game_state,data['players'])
                result.to_csv("game_%s_round_%d.csv"%(game_id,round_id))
        elif event_name == '__game_over':
            if record:
                if game_board is None: game_board = data['game']['board']
                if game_state is None:
                    game_state  = init_game_state(data['players'],data['table'],name_md5=name_md5)
                else:
                    update_game_state(game_state,data['players'],data['table'])
                result  = record_game_results(game_state,data['winners'])
                result.to_csv("game_%s.csv"%game_id)
        else:
            print("event received: %s\n" % event_name)
        #
        if event_name in ('__new_round','__deal','__show_action','__round_end','__game_over'):
            if record:
                #-- Output Game State --#
                print("Table %s: Game %s:\nRound %d-%s: Board [%s]: Event %s" % (data['table']['tableNumber'],game_id,round_id,data['table']['roundName'],pkr_to_str(game_board),event_name))
                output  = game_state.copy()
                output.index = ['Me' if name_md5==x else x for x in output.index]
                output.loc[output.allIn,'action'] = 'allin'
                output.loc[output.folded,'cards'] = 'fold'
                print(output[['chips','reloadCount','roundBet','bet','position','cards','action','amount']].rename(columns={'reloadCount':'reld','roundBet':'pot','position':'pos','action':'act','amount':'amt'}).fillna(''))
                print()
            else:
                print("event received: %s\n" % event_name)
    # except Exception as e:
    #     print("Exception: ",e)
    #     doListen(url,name,action)
