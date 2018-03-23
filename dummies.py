#! /usr/bin/env python
# -*- coding:utf-8 -*-

from agent_common import *

server = sys.argv[1] if len(sys.argv)>1 else 'beta'
if server == 'battle':
    url  = 'ws://allhands2018-battle.dev.spn.a1q7.net:3001'
elif server == 'training':
    url  = 'ws://allhands2018-training.dev.spn.a1q7.net:3001'
elif server == 'preliminary':
    url  = 'ws://allhands2018.dev.spn.a1q7.net:3001/'
else:
    url  = 'ws://allhands2018-beta.dev.spn.a1q7.net:3001'

name = sys.argv[2]
mode = sys.argv[3] # 'random', 'basic'

def agent_basic(event,data):
    DETERMINISM  = 0.9
    if event not in ('__action','__bet'): return None
    #
    #-- Calculate Basic Stats and Win Probability --#
    #
    input_var  = pd.Series()
    input_var['N']   = len(data['game']['players'])
    input_var['Nnf'] = len([x for x in data['game']['players'] if not x['folded']])
    input_var['round'] = data['game']['roundName']
    input_var['first'] = event == '__bet'
    hole   = pkr_to_cards(data['self']['cards'])
    board  = pkr_to_cards(data['game']['board'])
    input_var['hole']  = cards_to_str(hole)
    input_var['board'] = cards_to_str(board)
    if input_var.round == 'Deal':
        try:
            input_var['prWin'],input_var['prWinStd'] = read_win_prob(input_var.N,hole)
        except:
            input_var['prWin'],input_var['prWinStd'] = calculate_win_prob(input_var.N,hole,Nsamp=20)
    else:
        input_var['prWin'],input_var['prWinStd'] = calculate_win_prob(input_var.N,hole,board,Nsamp=20)
    input_var['pot']    = sum([x['roundBet'] for x in data['game']['players']])
    input_var['maxBet'] = max([x['bet'] for x in data['game']['players']])
    input_var['minBet'] = data['self']['minBet']
    #
    input_var['util_fold']  = -data['self']['roundBet'] - data['self']['bet']
    input_var['util_call']  = input_var.prWin*input_var.pot + input_var.prWin*input_var.maxBet*input_var.Nnf - data['self']['roundBet'] - input_var.maxBet
    input_var['util_raise_coeff']  = input_var.prWin*input_var.Nnf - 1
    #
    # Worst case scenario utility (everyone but one folds, i.e. a dual)
    input_var['util_call2'] = input_var.prWin*input_var.pot + input_var.prWin*input_var.maxBet*2 - data['self']['roundBet'] - input_var.maxBet
    input_var['util_raise_coeff2'] = input_var.prWin*2 - 1
    #
    if input_var.minBet > 0:
        # Need to "pay" to stay in game
        if input_var.util_fold > input_var.util_call:
            resp  = takeAction([DETERMINISM,1-DETERMINISM,0,0])
        elif input_var.util_raise_coeff > 0:
            if input_var.prWin > 0.9:
                resp  = takeAction([0,1-DETERMINISM,0,0])
            elif input_var.prWin > 0.75:
                resp  = takeAction([0,1-DETERMINISM,DETERMINISM,input_var.pot + sum([x['bet'] for x in data['game']['players']])])
            else:
                resp  = takeAction([0,1-DETERMINISM,DETERMINISM,'raise'])
        else:
            resp  = takeAction([0,DETERMINISM,1-DETERMINISM,0])
    else:
        # Can stay in the game for free
        if input_var.util_raise_coeff > 0:
            if input_var.prWin > 0.9:
                resp  = takeAction([0,1-DETERMINISM,0,0])
            elif input_var.prWin > 0.75:
                resp  = takeAction([0,1-DETERMINISM,DETERMINISM,input_var.pot + sum([x['bet'] for x in data['game']['players']])])
            else:
                if np.random.random() < 0.5:
                    resp  = takeAction([0,1-DETERMINISM,DETERMINISM,'raise'])
                else:
                    resp  = takeAction([0,1-DETERMINISM,DETERMINISM,0])
        else:
            resp  = takeAction([0,DETERMINISM,1-DETERMINISM,0])
    #
    input_var['action'] = resp[0]
    input_var['amount'] = resp[1]
    return resp,input_var

def agent_random(event,data):
    if event not in ('__action','__bet'): return None
    bet   = int(4*data['game']['bigBlind']['amount']*np.random.random())
    resp  = takeAction([0.15,0.4,0.4,bet])
    input_var = pd.Series()
    input_var['action'] = resp[0]
    input_var['amount'] = resp[1]
    return resp,input_var

#-- Agent Event Loop --#
ws  = None
def doListen(url,name,action):
    global ws
    while True:
        if ws is not None:
            try:
                msg  = ws.recv()
                msg  = json.loads(msg)
            except Exception as e:
                print(e)
                ws.close()
                ws   = None
                msg  = ''
        while ws is None or len(msg) == 0:
            try:
                time.sleep(3)
                print('Rejoining ...')
                ws   = create_connection(url)
                ws.send(json.dumps({
                    'eventName': '__join',
                    'data': {'playerName': name,},
                    }))
                msg  = ws.recv()
                msg  = json.loads(msg)
            except Exception as e:
                print(e)
                if ws is not None: ws.close()
                ws   = None
                msg  = ''
        #
        t0         = time.time()
        event_name = msg['eventName']
        data       = msg['data']
        if event_name in ('__action','__bet'):
            resp   = action(event_name,data)
            ws.send(json.dumps({
                'eventName': '__action',
                'data': {
                    'playerName': name,
                    'action': resp[0][0],
                    'amount': resp[0][1],
                    }
                }))
            resp[1]['cputime']  = time.time() - t0
            #
            print("Action:")
            print(resp[1])
            print()
            #
        elif event_name == '__game_prepare':
            print("Table %s: Game starts in %d sec(s)"%(data['tableNumber'],data['countDown']))
        elif event_name == '__game_start':
            print("Table %s: Game start!!!\n"%data['tableNumber'])
        elif event_name == '__new_round':
            _  = action(event_name,data)
        elif event_name == '__deal':
            _  = action(event_name,data)
        elif event_name == '__show_action':
            _  = action(event_name,data)
        elif event_name == '__round_end':
            _  = action(event_name,data)
        elif event_name in ('__game_over','__game_stop'):
            _  = action(event_name,data)
        elif event_name == '__start_reload':
            resp   = action(event_name,data)
            if resp:
                ws.send(json.dumps({
                    'eventName': '__reload',
                    }))
                print("Action: Reload")
        elif event_name not in ('__left','__new_peer'):
            print("event received: %s\n" % event_name)

if __name__ == '__main__':
    if mode == 'basic':
        agent  = agent_basic
    else:
        agent  = agent_random
    doListen(url,name,agent)
