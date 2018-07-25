from common import *
from decision_logic import *
import json,hashlib
from websocket import create_connection
from datetime import datetime

pd.set_option('display.width',120)
pd.set_option('display.unicode.east_asian_width',True)

MP_JOBS  = 3

TABLE_STATE = pd.Series({
    'name': None,
    'name_md5': None,
    'tableNumber': None,
    'game_id':    None,
    'round_id':   0,
    'smallBlind': 0,
    'roundName':  None,
    'board':      None,
    'forced_bet': 0,
    })
GAME_STATE  = None
AGENT_LOG   = []
ACTION_LOG  = []
ROUND_LOG   = []
GAME_LOG    = []
PREV_AGENT_STATE   = None # Info on previous decision point and resulting actions

player_stats = None
expsmo_alpha = 0.1

def init_game_state(players,table):
    global TABLE_STATE
    global GAME_STATE
    #
    TABLE_STATE['roundName']  = table['roundName']
    TABLE_STATE['board']      = ' '.join(table['board'])
    TABLE_STATE['smallBlind'] = table['smallBlind']['amount']
    #
    game_state  = pd.DataFrame(players)
    game_state['cards']  = game_state.cards.fillna('').str.join(' ')
    game_state.set_index('playerName',inplace=True)
    game_state.loc[table['smallBlind']['playerName'],'position'] = 'SB'
    game_state.loc[table['bigBlind']['playerName'],'position']   = 'BB'
    #
    SB_idx = (game_state.position=='SB').values.argmax()
    BB_idx = (game_state.position=='BB').values.argmax()
    D_idx  = (SB_idx - 1) % len(players)
    while not game_state.loc[game_state.index[D_idx],'isSurvive'] and D_idx != BB_idx:
        D_idx = (D_idx - 1) % len(players)
    if D_idx not in (SB_idx,BB_idx):
        game_state.loc[game_state.index[D_idx],'position'] = 'D'
    idx  = (BB_idx + 1) % len(players)
    pos  = 1
    while idx not in (SB_idx,BB_idx,D_idx):
        if game_state.loc[game_state.index[idx],'isSurvive']:
            game_state.loc[game_state.index[idx],'position'] = pos
            pos  += 1
        idx  = (idx + 1) % len(players)
    #
    game_state['action'] = np.nan
    game_state['amount'] = np.nan
    game_state['me']     = game_state.index==TABLE_STATE['name_md5']
    #
    game_state[['Nfold','Ncall','Nraise']]    = 0 # Number of actions performed by player in this betting roundName
    game_state['prev_action']  = 'none' # Previous action of the player
    game_state[['NRfold','NRcall','NRraise']] = 0 # Number of actions performed by other players in response to the player's last action
    #
    GAME_STATE  = game_state

def update_game_state(players,table,action=None):
    global TABLE_STATE
    global GAME_STATE
    #
    TABLE_STATE['roundName']  = table['roundName']
    TABLE_STATE['board']      = ' '.join(table['board'])
    TABLE_STATE['smallBlind'] = table['smallBlind']['amount']
    #
    game_state  = GAME_STATE
    for x in players:
        idx  = x['playerName']
        for col in ('allIn','bet','chips','folded','isHuman','isOnline','isSurvive','reloadCount','roundBet'):
            game_state.loc[idx,col]  = x[col]
        game_state.loc[idx,'cards']  = ' '.join(x['cards']) if 'cards' in x else ''
        if 'winMoney' in x and x['winMoney'] > 0:
            game_state.loc[idx,'action']  = 'win'
            game_state.loc[idx,'amount']  = x['winMoney']
    #
    if action == 'reset_action_count':
        game_state[['Nfold','Ncall','Nraise']]    = 0 # Number of actions performed by player in this betting roundName
        game_state['prev_action']  = 'none' # Previous action of the player
        game_state[['NRfold','NRcall','NRraise']] = 0 # Number of actions performed by other players in response to the player's last action
    elif action is not None:
        # Event is '__show_action'
        idx  = action['playerName']
        game_state.loc[idx,'action'] = action['action']
        game_state.loc[idx,'amount'] = action['amount'] if 'amount' in action else np.nan
        #
        game_state.loc[idx,'prev_action']  = action['action']
        game_state.loc[idx,'Nfold']       += action['action']=='fold'
        game_state.loc[idx,'Ncall']       += action['action']=='check/call'
        game_state.loc[idx,'Nraise']      += action['action']=='bet/raise/allin'
        #
        mask  = game_state.index != idx
        game_state.loc[mask,'NRfold']  += action['action']=='fold'
        game_state.loc[mask,'NRcall']  += action['action']=='check/call'
        game_state.loc[mask,'NRraise'] += action['action']=='bet/raise/allin'

def record_action(action):
    action  = pd.Series(action)
    for col in ('tableNumber','game_id','round_id','smallBlind','roundName','board',):
        action[col]  = TABLE_STATE[col]
    idx  = action['playerName']
    for col in ('isHuman','isOnline','chips','reloadCount','position','cards','roundBet','bet',):
        action[col]  = GAME_STATE.loc[idx,col]
    if 'amount' not in action: action['amount'] = 0
    #
    action  = action[['tableNumber','game_id','round_id','smallBlind','roundName','playerName','isHuman','isOnline','chips','reloadCount','position','cards','board','roundBet','bet','action','amount']]
    ACTION_LOG.append(action)

def record_round(players):
    global TABLE_STATE
    global GAME_STATE
    global ROUND_LOG
    result  = GAME_STATE.copy()
    for col in ('tableNumber','game_id','round_id','smallBlind','board',):
        result[col]  = TABLE_STATE[col]
    for x in players:
        idx  = x['playerName']
        if 'hand' in x:
            # result.loc[idx,'hand']     = ' '.join(x['hand']['cards'])
            result.loc[idx,'rank']     = x['hand']['rank']
            result.loc[idx,'message']  = x['hand']['message']
        result.loc[idx,'winMoney'] = x['winMoney']
    #
    result  = result.reset_index()[['tableNumber','game_id','round_id','smallBlind','playerName','isHuman','isOnline','isSurvive','chips','reloadCount','folded','allIn','position','cards','board','rank','message','winMoney']]
    ROUND_LOG.append(result)

def record_game(winners):
    global TABLE_STATE
    global GAME_STATE
    global GAME_LOG
    #
    result  = GAME_STATE.copy()
    for col in ('tableNumber','game_id','smallBlind',):
        result[col]  = TABLE_STATE[col]
    result['roundCount']   = TABLE_STATE['round_id']
    result['chips_final']  = 0
    for x in winners:
        idx  = x['playerName']
        result.loc[idx,'chips_final']  = x['chips']
    result  = result.reset_index()[['tableNumber','game_id','smallBlind','playerName','chips']]
    GAME_LOG.append(result)

#-- Agent Event Handler --#
TIGHTNESS    = {   # Reference tightness for N=3
    'deal':  0.034, #-0.058, # tightness + (N - 3)*0.11 for N players
    'flop':  -0.248, #-0.156,
    'turn':  -0.298, #-0.132,
    'river': -0.345, #-0.195,
    }
# {'deal': -0.058, 'flop': -0.156, 'turn': -0.132, 'river': -0.195}
# {'deal': 0.034, 'flop': -0.248, 'turn': -0.298, 'river': -0.345}
AGGRESIVENESS = 0.75

LOGIC_LIST   = [
    # ('basic',basic_logic),
    # ('player4',player4_logic),
    ('michael',michael_logic),
    ('michael2',michael2_logic),
    ]
LOGIC        = 1
INIT_LOGIC_DECAY = 1.1 #0.98
LOGIC_DECAY  = INIT_LOGIC_DECAY

def agent_jyp(event,data):
    global TABLE_STATE
    global GAME_STATE
    global PREV_AGENT_STATE
    #
    global TIGHTNESS
    global AGGRESIVENESS
    #
    global LOGIC_LIST
    global LOGIC
    global INIT_LOGIC_DECAY
    global LOGIC_DECAY
    #
    print('Tightness:\n',TIGHTNESS,'\nAggressiveness: ',AGGRESIVENESS)
    if event in ('__action','__bet'):
        #
        #-- Calculate Basic Stats and Win Probability --#
        #
        #-- Game State Variables --#
        players  = GAME_STATE.copy()
        players  = players[players.isSurvive]
        state    = pd.Series()
        for col in ('tableNumber','game_id','round_id','smallBlind','roundName',,'forced_bet'):
            state[col]  = TABLE_STATE[col]
        #
        state['N']      = len(players)
        state['Nnf']    = state.N - players.folded.sum()
        state['Nallin'] = players.allIn.sum()
        state['first']  = event == '__bet'
        hole   = pkr_to_cards(data['self']['cards'])
        board  = pkr_to_cards(data['game']['board'])
        state['hole']  = ' '.join(data['self']['cards']) #cards_to_str(hole)
        state['board'] = ' '.join(data['game']['board']) #cards_to_str(board)
        #
        #-- Calculate Win Probability --#
        if state.roundName == 'Deal':
            state['Nsim'],state['prWin'],state['prWinStd'] = read_win_prob(state.N,hole)
        else:
            try:
                prWin_samples = calculate_win_prob_mp_get()
                prWin_samples = [x['prWin'] for x in prWin_samples]
                state['Nsim']     = len(prWin_samples)
                state['prWin']    = np.mean(prWin_samples)
                state['prWinStd'] = np.std(prWin_samples)
            except Exception as e:
                print(e)
                state['Nsim'] = 120
                state['prWin'],state['prWinStd'] = calculate_win_prob(state.N,hole,board,Nsamp=state['Nsim'])
        #
        state['tightness']  = TIGHTNESS[state.roundName] - (state.N - 3)*0.11 if state.roundName=='Deal' else TIGHTNESS[state.roundName]
        state['aggresiveness'] = AGGRESIVENESS
        state['prWin_adj']  = np.maximum(state.prWin - state.tightness*np.sqrt(state.prWin*(1-state.prWin)),0)
        #
        #-- Betting Variables --#
        state['chips']  = data['self']['chips']
        state['pot']    = data['self']['roundBet'] # self accumulated contribution to pot
        state['bet']    = data['self']['bet'] # self bet on this round
        state['minBet'] = data['self']['minBet']
        state['cost_to_call']  = min(state.minBet,state.chips) # minimum bet to stay in game
        state['pot_sum'] = players.roundBet.sum()
        state['bet_sum'] = np.minimum(players.bet,state.bet + state.cost_to_call).sum()
        state['NMaxBet'] = ((players.bet>0) & ~players.folded & (players.allIn | (players.bet==players.bet.max()))).sum()
        #
        players['cost_to_call'] = np.minimum(players.bet.max() - players.bet,players.chips)
        #
        #-- Decide bluffing freq. --#
        player_stats  = get_player_stats()
        if state.Nallin>0 or players[players.index!=TABLE_STATE['name_md5']].chips.max() > 0.8*state.chips:
            bluff_freq   = 0
        elif player_stats is not None:
            player_stats = player_stats.loc[players[players.isSurvive & ~players.folded & (players.index!=TABLE_STATE['name_md5'])].index]
            bluff_freq   = player_stats[(state.roundName,'prFold')].prod()
        else:
            if state.roundName == 'Deal':    prFold0 = 0.345
            elif state.roundName == 'Flop':  prFold0 = 0.325
            elif state.roundName == 'Turn':  prFold0 = 0.209
            elif state.roundName == 'River': prFold0 = 0.161
            bluff_freq   = prFold0**(players.isSurvive & ~players.folded & (players.index!=TABLE_STATE['name_md5'])).sum()
        state['bluff_freq']  = bluff_freq
        print("bluff_freq: %.3f%%"%(100*bluff_freq))
        #
        #-- Decision Logic --#
        #
        state['logic'] = LOGIC_LIST[LOGIC][0]
        resp  = LOGIC_LIST[LOGIC][1](state,PREV_STATE)
        state['resp']  = resp
        resp  = takeAction(resp)
        state['action'] = resp[0]
        state['amount'] = resp[1]
        #
        return resp,state
    elif event in ('__new_round','__deal'):
        if event == '__new_round':
            samp  = np.random.random()
            if samp > LOGIC_DECAY:
                LOGIC  = (LOGIC + 1) % len(LOGIC_LIST)
                LOGIC_DECAY  = INIT_LOGIC_DECAY
            else:
                LOGIC_DECAY *= LOGIC_DECAY
        #
        #-- Determine Effective Number of Players --#
        players  = pd.DataFrame(data['players']).set_index('playerName')
        if event == '__deal' and data['table']['roundName'] == 'Flop':
            # Flop cards are just dealt, anyone that folded preflop can be considered out of the game and not figured in win probability calculation
            N_EFFECTIVE  = players.isSurvive.sum() #(players.isSurvive & ~players.folded).sum()
        elif event == '__new_round':
            N_EFFECTIVE  = players.isSurvive.sum()
        else:
            N_EFFECTIVE  = players.isSurvive.sum()
        #
        #-- Start win probability calculation --#
        hole   = pkr_to_cards(players.loc[name_md5,'cards'])
        board  = pkr_to_cards(data['table']['board'])
        calculate_win_prob_mp_start(N_EFFECTIVE,hole,board,n_jobs=MP_JOBS)
        #
        #-- Update current small blind amount --#
        if event == '__new_round':
            if SMALL_BLIND != data['table']['smallBlind']['amount']:
                SMALL_BLIND  = data['table']['smallBlind']['amount']
                NUM_BLIND    = 1
            else:
                NUM_BLIND   += 1
            #
            if data['table']['smallBlind']['playerName']==name_md5:
                FORCED_BET  = data['table']['smallBlind']['amount']
            elif data['table']['bigBlind']['playerName']==name_md5:
                FORCED_BET  = data['table']['bigBlind']['amount']
            else:
                FORCED_BET  = 0
            #
            player_stats  = get_player_stats()
            if player_stats is not None and players.loc[name_md5,'isSurvive']:
                player_stats  = player_stats.loc[players[players.isSurvive].index]
                for rnd in ('deal','flop','turn','river'):
                    player_stats_rnd  = player_stats[rnd]
                    if player_stats_rnd.loc[name_md5,'rounds'] > 3:
                        if player_stats_rnd.loc[name_md5,'prFold'] >= player_stats_rnd.prFold.median():
                            TIGHTNESS[rnd] -= 0.002
                        else:
                            TIGHTNESS[rnd] += 0.002
                    if rnd == 'turn':
                        TIGHTNESS[rnd]  = 0.98*TIGHTNESS[rnd] + 0.02*TIGHTNESS['flop']
                    elif rnd == 'river':
                        TIGHTNESS[rnd]  = 0.98*TIGHTNESS[rnd] + 0.02*TIGHTNESS['turn']
    #
    elif event == '__show_action':
        # Record aggressors
        players  = pd.DataFrame(data['players']).set_index('playerName')
        players  = players[players.index!=data['action']['playerName']]
        if data['action']['action'] == 'allin':
            if data['action']['amount'] > players.bet.max():
                ROUND_AGGRESSORS.append(data['action']['playerName'])
        elif data['action']['action'] in ('bet','raise'):
            ROUND_AGGRESSORS.append(data['action']['playerName'])
    elif event in ('__round_end','__game_over','__game_stop',):
        calculate_win_prob_mp_stop()

def agent(event_name,data):
    state  = pd.Series()
    for col in ('tableNumber','game_id','round_id','smallBlind','roundName','turn_id','board',):
        state[col]  = TABLE_STATE[col]
    #
    resp  = ('raise',0)
    #
    state['action']  = resp[0]
    state['amount']  = resp[1]
    #
    return resp,state

#-- WebSocket Listen --#
def doListen(url):
    global TABLE_STATE
    global GAME_STATE
    global AGENT_LOG
    global ACTION_LOG
    global ROUND_LOG
    global GAME_LOG
    global player_stats
    global expsmo_alpha
    #
    ws  = None
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
                    'data': {'playerName': TABLE_STATE['name'],},
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
        #
        #-- Get and Check table number --#
        if 'tableNumber' in data:
            tblnum  = data['tableNumber']
        elif 'table' in data and 'tableNumber' in data['table']:
            tblnum  = data['table']['tableNumber']
        else:
            continue
        if TABLE_STATE['tableNumber'] is None or event_name == '__game_start':
            TABLE_STATE['tableNumber']  = tblnum
        elif TABLE_STATE['tableNumber'] != tblnum: continue
        #
        #-- Handle Event --#
        if event_name in ('__action','__bet'):
            if GAME_STATE is None:
                init_game_state(data['game']['players'],data['game'])
            else:
                update_game_state(data['game']['players'],data['game'])
            #
            out   = agent(event_name,data)
            resp  = out[0]
            log   = out[1]
            #
            ws.send(json.dumps({
                'eventName': '__action',
                'data': {
                    'playerName': TABLE_STATE['name'],
                    'action': resp[0],
                    'amount': resp[1],
                    }
                }))
            log['cputime']  = time.time() - t0
            AGENT_LOG.append(log)
            TABLE_STATE['forced_bet']  = 0
            PREV_AGENT_STATE  = log
        elif event_name == '__game_prepare':
            print("Table %s: Game starts in %d sec(s)"%(data['tableNumber'],data['countDown']))
        elif event_name == '__game_start':
            TABLE_STATE['tableNumber']  = data['tableNumber']
            TABLE_STATE['game_id']  = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
            TABLE_STATE['round_id'] = 0
            GAME_STATE    = None
            ROUND_RESULTS = []
            GAME_ACTIONS  = None
            print("Table %s: Game %s start!!!\n"%(data['tableNumber'],TABLE_STATE['game_id']))
        elif event_name == '__new_round':
            TABLE_STATE['round_id'] += 1
            init_game_state(data['players'],data['table'])
            if data['table']['smallBlind']['playerName']==TABLE_STATE['name_md5']:
                TABLE_STATE['forced_bet']  = data['table']['smallBlind']['amount']
            elif data['table']['bigBlind']['playerName']==TABLE_STATE['name_md5']:
                TABLE_STATE['forced_bet']  = data['table']['bigBlind']['amount']
            else:
                TABLE_STATE['forced_bet']  = 0
            PREV_AGENT_STATE  = None
            #
            if player_stats is None:
                player_stats  = pd.DataFrame(0,columns=pd.MultiIndex.from_tuples([(x,y) for x in ('deal','flop','turn','river') for y in ('rounds','prBet','amtBet','prCall','amtCall','oddsCall','prFold','amtFold','lossFold')]),index=GAME_STATE.index)
                player_stats[('deal','prBet')]  = 0.33
                player_stats[('deal','amtBet')] = data['table']['bigBlind']['amount']
                player_stats[('deal','prCall')] = 0.33
                player_stats[('deal','amtCall')] = data['table']['smallBlind']['amount']
                player_stats[('deal','oddsCall')] = 1/(2*GAME_STATE.isSurvive.sum()+1)
                player_stats[('deal','prFold')] = 0.33
                player_stats[('deal','amtFold')] = data['table']['smallBlind']['amount']
                player_stats[('deal','lossFold')] = data['table']['smallBlind']['amount']
            else:
                for playerName in GAME_STATE.index:
                    if playerName not in player_stats.index:
                        player_stats.loc[playerName] = 0
                        player_stats.loc[playerName,('deal','prBet')]  = 0.33
                        player_stats.loc[playerName,('deal','amtBet')] = data['table']['bigBlind']['amount']
                        player_stats.loc[playerName,('deal','prCall')] = 0.33
                        player_stats.loc[playerName,('deal','amtCall')] = data['table']['smallBlind']['amount']
                        player_stats.loc[playerName,('deal','oddsCall')] = 1/(2*GAME_STATE.isSurvive.sum()+1)
                        player_stats.loc[playerName,('deal','prFold')] = 0.33
                        player_stats.loc[playerName,('deal','amtFold')] = data['table']['smallBlind']['amount']
                        player_stats.loc[playerName,('deal','lossFold')] = data['table']['smallBlind']['amount']
            #
            player_stats[('deal','rounds')] += 1
            #
            agent(event_name,data)
        elif event_name == '__deal':
            # New card is dealt, i.e. a new betting round
            if GAME_STATE is None:
                init_game_state(data['players'],data['table'])
            update_game_state(data['players'],data['table'],'reset_action_count')
            #
            rnd  = data['table']['roundName']
            if GAME_STATE is not None and player_stats is not None:
                player_stats[(rnd,'rounds')] += (~GAME_STATE.folded & GAME_STATE.isSurvive).astype(int)
            #
            agent(event_name,data)
        elif event_name == '__show_action':
            # Player action
            if GAME_STATE is None:
                init_game_state(data['players'],data['table'])
            update_game_state(data['players'],data['table'],data['action'])
            #
            record_action(data['action'])
            #
            act  = pd.Series(data['action'])
            if 'amount' not in act: act['amount'] = 0
            act['bet']  = GAME_STATE.loc[act.playerName,'bet']
            act['pot']  = GAME_STATE.loc[act.playerName,'roundBet'] + act.bet
            if player_stats is not None:
                roundName   = data['table']['roundName']
                playerName  = act.playerName
                if playerName not in player_stats.index:
                    player_stats.loc[playerName] = 0
                    player_stats.loc[playerName,('deal','prBet')]  = 0.33
                    player_stats.loc[playerName,('deal','amtBet')] = data['table']['bigBlind']['amount']
                    player_stats.loc[playerName,('deal','prCall')] = 0.33
                    player_stats.loc[playerName,('deal','amtCall')] = data['table']['smallBlind']['amount']
                    player_stats.loc[playerName,('deal','oddsCall')] = 1/(2*GAME_STATE.isSurvive.sum()+1)
                    player_stats.loc[playerName,('deal','prFold')] = 0.33
                    player_stats.loc[playerName,('deal','amtFold')] = data['table']['smallBlind']['amount']
                    player_stats.loc[playerName,('deal','lossFold')] = data['table']['smallBlind']['amount']
                #
                player_stats.loc[playerName,(roundName,'prBet')]  *= 1 - expsmo_alpha
                player_stats.loc[playerName,(roundName,'prCall')] *= 1 - expsmo_alpha
                player_stats.loc[playerName,(roundName,'prFold')] *= 1 - expsmo_alpha
                #
                if act.action == 'allin':
                    if act.amount > GAME_STATE.bet.max():
                        act['action'] = 'bet'
                    else:
                        act['action'] = 'call'
                if act.action in ('bet','raise'):
                    player_stats.loc[playerName,(roundName,'prBet')] += expsmo_alpha
                    player_stats.loc[playerName,(roundName,'amtBet')] = expsmo_alpha*act.amount + (1-expsmo_alpha)*player_stats.loc[playerName,(roundName,'amtBet')]
                elif act.action in ('check','call'):
                    player_stats.loc[playerName,(roundName,'prCall')] += expsmo_alpha
                    player_stats.loc[playerName,(roundName,'amtCall')] = expsmo_alpha*act.amount + (1-expsmo_alpha)*player_stats.loc[playerName,(roundName,'amtCall')]
                    player_stats.loc[playerName,(roundName,'oddsCall')] = expsmo_alpha*act.amount/(act.pot+act.amount) + (1-expsmo_alpha)*player_stats.loc[playerName,(roundName,'oddsCall')]
                elif act.action == 'fold':
                    player_stats.loc[playerName,(roundName,'prFold')] += expsmo_alpha
                    player_stats.loc[playerName,(roundName,'amtFold')] = expsmo_alpha*(GAME_STATE.bet.max() - act.bet) + (1-expsmo_alpha)*player_stats.loc[playerName,(roundName,'amtFold')]
                    player_stats.loc[playerName,(roundName,'lossFold')] = expsmo_alpha*(GAME_STATE.loc[playerName,'roundBet'] + act.bet) + (1-expsmo_alpha)*player_stats.loc[playerName,(roundName,'lossFold')]
            #
            agent(event_name,data)
        elif event_name == '__round_end':
            if GAME_STATE is None:
                init_game_state(data['players'],data['table'])
            else:
                update_game_state(data['players'],data['table'])
            #
            record_round(data['players'])
            #
            agent(event_name,data)
        elif event_name in ('__game_over','__game_stop'):
            try:
                if GAME_STATE is None:
                    init_game_state(data['players'],data['table'])
                else:
                    update_game_state(data['players'],data['table'])
            except:
                pass
            if GAME_STATE is not None and 'winners' in data:
                record_game(data['winners'])
            #
            if len(GAME_LOG) > 0:
                pd.concat(GAME_LOG,0).to_csv("game_log_%s.csv"%TABLE_STATE['game_id'],index=False,encoding='utf-8-sig')
            if len(ROUND_LOG) > 0:
                pd.concat(ROUND_LOG,0).to_csv("round_log_%s.csv"%TABLE_STATE['game_id'],index=False,encoding='utf-8-sig')
                ROUND_LOG  = []
            #
            if len(AGENT_LOG) > 0:
                pd.concat(AGENT_LOG,1,ignore_index=True).T.to_csv("agent_%s.csv"%TABLE_STATE['game_id'],index=False,encoding='utf-8-sig')
                AGENT_LOG  = []
            #
            if len(ACTION_LOG) > 0:
                pd.concat(ACTION_LOG,1,ignore_index=True).T.to_csv("action_log_%s.csv"%TABLE_STATE['game_id'],index=False,encoding='utf-8-sig')
                ACTION_LOG  = []
            #
            agent(event_name,data)
            ws  = None # Force re-join game
        elif event_name not in ('__left','__new_peer'):
            print("event received: %s\n" % event_name)
        #
        #-- Console Output --#
        if event_name in ('__new_round','__deal','__show_action','__round_end','__game_over'):
            try:
                if player_stats is not None:
                    output  = player_stats.copy()
                    output.index   = ['--> Me <--' if TABLE_STATE['name_md5']==x else x for x in output.index]
                    output.loc['Table Median'] = output.median(0)
                    print(pd.concat([output.loc[['Me','Table Median'],'deal'],output.loc[['Me','Table Median'],'flop'],output.loc[['Me','Table Median'],'turn'],output.loc[['Me','Table Median'],'river']],0,keys=('deal','flop','turn','river')))
                    print()
                #-- Output Game State --#
                print("Table %(tableNumber)s: Game %(game_id)s:\nRound %(round_id)2d-%(roundName)5s: Board [%(board)s]" % TABLE_STATE)
                output  = GAME_STATE.copy()
                output.index = ['--> Me <--' if TABLE_STATE['name_md5']==x else x for x in output.index]
                output.loc[output.allIn,'action'] = 'allin'
                output.loc[output.folded,['action','cards']] = 'fold'
                print(output[['chips','position','roundBet','bet','cards','action','amount',]].rename(columns={'roundBet':'pot','position':'pos','action':'act','amount':'amt'}).fillna(''))
                print()
            except Exception as e:
                print(e)

if __name__ == '__main__':
    #-- Command Line Arguments --#
    server = sys.argv[1] if len(sys.argv)>1 else 'battle'
    if server == 'battle':
        url  = 'ws://poker-battle.vtr.trendnet.org:3001'
    elif server == 'training':
        url  = 'ws://poker-training.vtr.trendnet.org:3001'
    #
    TABLE_STATE['name']  = sys.argv[2] if len(sys.argv)>2 else '790595a15ed748cc83de763fe4cbfeee' #'22d2bbdd47f74f458e5b8ae603d3a093'
    m   = hashlib.md5()
    m.update(TABLE_STATE['name'].encode('utf8'))
    TABLE_STATE['name_md5']  = m.hexdigest()
    #
    doListen(url)
