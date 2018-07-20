#! /usr/bin/env python

from agent_common import *
from decision_logic import *

server = sys.argv[1] if len(sys.argv)>1 else 'battle'
if server == 'battle':
    url  = 'ws://poker-battle.vtr.trendnet.org:3001' #'ws://allhands2018-battle.dev.spn.a1q7.net:3001'
elif server == 'training':
    url  = 'ws://poker-training.vtr.trendnet.org:3001' #'ws://allhands2018-training.dev.spn.a1q7.net:3001'
elif server == 'preliminary':
    url  = 'ws://allhands2018.dev.spn.a1q7.net:3001/'
else:
    url  = 'ws://allhands2018-beta.dev.spn.a1q7.net:3001'

deal_win_prob = pd.read_csv('deal_win_prob.csv',index_col='hole')

MP_JOBS  = 3

name = sys.argv[2] if len(sys.argv)>2 else '790595a15ed748cc83de763fe4cbfeee' #'22d2bbdd47f74f458e5b8ae603d3a093'
m    = hashlib.md5()
m.update(name.encode('utf8'))
name_md5 = m.hexdigest()

record  = bool(int(sys.argv[3])) if len(sys.argv)>3 else True

# Agent State
SMALL_BLIND  = 0
NUM_BLIND    = 0
FORCED_BET   = 0
ROUND_AGGRESSORS = [] # players who bet or raised during this round
LAST_BET_AMT = 0
PREV_STATE   = None # Info on previous decision point and resulting actions
PREV_WIN     = None # 

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
    global SMALL_BLIND
    global NUM_BLIND
    global FORCED_BET
    global ROUND_AGGRESSORS
    global LAST_BET_AMT
    #
    global PREV_STATE
    global PREV_WIN
    #
    global LOGIC_LIST
    global LOGIC
    global INIT_LOGIC_DECAY
    global LOGIC_DECAY
    #
    global TIGHTNESS
    global AGGRESIVENESS
    #
    print('Tightness:\n',TIGHTNESS,'\nAggressiveness: ',AGGRESIVENESS)
    if event in ('__action','__bet'):
        #
        #-- Calculate Basic Stats and Win Probability --#
        #
        #-- Game State Variables --#
        players  = pd.DataFrame(data['game']['players']).set_index('playerName')
        players  = players[players.isSurvive]
        state    = pd.Series()
        state['smallBlind'] = SMALL_BLIND
        state['forced_bet'] = FORCED_BET
        if FORCED_BET > 0: FORCED_BET = 0
        state['roundName']  = data['game']['roundName'].lower()
        state['N']     = len(players)
        state['Nnf']   = state.N - players.folded.sum()
        state['Nallin'] = players.allIn.sum()
        state['first'] = event == '__bet'
        hole   = pkr_to_cards(data['self']['cards'])
        board  = pkr_to_cards(data['game']['board'])
        state['hole']  = data['self']['cards'] #cards_to_str(hole)
        state['board'] = data['game']['board'] #cards_to_str(board)
        #
        #-- Calculate Win Probability --#
        if state.roundName == 'deal':
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
        state['tightness']  = TIGHTNESS[state.roundName] - (state.N - 3)*0.11 if state.roundName=='deal' else TIGHTNESS[state.roundName]
        state['aggresiveness'] = AGGRESIVENESS
        state['prWin_adj']  = np.maximum(state.prWin - state.tightness*np.sqrt(state.prWin*(1-state.prWin)),0)
        #
        #-- Betting Variables --#
        state['chips']  = data['self']['chips']
        state['reloadCount']   = data['self']['reloadCount']
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
        if state.Nallin>0 or len(ROUND_AGGRESSORS)>0 or players[players.index!=name_md5].chips.max() > 0.8*state.chips:
            bluff_freq   = 0
        elif player_stats is not None:
            player_stats = player_stats.loc[players[players.isSurvive & ~players.folded & (players.index!=name_md5)].index]
            bluff_freq   = player_stats[(state.roundName,'prFold')].prod()
        else:
            if state.roundName == 'deal':    prFold0 = 0.345100375
            elif state.roundName == 'flop':  prFold0 = 0.324900751
            elif state.roundName == 'turn':  prFold0 = 0.208825543
            elif state.roundName == 'river': prFold0 = 0.161097504
            bluff_freq   = prFold0**(players.isSurvive & ~players.folded & (players.index!=name_md5)).sum()
        state['bluff_freq']  = bluff_freq
        print("bluff_freq: %.3f%%"%(100*bluff_freq))
        print("round_aggressors: %s"%[playerMD5[x] if x in playerMD5 else x for x in ROUND_AGGRESSORS])
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
        PREV_STATE  = state
        #
        return resp,state
    #
    elif event in ('__new_round','__deal'):
        if event == '__new_round':
            PREV_STATE = None
            #
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
            ROUND_AGGRESSORS = []
        elif event == '__new_round':
            N_EFFECTIVE  = players.isSurvive.sum()
            ROUND_AGGRESSORS = []
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
    elif event in ('__round_end',):
        pass
        # players = pd.DataFrame(data['players']).set_index('playerName')
        # N_EFFECTIVE = players.isSurvive.sum()
        # hole   = pkr_to_cards(players.loc[name_md5,'cards'])
        # board  = pkr_to_cards(data['table']['board'])
        # calculate_win_prob_mp_stop(N_EFFECTIVE,hole,board)
    elif event in ('__game_over','__game_stop'):
        pass
        # players  = pd.DataFrame(data['players']).set_index('playerName')
        # N_EFFECTIVE = players.isSurvive.sum()
        # hole   = pkr_to_cards(players.loc[name_md5,'cards'])
        # board  = pkr_to_cards(data['table']['board'])
        # calculate_win_prob_mp_stop(N_EFFECTIVE,hole,board)

if __name__ == '__main__':
    doListen(url,name,agent_jyp,record)
