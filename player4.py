#! /usr/bin/env python

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

deal_win_prob = pd.read_csv('deal_win_prob.csv',index_col='hole')

MP_JOBS  = 1

name = sys.argv[2] if len(sys.argv)>2 else 'jyp'
m    = hashlib.md5()
m.update(name.encode('utf8'))
name_md5 = m.hexdigest()

record  = bool(int(sys.argv[3])) if len(sys.argv)>3 else False

# Agent State
SMALL_BLIND  = 0
NUM_BLIND    = 0
TIGHTNESS    = {    # Reference tightness for N=3
    'deal':  -0.37, # tightness + (N - 3)*0.11 for N players
    'flop':  0,
    'turn':  0,
    'river': 0,
    }
AGGRESIVENESS = 0.8
FORCED_BET    = 0
N_EFFECTIVE   = None # Effective number of players
DETERMINISM   = 0.9
ROUND_AGGRESSORS = [] # players who bet or raised during this round

def agent_jyp(event,data):
    global SMALL_BLIND
    global NUM_BLIND
    global TIGHTNESS
    global AGGRESIVENESS
    global FORCED_BET
    global N_EFFECTIVE
    global DETERMINISM
    global ROUND_AGGRESSORS
    #
    print('Tightness:\n',TIGHTNESS,'\nAggressiveness: ',AGGRESIVENESS)
    if event in ('__action','__bet'):
        #
        #-- Calculate Basic Stats and Win Probability --#
        #
        #-- Game State Variables --#
        players  = pd.DataFrame(data['game']['players']).set_index('playerName')
        players  = players[players.isSurvive]
        if N_EFFECTIVE is None: N_EFFECTIVE = len(players)
        state    = pd.Series()
        state['N']     = len(players)
        state['Nnf']   = state.N - players.folded.sum()
        state['roundName'] = data['game']['roundName'].lower()
        state['first'] = event == '__bet'
        hole   = pkr_to_cards(data['self']['cards'])
        board  = pkr_to_cards(data['game']['board'])
        state['hole']  = cards_to_str(hole)
        state['board'] = cards_to_str(board)
        #
        #-- Win Probability --#
        prWin_OK  = False
        if state.roundName == 'deal' and N_EFFECTIVE <= 10:
            state['Nsim'],state['prWin'],state['prWinStd'] = read_win_prob(N_EFFECTIVE,hole)
            prWin_OK  = True
        if not prWin_OK:
            try:
                prWin_samples = calculate_win_prob_mp_get()
                time.sleep(1.1)
                prWin_samples = calculate_win_prob_mp_get()
                prWin_samples = [x['prWin'] for x in prWin_samples]
                state['Nsim']     = len(prWin_samples)
                state['prWin']    = np.mean(prWin_samples)
                state['prWinStd'] = np.std(prWin_samples)
            except Exception as e:
                print(e)
                state['Nsim'] = 160
                state['prWin'],state['prWinStd'] = calculate_win_prob(N_EFFECTIVE,hole,board,Nsamp=state['Nsim'])
            prWin_OK  = True
        tightness  = TIGHTNESS[state.roundName] - (N_EFFECTIVE - 3)*0.11 if state.roundName=='deal' else TIGHTNESS[state.roundName]
        state['prWin_adj']  = np.maximum(state.prWin - tightness*np.sqrt(state.prWin*(1-state.prWin)),0)
        #
        #-- Betting Variables --#
        state['chips']  = data['self']['chips']
        state['reld']   = data['self']['reloadCount']
        state['stack']  = state.chips + (2 - state.reld)*1000
        state['pot']    = data['self']['roundBet'] # self accumulated contribution to pot
        state['bet']    = data['self']['bet'] # self bet on this round
        state['cost_to_call']  = min(data['self']['minBet'],state.chips) # minimum bet to stay in game
        state['pot_sum'] = players.roundBet.sum()
        state['bet_sum'] = np.minimum(players.bet,state.bet + state.cost_to_call).sum()
        state['maxBet_all']    = players.bet.max()
        state['N_maxBet_all']  = (players.bet==state.maxBet_all).sum()
        players['cost_to_call'] = np.minimum(state.maxBet_all - players.bet,players.chips)
        #
        # Decide bluffing prob.
        player_stats  = get_player_stats()
        if players.allIn.any() or len(ROUND_AGGRESSORS)>0 or players[players.index!=name_md5].chips.max() > 0.8*state.chips:
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
        P  = state.pot_sum + state.bet_sum
        B0 = state.cost_to_call
        state['thd_call']  = (B0 - FORCED_BET)/(P + B0)
        if FORCED_BET > 0: FORCED_BET = 0
        #
        op_wthd = 0.45 # opponent win prob. thd.
        bet     = int(op_wthd*P/(1-2*op_wthd))
        if B0 > 0:
            # Need to pay "cost_to_call" to stay in game
            if state.prWin_adj < state.thd_call:
                state['bet_limit']  = min((state.prWin_adj*P + FORCED_BET)/(1 - 2*state.prWin_adj),P,state.chips/2)
                resp  = takeAction([1-bluff_freq,0,bluff_freq,int(state.bet_limit)])
            elif state.prWin_adj >= 0.5:
                if state.prWin_adj >= 0.9:
                    state['bet_limit']  = np.inf
                    resp  = takeAction([0,1-DETERMINISM,0,0])
                else:
                    if state.prWin_adj >= 0.8:
                        state['bet_limit']  = min(2*P,3*state.chips/4)
                    elif state.prWin_adj >= 0.7:
                        state['bet_limit']  = min(P,state.chips/2)
                    elif state.prWin_adj >= 0.6:
                        state['bet_limit']  = min(P/2,state.chips/4)
                    else:
                        state['bet_limit']  = min(P/4,state.chips/8) if np.random.random()>0.5 else 0
                    #
                    resp  = takeAction([0,1-DETERMINISM,DETERMINISM,int(state.bet_limit)])
            else: # state.prWin_adj < 0.5
                state['bet_limit']  = min((1 - state.prWin_adj)*B0/(1 - 2*state.prWin_adj),2*P,3*state.chips/4)
                resp  = takeAction([0,1-bluff_freq,bluff_freq,int(state.bet_limit)])
        else: # B0 == 0, i.e. can stay in the game for free
            if state.prWin_adj >= 0.5:
                if state.prWin_adj >= 0.9:
                    state['bet_limit']  = np.inf
                    resp  = takeAction([0,1-DETERMINISM,0,0])
                else:
                    if state.prWin_adj >= 0.8:
                        state['bet_limit']  = min(2*P,3*state.chips/4)
                    elif state.prWin_adj >= 0.7:
                        state['bet_limit']  = min(P,state.chips/2)
                    elif state.prWin_adj >= 0.6:
                        state['bet_limit']  = min(P/2,state.chips/4)
                    else:
                        state['bet_limit']  = min(P/4,state.chips/8) if np.random.random()>0.5 else 0
                    #
                    resp  = takeAction([0,1-DETERMINISM,DETERMINISM,int(state.bet_limit)])
            else: # state.prWin_adj < 0.5
                state['bet_limit']  = min(P/2,state.chips/4)
                resp  = takeAction([0,1-bluff_freq,bluff_freq,int(state.bet_limit)])
        #
        state['action'] = resp[0]
        state['amount'] = resp[1]
        return resp,state
    #
    elif event in ('__new_round','__deal'):
        #
        #-- Determine Effective Number of Players --#
        players  = pd.DataFrame(data['players']).set_index('playerName')
        if event == '__deal' and data['table']['roundName'] == 'Flop':
            # Flop cards are just dealt, anyone that folded preflop can be considered out of the game and not figured in win probability calculation
            N_EFFECTIVE  = (players.isSurvive & ~players.folded).sum()
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
                            TIGHTNESS[rnd] -= 0.01
                        else:
                            TIGHTNESS[rnd] += 0.01
                    if rnd == 'turn':
                        TIGHTNESS[rnd]  = 0.95*TIGHTNESS[rnd] + 0.05*TIGHTNESS['flop']
                    elif rnd == 'river':
                        TIGHTNESS[rnd]  = 0.95*TIGHTNESS[rnd] + 0.05*TIGHTNESS['turn']
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
    elif event in ('__round_end','__game_over','__game_stop'):
        calculate_win_prob_mp_stop()

if __name__ == '__main__':
    doListen(url,name,agent_jyp,record)
