#! /usr/bin/env python

from agent_common import *

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

name = sys.argv[2] if len(sys.argv)>2 else '22d2bbdd47f74f458e5b8ae603d3a093'
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

TIGHTNESS    = {   # Reference tightness for N=3
    'deal':  -0.2, # tightness + (N - 3)*0.11 for N players
    'flop':  -0.1,
    'turn':  0,
    'river': 0,
    }
AGGRESIVENESS = 0.5

LOGIC_LIST   = ('basic','player4')
LOGIC        = 0 #'basic' #'player4'
INIT_LOGIC_DECAY = 0.95
LOGIC_DECAY  = INIT_LOGIC_DECAY

def basic_logic(state):
    DETERMINISM  = 0.9
    #
    state['maxBet']     = state.bet + state.minBet
    state['util_fold']  = -state.pot - state.bet
    state['util_call']  = state.prWin*state.pot_sum + state.prWin*state.maxBet*state.Nnf - state.pot - state.maxBet
    state['util_raise_coeff']  = state.prWin*state.Nnf - 1
    #
    # Worst case scenario utility (everyone but one folds, i.e. a dual)
    state['util_call2'] = state.prWin*state.pot_sum + state.prWin*state.maxBet*2 - state.pot - state.maxBet
    state['util_raise_coeff2'] = state.prWin*2 - 1
    #
    if state.cost_to_call > 0:
        # Need to "pay" to stay in game
        if state.util_fold > state.util_call:
            return [DETERMINISM,1-DETERMINISM,0,0]
        elif state.util_raise_coeff > 0:
            if state.prWin > 0.9:
                return [0,1-DETERMINISM,0,0]
            elif state.prWin > 0.75:
                return [0,1-DETERMINISM,DETERMINISM,state.pot_sum + state.bet_sum]
            else:
                return [0,1-DETERMINISM,DETERMINISM,'raise']
        else:
            return [0,DETERMINISM,1-DETERMINISM,0]
    else:
        # Can stay in the game for free
        if state.util_raise_coeff > 0:
            if state.prWin > 0.9:
                return [0,1-DETERMINISM,0,0]
            elif state.prWin > 0.75:
                return [0,1-DETERMINISM,DETERMINISM,state.pot_sum + state.bet_sum]
            else:
                if np.random.random() < 0.5:
                    return [0,1-DETERMINISM,DETERMINISM,'raise']
                else:
                    return [0,1-DETERMINISM,DETERMINISM,0]
        else:
            return [0,DETERMINISM,1-DETERMINISM,0]

def player4_logic(state):
    DETERMINISM  = 0.9
    P  = state.pot_sum + state.bet_sum
    B0 = state.cost_to_call
    state['thd_call']  = (B0 - state.forced_bet)/(P + B0)
    #
    # op_wthd = 0.45 # opponent win prob. thd.
    # bet     = int(op_wthd*P/(1-2*op_wthd))
    if B0 > 0:
        # Need to pay "cost_to_call" to stay in game
        if state.prWin_adj < state.thd_call:
            state['bet_limit']  = min((state.prWin_adj*P + state.forced_bet)/(1 - 2*state.prWin_adj),P,state.chips*state.aggresiveness/2)
            return [1-state.bluff_freq,0,state.bluff_freq,int(state.bet_limit)]
        elif state.prWin_adj >= 0.5:
            if state.prWin_adj >= 0.9:
                state['bet_limit']  = np.inf
                return [0,1-DETERMINISM,0,0]
            else:
                if state.prWin_adj >= 0.8:
                    state['bet_limit']  = min(2*P,3*state.chips*state.aggresiveness/4)
                elif state.prWin_adj >= 0.7:
                    state['bet_limit']  = min(P,state.chips*state.aggresiveness/2)
                elif state.prWin_adj >= 0.6:
                    state['bet_limit']  = min(P/2,state.chips*state.aggresiveness/4)
                else:
                    state['bet_limit']  = min(P/4,state.chips*state.aggresiveness/8) if np.random.random()>0.5 else 0
                #
                return [0,1-DETERMINISM,DETERMINISM,int(state.bet_limit)]
        else: # state.prWin_adj < 0.5
            if state.prWin_adj >= 0.4:
                state['bet_limit']  = min((1 - state.prWin_adj)*B0/(1 - 2*state.prWin_adj),P,state.chips*state.aggresiveness/2)
            elif state.prWin_adj >= 0.3:
                state['bet_limit']  = min((1 - state.prWin_adj)*B0/(1 - 2*state.prWin_adj),P/2,state.chips*state.aggresiveness/4)
            elif state.prWin_adj >= 0.2:
                state['bet_limit']  = min((1 - state.prWin_adj)*B0/(1 - 2*state.prWin_adj),P/4,state.chips*state.aggresiveness/8)
            else:
                state['bet_limit']  = state.smallBlind
            #
            if state.cost_to_call < state.bet_limit:
                return [0,1-state.bluff_freq,state.bluff_freq,int(state.bet_limit)]
            else:
                return [1-state.bluff_freq,0,state.bluff_freq,int(state.bet_limit)]
        #
    else: # B0 == 0, i.e. can stay in the game for free
        if state.prWin_adj >= 0.5:
            if state.prWin_adj >= 0.9:
                state['bet_limit']  = np.inf
                return [0,1-DETERMINISM,0,0]
            else:
                if state.prWin_adj >= 0.8:
                    state['bet_limit']  = min(2*P,3*state.chips*state.aggresiveness/4)
                elif state.prWin_adj >= 0.7:
                    state['bet_limit']  = min(P,state.chips*state.aggresiveness/2)
                elif state.prWin_adj >= 0.6:
                    state['bet_limit']  = min(P/2,state.chips*state.aggresiveness/4)
                else:
                    state['bet_limit']  = min(P/4,state.chips*state.aggresiveness/8) if np.random.random()>0.5 else 0
                #
                return [0,1-DETERMINISM,DETERMINISM,int(state.bet_limit)]
        else: # state.prWin_adj < 0.5
            if state.prWin_adj >= 0.4:
                state['bet_limit']  = min((1 - state.prWin_adj)*B0/(1 - 2*state.prWin_adj),P,state.chips*state.aggresiveness/2)
            elif state.prWin_adj >= 0.3:
                state['bet_limit']  = min((1 - state.prWin_adj)*B0/(1 - 2*state.prWin_adj),P/2,state.chips*state.aggresiveness/4)
            elif state.prWin_adj >= 0.2:
                state['bet_limit']  = min((1 - state.prWin_adj)*B0/(1 - 2*state.prWin_adj),P/4,state.chips*state.aggresiveness/8)
            else:
                state['bet_limit']  = state.smallBlind
            #
            state['bet_limit']  = min(P/2,state.chips*state.aggresiveness/4)
            return [0,1-state.bluff_freq,state.bluff_freq,int(state.bet_limit)]

def michael_logic(state):
    """
    state.smallBlind
    state.forced_bet
    state.roundName - data['game']['roundName'].lower()
    state.N      - number of surviving players
    state.Nnf    - number of non-folded players
    state.Nallin - number of all in players
    state.first  - event == '__bet'
    state.hole   - data['self']['cards']
    state.board  - data['game']['board']
    state.Nsim   - number of Monte Carlo samples
    state.prWin  - hand win probability
    state.prWinStd - hand win probability St.D.
    state.chips  - data['self']['chips']
    state.reloadCount - data['self']['reloadCount']
    state.pot    - self.roundBet
    state.bet    - self.bet
    state.minBet
    state.cost_to_call - min(state.minBet,state.chips)
    state.pot_sum - players.roundBet.sum()
    state.bet_sum - np.minimum(players.bet,state.bet + state.cost_to_call).sum()
    state.NMaxBet - number of players that matched largest bet
    """
    C   = state.pot + state.bet
    P   = state.pot_sum + state.bet_sum
    B0  = state.cost_to_call
    state['thd_call']  = (B0 - state.forced_bet)/(P + B0)
    bet_amt_range      = np.array([0.25*P,0.5*P,0.75*P,P,1.25*P,1.5*B0,2*B0,2.5*B0]).round().astype(int)
    #
    # op_wthd = 0.45 # opponent win prob. thd.
    # bet     = int(op_wthd*P/(1-2*op_wthd))
    if B0 > 0:
        # Need to pay "cost_to_call" to stay in game
        if state.prWin_adj < state.thd_call:
            state['bet_limit']  = min((state.prWin_adj*P + state.forced_bet)/(1 - 2*state.prWin_adj),P,state.chips*state.aggresiveness/2)
            return [1-state.bluff_freq,0,state.bluff_freq,int(state.bet_limit)]
        elif state.prWin_adj >= 0.5:
            if state.prWin_adj >= 0.9:
                state['bet_limit']  = np.inf
                return [0,1-DETERMINISM,0,0]
            else:
                if state.prWin_adj >= 0.8:
                    state['bet_limit']  = min(2*P,3*state.chips*state.aggresiveness/4)
                elif state.prWin_adj >= 0.7:
                    state['bet_limit']  = min(P,state.chips*state.aggresiveness/2)
                elif state.prWin_adj >= 0.6:
                    state['bet_limit']  = min(P/2,state.chips*state.aggresiveness/4)
                else:
                    state['bet_limit']  = min(P/4,state.chips*state.aggresiveness/8) if np.random.random()>0.5 else 0
                #
                return [0,1-DETERMINISM,DETERMINISM,int(state.bet_limit)]
        else: # state.prWin_adj < 0.5
            if state.prWin_adj >= 0.4:
                state['bet_limit']  = min((1 - state.prWin_adj)*B0/(1 - 2*state.prWin_adj),P,state.chips*state.aggresiveness/2)
            elif state.prWin_adj >= 0.3:
                state['bet_limit']  = min((1 - state.prWin_adj)*B0/(1 - 2*state.prWin_adj),P/2,state.chips*state.aggresiveness/4)
            elif state.prWin_adj >= 0.2:
                state['bet_limit']  = min((1 - state.prWin_adj)*B0/(1 - 2*state.prWin_adj),P/4,state.chips*state.aggresiveness/8)
            else:
                state['bet_limit']  = state.smallBlind
            #
            if state.cost_to_call < state.bet_limit:
                return [0,1-bluff_freq,bluff_freq,int(state.bet_limit)]
            else:
                return [1-bluff_freq,0,bluff_freq,int(state.bet_limit)]
        #
    else: # B0 == 0, i.e. can stay in the game for free
        if state.prWin_adj >= 0.5:
            if state.prWin_adj >= 0.9:
                state['bet_limit']  = np.inf
                return [0,1-DETERMINISM,0,0]
            else:
                if state.prWin_adj >= 0.8:
                    state['bet_limit']  = min(2*P,3*state.chips*state.aggresiveness/4)
                elif state.prWin_adj >= 0.7:
                    state['bet_limit']  = min(P,state.chips*state.aggresiveness/2)
                elif state.prWin_adj >= 0.6:
                    state['bet_limit']  = min(P/2,state.chips*state.aggresiveness/4)
                else:
                    state['bet_limit']  = min(P/4,state.chips*state.aggresiveness/8) if np.random.random()>0.5 else 0
                #
                return [0,1-DETERMINISM,DETERMINISM,int(state.bet_limit)]
        else: # state.prWin_adj < 0.5
            if state.prWin_adj >= 0.4:
                state['bet_limit']  = min((1 - state.prWin_adj)*B0/(1 - 2*state.prWin_adj),P,state.chips*state.aggresiveness/2)
            elif state.prWin_adj >= 0.3:
                state['bet_limit']  = min((1 - state.prWin_adj)*B0/(1 - 2*state.prWin_adj),P/2,state.chips*state.aggresiveness/4)
            elif state.prWin_adj >= 0.2:
                state['bet_limit']  = min((1 - state.prWin_adj)*B0/(1 - 2*state.prWin_adj),P/4,state.chips*state.aggresiveness/8)
            else:
                state['bet_limit']  = state.smallBlind
            #
            state['bet_limit']  = min(P/2,state.chips*state.aggresiveness/4)
            return [0,1-bluff_freq,bluff_freq,int(state.bet_limit)]

def agent_jyp(event,data):
    global SMALL_BLIND
    global NUM_BLIND
    global FORCED_BET
    global ROUND_AGGRESSORS
    global LAST_BET_AMT
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
        state['NMaxBet'] = ((players.bet>0) & (players.bet==players.bet.max())).sum()
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
        state['logic'] = LOGIC_LIST[LOGIC]
        if state.logic == 'player4':
            resp  = player4_logic(state)
        elif state.logic == 'basic':
            resp  = basic_logic(state)
        #
        state['resp']  = resp
        resp  = takeAction(resp)
        state['action'] = resp[0]
        state['amount'] = resp[1]
        return resp,state
    #
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
