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

name = sys.argv[2] if len(sys.argv)>2 else 'jyp'
m    = hashlib.md5()
m.update(name.encode('utf8'))
name_md5 = m.hexdigest()

# Agent State
SMALL_BLIND  = 0
NUM_BLIND    = 0
BANKRUPT_TOL = {
    'deal': 0.0005,
    'flop': 0.01,
    'turn': 0.05,
    'river': 0.1,
    }
TIGHTNESS    = {
    'deal':  -0.17,
    'flop':  -0.17,
    'turn':  -0.22,
    'river': -0.26,
    }
AGGRESIVENESS = 0.8
FORCED_BET    = 0
TABLE_STATS   = None

# Winning config against ['隨便','( ´_ゝ`) Dracarys','ERS yo','Yeeeee','(=´ᴥ`)']
# Tightness:
#  {'deal': -0.225, 'flop': -0.51, 'turn': -0.465, 'river': -0.28}
# Aggressiveness:  0.8

def adj_win_prob_limit_bet(prWin,prWin_adj,bankrupt_tol):
    prWin    = np.maximum(prWin - prWin_adj*np.sqrt(prWin*(1-prWin)),0)
    limitBet = (np.log(1-prWin)/np.log(bankrupt_tol))
    return prWin,limitBet

def agent_jyp(event,data):
    global SMALL_BLIND
    global NUM_BLIND
    global BANKRUPT_TOL
    global TIGHTNESS
    global AGGRESIVENESS
    global FORCED_BET
    global TABLE_STATS
    #
    print('Tightness:\n',TIGHTNESS,'\nAggressiveness: ',AGGRESIVENESS)
    if event in ('__action','__bet'):
        #
        #-- Calculate Basic Stats and Win Probability --#
        #
        #-- Game State Variables --#
        players  = pd.DataFrame(data['game']['players'])
        players  = players[players.isSurvive]
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
        if state.roundName == 'deal' and state.N <= 10:
            state['Nsim'],state['prWin'],state['prWinStd'] = read_win_prob(state.N,hole)
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
            except:
                state['Nsim'] = 14
                state['prWin'],state['prWinStd'] = calculate_win_prob(state.N,hole,board,Nsamp=state['Nsim'])
            prWin_OK  = True
        #
        #-- Betting Variables --#
        state['chips']  = data['self']['chips']
        state['reld']   = data['self']['reloadCount']
        state['chips_total'] = state.chips + (2 - state.reld)*1000
        state['pot']    = data['self']['roundBet'] # self accumulated contribution to pot
        state['bet']    = data['self']['bet'] # self bet on this round
        state['cost_on_table'] = state.pot + state.bet
        state['cost_to_call']  = min(data['self']['minBet'],state.chips) # minimum bet to stay in game
        state['cost_total']    = state.cost_on_table + state.cost_to_call
        state['maxBet_all']    = players.bet.max()
        state['N_maxBet_all']  = (players.bet==state.maxBet_all).sum()
        #
        #-- Betting limit heuristic --#
        T_BB   = num_rounds_to_bankrupt(state.chips_total,state.N,SMALL_BLIND,NUM_BLIND)
        state['prWin_adj'],state['limitBet'] = adj_win_prob_limit_bet(state.prWin,TIGHTNESS[state.roundName],BANKRUPT_TOL[state.roundName])
        state['limitBet'] *= state.chips_total
        #
        #-- Decision Support Variables --#
        players  = players[players.playerName!=name_md5] # Consider only other players
        players['cost_on_table'] = players.roundBet + players.bet
        players['cost_to_call']  = np.minimum(state.maxBet_all - players.bet,players.chips)
        #
        state['util_fold']        = -FORCED_BET
        if FORCED_BET > 0: FORCED_BET = 0
        state['util_call']        = state.prWin_adj*(players.cost_on_table.sum() + state.cost_on_table + 2*state.cost_to_call) - state.cost_to_call
        state['util_raise_coeff'] = state.prWin_adj*2 - 1
        #
        #-- Decision Logic --#
        #
        game_state    = get_game_state()[-1]
        player_stats  = get_player_stats()
        if player_stats is not None:
            player_stats  = player_stats.loc[game_state[game_state.isSurvive & ~game_state.folded].index]
            bluff_freq    = player_stats[(state.roundName,'prFold')].prod()
        else:
            bluff_freq    = 0.33**(game_state.isSurvive & ~game_state.folded).sum()
        #
        pot     = int(players.cost_on_table.sum())
        op_wthd = 0.45 # opponent win prob. thd.
        bet     = int(op_wthd*pot/(1-2*op_wthd))
        if state.cost_to_call > 0:
            # Need to pay "cost_to_call" to stay in game
            if state.util_call < state.util_fold:
                # resp  = takeAction([1-bluff_freq,0,bluff_freq,0])
                resp  = takeAction([1,0,0,0])
            elif state.util_raise_coeff > 0:
                if state.prWin_adj > 0.8:
                    pr_allin = state.prWin_adj**4
                    resp  = takeAction([0,0.05,0.95-pr_allin,pot])
                elif state.prWin_adj > 0.65:
                    bet   = 'raise' if np.random.random()<0.5 else int(pot)
                    resp  = takeAction([0,0.1,0.9,bet])
                else:
                    bet   = 0 if np.random.random()<0.5 else int(pot/2)
                    resp  = takeAction([0,0.2,0.8,bet])
            else:
                resp  = takeAction([0,1-bluff_freq/2,bluff_freq/2,int(pot/4)])
        else:
            # Can stay in the game for free
            if state.util_raise_coeff > 0:
                if state.prWin_adj > 0.8:
                    pr_allin = state.prWin_adj**4
                    resp  = takeAction([0,0.05,0.95-pr_allin,pot])
                elif state.prWin_adj > 0.65:
                    bet   = 'raise' if np.random.random()<0.5 else int(pot)
                    resp  = takeAction([0,0.1,0.9,bet])
                else:
                    bet   = 0 if np.random.random()<0.5 else int(pot/2)
                    resp  = takeAction([0,0.2,0.8,bet])
            else:
                resp  = takeAction([0,1-bluff_freq,bluff_freq,int(pot/4)])
        #
        state['action'] = resp[0]
        state['amount'] = resp[1]
        return resp,state
    #
    elif event in ('__new_round','__deal'):
        board  = pkr_to_cards(data['table']['board'])
        N      = len(data['players'])
        for x in data['players']:
            if x['playerName']==name_md5:
                hole = pkr_to_cards(x['cards']) if 'cards' in x else pkr_to_cards([])
                break
        calculate_win_prob_mp_start(N,hole,board,n_jobs=3)
        #
        # if event == '__new_round' and random.random() < GAMBLE_STATE_TRANSITION:
        #     GAMBLE_STATE  = not GAMBLE_STATE
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
            #
            game_state    = get_game_state()[2]
            player_stats  = get_player_stats()
            print(game_state.loc[name_md5,'isSurvive'])
            if game_state is not None and player_stats is not None and game_state.loc[name_md5,'isSurvive'] and player_stats[('deal','rounds')].median() > 3:
                player_stats  = player_stats.loc[game_state[game_state.isSurvive].index]
                for rnd in ('deal','flop','turn','river'):
                    player_stats_rnd  = player_stats[rnd]
                    if player_stats_rnd.loc[name_md5,'rounds'] > 3:
                        if player_stats_rnd.loc[name_md5,'prFold'] >= player_stats_rnd.prFold.median():
                            TIGHTNESS[rnd] -= 0.005
                        else:
                            TIGHTNESS[rnd] += 0.005
                    if rnd == 'flop':
                        TIGHTNESS[rnd]  = 0.9*TIGHTNESS[rnd] + 0.1*TIGHTNESS['deal']
                    elif rnd == 'turn':
                        TIGHTNESS[rnd]  = 0.9*TIGHTNESS[rnd] + 0.1*TIGHTNESS['flop']
                    elif rnd == 'river':
                        TIGHTNESS[rnd]  = 0.9*TIGHTNESS[rnd] + 0.1*TIGHTNESS['turn']
    #
    elif event in ('__round_end','__game_over','__game_stop'):
        calculate_win_prob_mp_stop()
    # elif event == '__start_reload':
    #     self_chips = 0
    #     avg_chips  = 0
    #     for x in data['players']:
    #         if x['playerName']==name_md5:
    #             self_chips  = x['chips']
    #         else:
    #             avg_chips  += x['chips']
    #     avg_chips  = avg_chips/(len(data['players'])-1)
    #     print("Reload?: %d" % (avg_chips - self_chips))
    #     return avg_chips - self_chips > 1000
    # elif event == '__show_action':
    #     game_id,round_id,game_state = get_game_state()
    #     turn_id += 1
    #     action   = data['action']
    #     action['game_id']   = game_id
    #     action['round_id']  = round_id
    #     action['turn_id']   = turn_id
    #     action['roundName'] = data['table']['roundName']
    #     action['position']  = game_state.loc[action['playerName'],'position'] if game_state is not None else None
    #     player_actions.append(action)

if __name__ == '__main__':
    doListen(url,name,agent_jyp,True)
