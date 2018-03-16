#! /usr/bin/env python
# -*- coding:utf-8 -*-

from agent_common import *
import random

server = sys.argv[1] if len(sys.argv)>1 else 'beta'
if server == 'battle':
    url  = 'ws://allhands2018-battle.dev.spn.a1q7.net:3001'
elif server == 'training':
    url  = 'ws://allhands2018-training.dev.spn.a1q7.net:3001'
else:
    url  = 'ws://allhands2018-beta.dev.spn.a1q7.net:3001'

name = sys.argv[2] if len(sys.argv)>2 else 'jyp'
m    = hashlib.md5()
m.update(name.encode('utf8'))
name_md5 = m.hexdigest()

# Agent State
GAMBLE_STATE  = True
GAMBLE_STATE_TRANSITION = 0.15
BANKRUPT_TOL  = 0.1 # Taking the same prWin chance, make sure we only risk a limited amount each trial so there's (1-BANKRUPT_TOL)% chance we win one trial before bankruptcy
GAMBLE_THD    = 0.1 # How much percentage of total asset to gamble when the odds (utility to call/bet/raise) are against us

def agent_jyp(event,data):
    global GAMBLE_STATE
    global GAMBLE_STATE_TRANSITION
    global BANKRUPT_TOL
    global GAMBLE_THD
    #
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
            try:
                state['Nsim'] = 20000
                state['prWin'],state['prWinStd'] = read_win_prob(state.N,hole)
                prWin_OK  = True
            except:
                pass
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
        #-- Decision Support Variables --#
        players  = players[players.playerName!=name_md5] # Consider only other players
        players['cost_on_table'] = players.roundBet + players.bet
        players['cost_to_call']  = np.minimum(state.maxBet_all - players.bet,players.chips)
        #
        util_call_lose = -np.minimum(players.cost_on_table + players.cost_to_call,state.cost_total).mean()
        util_call_win  = state.cost_total + np.minimum(players.cost_on_table + players.cost_to_call,state.cost_total).sum()
        state['util_fold']   = -state.cost_on_table
        state['util_call']   = state.prWin*util_call_win + (1-state.prWin)*util_call_lose
        state['N_canraise']  = (~players.folded & (players.chips>players.cost_to_call)).sum()
        state['util_raise_coeff'] = state.prWin*(state.N_canraise + 1) - 1
        #
        #-- Worst case scenario utility (everyone afterwards folds, becomes a dual in the worst case) --#
        util_call_win2 = state.cost_total + np.minimum(players.cost_on_table,state.cost_total).sum()
        state['util_call2']        = state.prWin*util_call_win2 + (1-state.prWin)*util_call_lose
        state['util_raise_coeff2'] = state.prWin*2 - 1
        #
        #-- Betting limit heuristic --#
        num_round_to_bankrupt  = min(np.log(BANKRUPT_TOL)/np.log(1-state.prWin),state.chips_total*5/(data['game']['bigBlind']['amount']+data['game']['smallBlind']['amount']))
        state['limitRoundBet'] = state.chips_total/num_round_to_bankrupt if state.prWin<1 else state.chips_total
        state['limitBet']      = state.limitRoundBet - state.cost_on_table
        #
        #-- Decision Logic --#
        base_prWin  = 1/state.N
        allin_prWin = 0.95
        bet_mult    = 0.9 #0.1
        # if state.roundName == 'deal':
        #     bet_mult  = 0.2
        # elif state.roundName == 'flop':
        #     bet_mult  = 0.5
        # elif state.roundName == 'turn':
        #     bet_mult  = 0.618
        # elif state.roundName == 'river':
        #     bet_mult  = 0.8
        state['gamble']      = GAMBLE_STATE
        state['gamble_tol']  = max(int(state.chips_total*GAMBLE_THD),2*data['game']['bigBlind']['amount'])
        #
        if state.cost_to_call > 0:
            # Need to pay "cost_to_call" to stay in game
            if state.Nsim > 200 and state.prWin > allin_prWin:
                resp = ('allin',0)
            elif state.Nsim > 200 and state.prWin < base_prWin and state.roundName != 'deal':
                resp = ('fold',0)
            elif state.util_fold > state.util_call:
                if GAMBLE_STATE and state.util_fold < 0 and state.util_fold - state.util_call <= state.gamble_tol:
                    resp = ('call',0)
                else:
                    resp = ('fold',0)
            elif state.cost_to_call > state.limitBet:
                if GAMBLE_STATE and state.cost_to_call <= state.gamble_tol:
                    resp = ('call',0)
                else:
                    resp = ('fold',0)
            elif state.util_raise_coeff > 0:
                if GAMBLE_STATE or state.util_raise_coeff > 0.2:
                    resp = ('bet',int(state.limitBet*bet_mult))
                else:
                    resp = ('call',0)
            else:
                resp = ('call',0)
        else:
            # Can stay in the game for free
            if state.Nsim > 200 and state.prWin > allin_prWin:
                resp = ('allin',0)
            elif state.util_raise_coeff > 0:
                if GAMBLE_STATE or state.util_raise_coeff > 0.3:
                    resp = ('bet',int(state.limitBet*bet_mult))
                else:
                    resp = ('check',0)
            else:
                resp = ('check',0)
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
        if event in '__new_round' and random.random() < GAMBLE_STATE_TRANSITION:
            GAMBLE_STATE  = not GAMBLE_STATE
    elif event in ('__round_end','__game_over','__game_stop'):
        calculate_win_prob_mp_stop()
    elif event == '__start_reload':
        self_chips = 0
        avg_chips  = 0
        for x in data['players']:
            if x['playerName']==name_md5:
                self_chips  = x['chips']
            else:
                avg_chips  += x['chips']
        avg_chips  = avg_chips/(len(data['players'])-1)
        print("Reload?: %d" % (avg_chips - self_chips))
        return avg_chips - self_chips > 1000
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
