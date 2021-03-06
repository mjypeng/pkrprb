#! /usr/bin/env python
# -*- coding:utf-8 -*-

from agent_common import *

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

def agent_jyp(event,data):
    if event in ('__action','__bet'):
        #
        #-- Calculate Basic Stats and Win Probability --#
        #
        #-- Game State Variables --#
        input_var  = pd.Series()
        input_var['N']   = len(data['game']['players'])
        input_var['Nnf'] = len([x for x in data['game']['players'] if not x['folded']])
        input_var['roundName'] = data['game']['roundName'].lower()
        input_var['first'] = int(event == '__bet')
        hole   = pkr_to_cards(data['self']['cards'])
        board  = pkr_to_cards(data['game']['board'])
        input_var['hole']  = cards_to_str(hole)
        input_var['board'] = cards_to_str(board)
        #
        #-- Win Probability --#
        prWin_OK  = False
        if input_var.roundName == 'deal' and input_var.N <= 10:
            try:
                input_var['Nsim'] = 20000
                input_var['prWin'],input_var['prWinStd'] = read_win_prob(input_var.N,hole)
                prWin_OK  = True
            except:
                pass
        if not prWin_OK:
            try:
                prWin_samples = calculate_win_prob_mp_get()
                time.sleep(1.2)
                prWin_samples = calculate_win_prob_mp_get()
                prWin_samples = [x['prWin'] for x in prWin_samples]
                input_var['Nsim']     = len(prWin_samples)
                input_var['prWin']    = np.mean(prWin_samples)
                input_var['prWinStd'] = np.std(prWin_samples)
            except:
                input_var['Nsim'] = 14
                input_var['prWin'],input_var['prWinStd'] = calculate_win_prob(input_var.N,hole,Nsamp=input_var['Nsim'])
            prWin_OK  = True
        #
        #-- Betting Variables --#
        input_var['chips']  = data['self']['chips']
        input_var['reld']   = data['self']['reloadCount']
        input_var['pot']    = data['self']['roundBet'] # self accumulated contribution to pot
        input_var['bet']    = data['self']['bet'] # self bet on this round
        input_var['minBet'] = data['self']['minBet'] # minimum additional bet
        input_var['maxBet'] = input_var.bet + input_var.chips # current maximum bet
        input_var['sumPot_all'] = sum([x['roundBet'] for x in data['game']['players']])
        input_var['sumBet_all'] = sum([min(x['bet'],input_var.maxBet) for x in data['game']['players']])
        input_var['maxBet_all'] = max([x['bet'] for x in data['game']['players']])
        input_var['N_maxBet_all'] = len([x for x in data['game']['players'] if x['bet']==input_var.maxBet_all])
        for x in data['game']['players']:
            x['minBet'] = input_var.maxBet_all - x['bet']
        input_var['N_canraise']    = len([x for x in data['game']['players'] if x['chips'] > x['minBet'] and not x['folded']])
        input_var['sumMinBet_all'] = sum([min(x['minBet'],x['chips'],input_var.chips) for x in data['game']['players'] if x['bet']<input_var.maxBet_all and not x['folded']])
        #
        #-- Decision Support Variables --#
        input_var['util_fold']  = -input_var.pot - input_var.bet
        input_var['util_call']  = input_var.prWin*(input_var.sumPot_all + input_var.sumBet_all + input_var.sumMinBet_all) - input_var.pot - input_var.bet - min(input_var.minBet,input_var.chips)
        input_var['util_raise_coeff']  = input_var.prWin*input_var.N_canraise - 1
        #
        #-- Worst case scenario utility (everyone afterwards folds, becomes a dual in the worst case) --#
        input_var['util_call2']  = input_var.prWin*(input_var.sumPot_all + input_var.sumBet_all) - input_var.pot - input_var.bet - min(input_var.minBet,input_var.chips)
        input_var['util_raise_coeff2']  = input_var.prWin*2 - 1
        #
        #-- Betting limit heuristic --#
        BANKRUPT_TOL  = 0.05
        input_var['maxRoundBet'] = (input_var.chips + (2 - input_var.reld)*1000)*np.log(1-input_var.prWin)/np.log(BANKRUPT_TOL)
        #
        if input_var.minBet > 0:
            # Need to "pay" to stay in game
            if input_var.util_fold > input_var.util_call:
                resp = ('fold',0)
            elif input_var.util_raise_coeff > 0:
                resp = ('raise',0)
            else:
                resp = ('call',0)
        else:
            # Can stay in the game for free
            if input_var.util_raise_coeff > 0:
                resp = ('raise',0)
            else:
                resp = ('check',0)
        #
        input_var['action'] = resp[0]
        input_var['amount'] = resp[1]
        return resp,input_var
    elif event in ('__new_round','__deal'):
        board  = pkr_to_cards(data['table']['board'])
        N      = len(data['players'])
        # print(data)
        for x in data['players']:
            if x['playerName']==name_md5:
                # print(x)
                hole = pkr_to_cards(x['cards']) if 'cards' in x else pkr_to_cards([])
                break
        calculate_win_prob_mp_start(N,hole,board,n_jobs=3)
    elif event in ('__round_end','__game_over'):
        calculate_win_prob_mp_stop()

if __name__ == '__main__':
    doListen(url,name,agent_jyp,True)
