#! /usr/bin/env python
# -*- coding:utf-8 -*-

from agent_common import *
import random

name = sys.argv[1]
mode = sys.argv[2] # 'random', 'basic'
record = bool(int(sys.argv[3])) if len(sys.argv)>3 else False
url  = 'ws://allhands2018-training.dev.spn.a1q7.net:3001'
url  = 'ws://allhands2018-beta.dev.spn.a1q7.net:3001'

def agent_basic(event,data):
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

def agent_random(event,data):
    if event not in ('__action','__bet'): return None
    inertia  = 0.5
    # if mode == 'fold':
    #     if event=='__bet' or data['self']['bet']>=maxbet or random.random()>inertia:
    #         resp  = ('check',0)
    #     else:
    #         resp  = ('fold',0)
    # elif mode == 'call':
    #     resp  = ('check',0)
    # elif mode == 'allin':
    #     resp  = ('check',0) if event=='__bet' or random.random()>inertia else ('allin',0)
    # elif mode == 'fixed':
    #     bet   = 2*data['game']['bigBlind']['amount']
    #     resp  = ('bet',bet) # if event=='__bet' else (('fold',0) if bet<maxbet else ('check',0))
    # elif mode == 'random':
    bet   = int(4*data['game']['bigBlind']['amount']*random.random())
    resp  = ('bet',bet) if random.random()>(inertia**5) else (('allin',0) if random.random()>0.4 else ('fold',0))
    input_var = pd.Series()
    input_var['action'] = resp[0]
    input_var['amount'] = resp[1]
    return resp,input_var

if __name__ == '__main__':
    if mode == 'basic':
        agent  = agent_basic
    else:
        agent  = agent_random
    doListen(url,name,agent,record)
