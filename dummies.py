#! /usr/bin/env python
# -*- coding:utf-8 -*-

import agent_common
from agent_common import *
import random

name = sys.argv[1]
mode = sys.argv[2] # 'fold','call','allin','fixed','random'
url  = 'ws://allhands2018-training.dev.spn.a1q7.net:3001'
url  = 'ws://allhands2018-beta.dev.spn.a1q7.net:3001'

def takeAction(ws,event,data):
    if event in ('__bet','__action'):
        #
        if agent_common.GLOBAL_GAME is None:
            build_game_info(data['game']['players'],data['game'],name_md5=data['self']['playerName'])
        else:
            update_game_info(data['game']['players'],data['game'],name_md5=data['self']['playerName'])
        #-- Calculate Win Probability --#
        N      = len(agent_common.GLOBAL_GAME)
        hole   = pkr_to_cards(data['self']['cards'])
        board  = pkr_to_cards(data['game']['board'])
        PrWin,PrWinStd = calculate_win_prob(N,hole,board,Nsamp=20)
        #
        #-- Calculate Basic Stats --#
        Nnf    = len([x for x in data['game']['players'] if not x['folded']])
        Pot    = agent_common.GLOBAL_GAME.pot.sum()
        MaxBet = agent_common.GLOBAL_GAME.bet.max()
        util_fold  = -data['self']['roundBet'] - data['self']['bet']
        util_call  = PrWin*Pot + PrWin*MaxBet*Nnf - data['self']['roundBet'] - MaxBet
        util_call2 = PrWin*Pot + PrWin*MaxBet*2 - data['self']['roundBet'] - MaxBet
        util_raise_coeff  = PrWin*N - 1
        util_raise_coeff2 = PrWin*2 - 1
        #
        print("Table %s: Round %s: Board [%s]: Action ==> Bet >= %d?" % (data['tableNumber'],data['game']['roundName'],' '.join(pkr_to_str(data['game']['board'])),data['self']['minBet']))
        print(agent_common.GLOBAL_GAME)
        print()
        print("Win Prob:   %-3.2f%% (%-3.2f%%)" % (100*PrWin,100*PrWinStd))
        print("Fold:       %-3d" % util_fold)
        print("Check/Call: %-3.2f  (%-3.2f)" % (util_call,util_call2))
        print("Bet/Raise Coeff: %.3f (%.3f)" % (util_raise_coeff,util_raise_coeff2))
        print()
        #
        inertia  = 0.5
        if mode == 'fold':
            if event=='__bet' or data['self']['bet']>=maxbet or random.random()>inertia:
                resp  = ('check',0)
            else:
                resp  = ('fold',0)
        elif mode == 'call':
            resp  = ('check',0)
        elif mode == 'allin':
            resp  = ('check',0) if event=='__bet' or random.random()>inertia else ('allin',0)
        elif mode == 'fixed':
            bet   = 2*data['game']['bigBlind']['amount']
            resp  = ('bet',bet) # if event=='__bet' else (('fold',0) if bet<maxbet else ('check',0))
        elif mode == 'random':
            bet   = int(4*data['game']['bigBlind']['amount']*random.random())
            resp  = ('bet',bet) if random.random()>(inertia**5) else (('allin',0) if random.random()>0.4 else ('fold',0))
        elif mode == 'basic':
            if data['self']['minBet'] > 0:
                # Need to "pay" at least data['self']['minBet'] to stay in game
                if util_fold > util_call:
                    resp = ('fold',0)
                elif util_raise_coeff > 0:
                    resp = ('raise',0)
                else:
                    resp = ('call',0)
            else:
                # Can stay in the game for free
                if util_raise_coeff > 0:
                    resp = ('raise',0)
                else:
                    resp = ('check',0)
        #
        ws.send(json.dumps({
            'eventName': '__action',
            'data': {
                'playerName': name,
                'action': resp[0],
                'amount': resp[1],
                }
            }))

if __name__ == '__main__':
    doListen(url,name,takeAction)
