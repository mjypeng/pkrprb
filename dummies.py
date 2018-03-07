#! /usr/bin/env python
# -*- coding:utf-8 -*-

from agent_common import *
import random

name = sys.argv[1]
mode = sys.argv[2] # 'folder','caller','allin','fixed','random'
url  = 'ws://allhands2018-training.dev.spn.a1q7.net:3001'
url  = 'ws://allhands2018-beta.dev.spn.a1q7.net:3001'

def takeAction(ws,event,data):
    if event in ('__bet','__action'):
        # time.sleep(1.5)
        # print(json.dumps(data,indent=4))
        N      = len(data['game']['players'])
        hole   = pkr_to_str(data['self']['cards'])
        board  = pkr_to_str(data['game']['board'])
        pot    = 0
        maxbet = 0
        for x in data['game']['players']:
            pot  += x['roundBet']
            if maxbet < x['bet']:
                maxbet = x['bet']
        print("N = %d" % N)
        print("Hole: [%s]" % ' '.join(hole))
        print("Board: [%s]" % ' '.join(board))
        print("Pot: %d" % pot)
        print("MaxBet: %d" % maxbet)
        print()
        #
        if mode == 'folder':
            resp  = ('check',0) if event=='__bet' or data['self']['bet']>=maxbet else ('fold',0)
        elif mode == 'caller':
            resp  = ('check',0)
        elif mode == 'allin':
            resp  = ('check',0) if event=='__bet' or random.random()<0.5 else ('allin',0)
        elif mode == 'fixed':
            bet   = data['game']['bigBlind']['amount']
            resp  = ('bet',bet) if event=='__bet' else (('fold',0) if bet<maxbet else ('check',0))
        elif mode == 'random':
            bet   = int(2*data['game']['bigBlind']['amount']*random.random())
            resp  = ('bet',bet) if event=='__bet' else (('fold',0) if bet<maxbet else ('check',0))
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
