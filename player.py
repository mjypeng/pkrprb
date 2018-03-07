#! /usr/bin/env python
# -*- coding:utf-8 -*-

from agent_common import *

url  = 'ws://allhands2018-training.dev.spn.a1q7.net:3001'
url  = 'ws://allhands2018-beta.dev.spn.a1q7.net:3001'

name = 'jyp'
m    = hashlib.md5()
m.update(name.encode('utf8'))
name_md5 = m.hexdigest()

names_op = []

def takeAction(ws,event,data):
    if event in ('__bet','__action'):
        print(json.dumps(data,indent=2))
        N      = len(data['game']['players'])
        hole   = pkr_to_cards(data['self']['cards'])
        board  = pkr_to_cards(data['game']['board'])
        pot    = 0
        maxbet = 0
        for x in data['game']['players']:
            pot  += x['roundBet']
            if maxbet < x['bet']:
                maxbet = x['bet']
        #
        deck0  = new_deck()
        deck0  = deck0[~deck0.c.isin(hole.c)]
        #
        draw_flop  = len(board) < 3
        draw_turn  = len(board) < 4
        draw_river = len(board) < 5
        #
        if not draw_flop:
            flop   = board.iloc[:3]
            deck0  = deck0[~deck0.c.isin(flop.c)]
        if not draw_turn:
            turn   = board.iloc[3:4]
            deck0  = deck0[~deck0.c.isin(turn.c)]
        if not draw_river:
            river  = board.iloc[4:5]
            deck0  = deck0[~deck0.c.isin(river.c)]
        #
        t0      = time.clock()
        Nsamp   = 50
        results = [] #pd.DataFrame(columns=('score','rank','pot','winner'))
        for j in range(Nsamp):
            deck  = deck0.copy()
            #
            if draw_flop:  flop  = draw(deck,3)
            if draw_turn:  turn  = draw(deck)
            if draw_river: river = draw(deck)
            holes_op = [draw(deck,2) for _ in range(N-1)]
            #
            score  = score_hand(pd.concat([hole,flop,turn,river]))
            resj   = pd.Series() #pd.DataFrame(columns=('score','hand'))
            resj.loc['you'] = score[0]
            for i in range(N-1):
                scoresi = score_hand(pd.concat([holes_op[i],flop,turn,river]))
                resj.loc[i] = scoresi[0]
            #
            if resj.loc['you'] == resj.max():
                Nrank1  = (resj==resj.max()).sum()
                results.append(1/Nrank1)
            else:
                results.append(0)
        #
        pot_hat = np.mean(results)
        print(time.clock() - t0)
        print()
        print("N = %d" % N)
        print("Round %s" % data['game']['roundName'])
        print("Hole: [%s]" % cards_to_str(hole.c))
        print("Board: [%s]" % cards_to_str(board.c))
        print("Pot: %d" % pot)
        print("MaxBet: %d" % maxbet)
        print("Base prob: %.2f%%" % (100/N))
        print("Win prob: %.2f%%" % (100*pot_hat))
        print()
        #
        # data['game']['roundName'] == 'Deal','Flop','Turn','River'
        #
        if event == '__bet': # We are first player on this round
            resp  = ('bet',2*data['game']['bigBlind']['amount']) if pot_hat>0.85 else (('check',0) if pot_hat>0.7 else (('check',0) if maxbet-data['self']['bet']<data['game']['bigBlind']['amount'] else ('fold',0)))
        elif event == '__action':
            resp  = ('raise',0) if pot_hat>0.85 else (('check',0) if pot_hat>0.7 else (('check',0) if maxbet-data['self']['bet']<data['game']['bigBlind']['amount'] else ('fold',0)))
        #
        print("Action: ",resp[0],resp[1])
        #
        ws.send(json.dumps({
            'eventName': '__action',
            'data': {
                'playerName': name,
                'action': resp[0],
                'amount': resp[1],
                }
            }))
    # elif event in ('__new_round','_join','__deal','__round_end'):
    #     print("Table %s: Round %s:"%(data['table']['tableNumber'],data['table']['roundName']))
    #     print("Board [%s]" % ' '.join(cards_to_str(data['table']['board'])))
    #     SB_name = data['table']['smallBlind']['playerName']
    #     BB_name = data['table']['bigBlind']['playerName']
    #     for x in data['players']:
    #         print("%32s %6d %6d %s [%s]" % (
    #             'Me' if x['playerName']==name_md5 else x['playerName'],
    #             x['chips'],
    #             x['bet'],
    #             'SB' if SB_name==x['playerName'] else ('BB' if BB_name==x['playerName'] else '  '),
    #             ' '.join(cards_to_str(x['cards'])) if not x['folded'] else 'Folded',
    #             ))
    #     print()

if __name__ == '__main__':
    doListen(url,name,takeAction)
