import numpy as np

# Hole Cards Category
# Category 1: AA, KK
# Category 2: QQ, AKs, AKo, JJ
# Category 3: AQs, AQo, TT, 99
# Category 4: AJs, KQs, 88, 77
# Category 5: AJo, ATs, ATo, KQo, KJs, 66, 55
# Category 6: A9s-A2s, KJo, KTs, QJs, QTs, JTs, 44, 33, 22
# Category 7: A9-A2, KTo, QJo, QTo, JTo, T9s, 98s, 87s, 76s, 65s, 54s
# Category 8: K9s, K9o, K8s, K8o, Q9s, Q8s, J9s, T8s, T9o, 97s, 98o, 86s, 87o, 75s, 76o, 64s

def stt_early_preflop(state,allin_thd=3,raise_thd=5):
    # Single-Table Tournament Early Phase Basic Pre-Flop Play
    # https://www.pokerstarsschool.com/article/SNG-Poker-Tournament-Early-Beginning
    if state.cards_category <= allin_thd:
        if state.NRraise > 0:
            play  = 'reraise'
        else:
            play  = 'raise'
    elif state.cards_category <= raise_thd:
        if (state.position_feature=='L' or state.position=='SB') and state.NRraise == 0:
            play  = 'raise'
        else:
            play  = 'fold'
    else:
        play  = 'fold'
    #
    if play == 'reraise':
        bet_amount  = 4*state.minBet
    elif play == 'raise':
        bet_amount  = (4 + state.NRcall)*2*state.smallBlind
    else:
        bet_amount  = 0
    #
    if bet_amount > state.chips/3:
        bet_amount  = state.chips
    #
    return play,bet_amount

def stt_preflop_pairs(state,multiple=2):
    if state.cards_pair:
        bet_commit  = state.bet + 4*max(state.minBet,2*state.smallBlind)
        call_commit = state.bet + state.minBet
        max_commit  = int((multiple*state.cards_rank1/100)*state.chips)
        if max_commit >= bet_commit:
            play        = 'raise'
            bet_amount  = 4*max(state.minBet,2*state.smallBlind)
        elif max_commit >= call_commit:
            play        = 'call'
            bet_amount  = 0
        else:
            play        = 'fold'
            bet_amount  = 0
    else:
        play        = 'fold'
        bet_amount  = 0
    #
    return play,bet_amount

def stt_middle_preflop(state):
    # Single-Table Tournament Middle Phase Basic Pre-Flop Play
    if state.prev_action == 'bet/raise/allin' and state.NRraise > 0:
        #-- Somebody Re-raised after we raised --#
        pot         = state.pot_sum + state.bet_sum
        self_minBet = state.chips if state.minBet>state.chips/3 else state.minBet
        pot_odds    = pot/self_minBet
        #
        if pot_odds >= 2.5:
            play  = 'call'
        elif pot_odds >= 2:
            play  = 'call' if state.cards_category<=5 else 'fold'
        elif pot_odds >= 1.5:
            play  = 'call' if state.cards_category<=4 else 'fold'
        else:
            play  = 'call' if state.cards_category<=3 else 'fold'
    else:
        cards_category_thd  = 9 - (state.chips > 10*state.smallBlind) - (state.chips > 20*state.smallBlind)
        if state.NRraise > 0:
            cards_category_thd  -= 4
        elif state.NRcall > 0:
            cards_category_thd  -= 3
        else: # Everyone folds to you
            if state.position_feature == 'E' or state.position == 'BB':
                cards_category_thd  -= 2
            elif state.position_feature == 'M':
                cards_category_thd  -= 1
        #
        cards_category_thd  = min(cards_category_thd,8)
        if state.cards_category <= cards_category_thd:
            play  = 'raise'
        else:
            play  = 'fold'
    #
    # Never just call unless the other player is all-in
    # Never limp
    if play == 'raise':
        bet_amount  = 4*2*state.smallBlind
    else:
        bet_amount  = 0
    #
    if bet_amount > state.chips/3:
        bet_amount  = state.chips
    #
    return play,bet_amount

def stt_late_preflop_allin(state,raise_thd=5,call_thd=0):
    if state.Nallin > 1:
        play        = 'fold'
        bet_amount  = 0
    elif state.Nallin > 0:
        chips_BB  = np.ceil(state.chips / (2*state.smallBlind))
        if state.chips > 3*state.minBet or state.chips < 0.33*state.minBet:
            # Big or Short Stack
            if chips_BB > 10:
                cards_category_thd  = 0
            else:
                cards_category_thd  = call_thd + 2 + (state.position=='BB') + (chips_BB<8) + (chips_BB<6) + (chips_BB<4) + (chips_BB<2)
                if cards_category_thd >= 7: cards_category_thd = 9
        else:
            # Medium Stack
            if state.chips >= 10*2*state.smallBlind:
                cards_category_thd  = 0
            else:
                cards_category_thd  = call_thd + (state.position=='BB') + (state.chips<8*2*state.smallBlind) + (state.chips<6*2*state.smallBlind) + (state.chips<4*2*state.smallBlind) + (state.chips<2*2*state.smallBlind)
                if cards_category_thd == 0: cards_category_thd = 1
        #
        play       = 'allin' if state.cards_category<=cards_category_thd else 'fold'
        bet_amount = state.chips if play == 'allin' else 0
    elif state.chips > 3*state.op_chips_max:
        # Big Stack
        if state.position_feature == 'L' or state.position == 'SB':
            if state.cards_category <= 8:
                play        = 'allin'
                bet_amount  = state.chips
            else:
                play        = 'fold'
                bet_amount  = 0
        else:
            play        = 'fold'
            bet_amount  = 0
    elif state.chips <= 2*2*state.smallBlind:
        play        = 'allin'
        bet_amount  = state.chips
    else:
        if state.position in ('D','SB',):
            if state.cards_category <= raise_thd + 1:
                play        = 'allin'
                bet_amount  = state.chips
            else:
                play        = 'fold'
                bet_amount  = 0
        elif state.position_feature == 'L':
            if state.cards_category <= raise_thd:
                play        = 'allin'
                bet_amount  = state.chips
            else:
                play        = 'fold'
                bet_amount  = 0
        else:
            play        = 'fold'
            bet_amount  = 0
    #
    return play,bet_amount
