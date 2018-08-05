
def stt_early_preflop(state):
    # Single-Table Tournament Early Phase Basic Pre-Flop Play
    # https://www.pokerstarsschool.com/article/SNG-Poker-Tournament-Early-Beginning
    if state.cards_category <= 2:
        if state.NRraise > 0:
            play  = 'reraise'
        else:
            play  = 'raise'
    elif state.cards_category <= 4:
        if (state.position_feature=='L' or state.position=='SB') and state.NRraise == 0:
            play  = 'raise'
        else:
            play  = 'fold'
    else:
        play  = 'fold'
    #
    if play == 'reraise':
        bet_amount  = 4*state.cost_to_call
    elif play == 'raise':
        bet_amount  = (4 + state.NRcall)*2*state.smallBlind
    else:
        bet_amount  = 0
    #
    if bet_amount > state.chips/3:
        bet_amount  = state.chips
    #
    return play,bet_amount

def stt_middle_preflop(state):
    # Single-Table Tournament Middle Phase Basic Pre-Flop Play
    if state.prev_action == 'bet/raise/allin' and state.NRraise > 0:
        pot         = state.pot_sum + state.bet_sum
        minBet      = state.chips if minBet > state.chips/3 else minBet
        pot_odds    = pot/minBet
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

def stt_early_preflop_pairs(state):
    if state.cards_pair:
        bet_commit  = state.bet + 4*max(state.minBet,2*state.smallBlind)
        call_commit = state.bet + state.minBet
        max_commit  = int((state.cards_rank1/100)*state.chips)
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
