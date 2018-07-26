from agent_common import *
from sklearn.externals import joblib

def decision_logic(state,prev_state=None):
    """
    -- Table State --
    state.tableNumber                    572
    state.game_id          20180726113059766
    state.round_id                         1
    state.smallBlind                      10
    state.roundName                     Deal
    state.forced_bet                       0
    state.N_effective - number of players for win probability calculation
    state.N      - number of surviving players
    state.Nnf    - number of non-folded players
    state.Nallin - number of all in players
    state.first  - event == '__bet'
    -- Current Hand --
    state.hole                         7C 4D
    state.board               5S 8C AC 4S 7C
    state.cards_rank1                      7
    state.cards_rank2                      4
    state.cards_aces                       0
    state.cards_faces                      0
    state.cards_pair                       0
    state.cards_suit                       0
    state.cards_conn                       0
    state.hand_score0                      0
    state.hand_score1                      7
    -- Betting State --
    state.chips       - data['self']['chips']
    state.position    - 1~7 or D,SB,BB
    state.pot    - self.roundBet
    state.bet    - self.bet
    state.minBet
    state.cost_to_call - min(state.minBet,state.chips)
    state.pot_sum - players.roundBet.sum()
    state.bet_sum - np.minimum(players.bet,state.bet + state.cost_to_call).sum()
    state.NMaxBet - number of players that matched largest bet
    -- Opponent Action --
    state.Nfold   - number of fold for current betting round
    state.Ncall   - number of check/call for current betting round
    state.Nraise  - number of bet/raise/allin for current betting round
    state.self_Ncall  - self number of check/call for current betting round
    state.self_Nraise - self number of bet/raise/allin for current betting round
    state.prev_action - self previous action for current betting round: 'none', 'check/call' or 'bet/raise/allin'
    state.NRfold  - number of fold for current betting round in response to player action
    state.NRcall  - number of check/call for current betting round in response to player action
    state.NRraise - number of bet/raise/allin for current betting round in response to player action
    -- Monte Carlo Simulation --
    state.Nsim     - number of Monte Carlo samples
    state.prWin    - hand win probability
    state.prWinStd - hand win probability St.D.
    --
    state.logic   - Specifies decision logic function
    --
    state.resp    - Stores decision logic function return value (only available for previous state)
    state.action  - Final action taken (only available for previous state)
    state.amount  - Final action amount (only available for previous state)
    """
    return [prFold,prCheck/Call,prBet/Raise,BetAmount] # prAllIn = 1 - prFold - prCheck - prBet

def basic_logic(state,prev_state=None):
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

def player4_logic(state,prev_state=None):
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

def michael_logic(state,prev_state=None):
    state['prev_round']   = prev_state.roundName if prev_state is not None else None
    state['prev_play']    = prev_state.play if prev_state is not None else None
    state['prev_action']  = prev_state.action if prev_state is not None else None
    C   = state.pot + state.bet
    P   = state.pot_sum + state.bet_sum
    B0  = state.cost_to_call
    state['thd_call']  = (B0 - state.forced_bet)/(P + B0)
    bet_amt_range      = np.array([0.25*P,0.5*P,P,1.5*B0,2*B0,2.5*B0]).round().astype(int) #,0.75*P,1.25*P
    bet_amt_range      = bet_amt_range[(bet_amt_range>B0)&(bet_amt_range<state.chips)]
    bet_amt  = np.random.choice(bet_amt_range) if len(bet_amt_range) > 0 else 0
    state['bet_amt']  = bet_amt
    #
    if B0 > 0:
        # Need to pay "cost_to_call" to stay in game
        if state.prWin_adj < state.thd_call:
            state['play']  = 'bluff'
            pr_bluff  = 0.1 if state.bluff_freq>0 else 0
            return [1-pr_bluff,0,pr_bluff,bet_amt]
        else:
            state['play']  = 'value'
            if state.prWin_adj >= 0.9:
                pr_bet    = 0.1
                pr_allin  = 0.8
            elif state.prWin_adj >= 0.8:
                pr_bet    = 0.2
                pr_allin  = 0.6
            elif state.prWin_adj >= 0.7:
                pr_bet    = 0.3
                pr_allin  = 0.4
            elif state.prWin_adj >= 0.6:
                pr_bet    = 0.3
                pr_allin  = 0.15
            else:
                pr_bet    = 0.25
                pr_allin  = 0.05
            if state.roundName=='deal': pr_allin = max(pr_allin - 0.1,0)
            return [0,1-pr_bet-pr_allin,pr_bet,bet_amt]
    else: # B0 == 0, i.e. can stay in the game for free
        if state.prWin_adj >= 0.5:
            state['play']  = 'value'
            if state.prWin_adj >= 0.9:
                pr_bet    = 0.1
                pr_allin  = 0.8
            elif state.prWin_adj >= 0.8:
                pr_bet    = 0.2
                pr_allin  = 0.65
            elif state.prWin_adj >= 0.7:
                pr_bet    = 0.3
                pr_allin  = 0.5
            elif state.prWin_adj >= 0.6:
                pr_bet    = 0.4
                pr_allin  = 0.2
            else:
                pr_bet    = 0.3
                pr_allin  = 0.1
            if state.roundName=='deal': pr_allin = max(pr_allin - 0.1,0)
            return [0,1-pr_bet-pr_allin,pr_bet,bet_amt]
        else: # state.prWin_adj < 0.5
            state['play']  = 'bluff'
            pr_bluff  = 0.1 if state.bluff_freq>0 else 0
            return [0,1-pr_bluff,pr_bluff,bet_amt]

def michael2_logic(state,prev_state=None):
    state['prev_round']   = prev_state.roundName if prev_state is not None else None
    state['prev_play']    = prev_state.play if prev_state is not None else None
    state['prev_action']  = prev_state.action if prev_state is not None else None
    if prev_state is not None:
        state['prev_prWin']  = prev_state.prWin if prev_state.roundName!=state.roundName else prev_state.prev_prWin
        state['prWin_delta'] = state.prWin - state.prev_prWin if state.prev_prWin is not None else 0
    else:
        state['prev_prWin']  = None
        state['prWin_delta'] = 0
    C   = state.pot + state.bet
    P   = state.pot_sum + state.bet_sum
    B0  = state.cost_to_call
    state['thd_call']  = (B0 - state.forced_bet)/(P + B0)
    bet_amt  = int((np.random.random()*3 + 0.5)*B0) if np.random.random()>0.4 else 'raise'
    state['bet_amt']  = bet_amt
    #
    if B0 > 0:
        # Need to pay "cost_to_call" to stay in game
        if state.prWin_adj < state.thd_call and state.prWin_delta < 0.1:
            state['play']  = 'bluff'
            pr_bluff  = 0 if state.Nallin==0 else 0
            return [1-pr_bluff,0,pr_bluff,bet_amt]
        else:
            state['play']  = 'value'
            if state.prWin_adj >= 0.9:
                pr_bet    = 0.1
                pr_allin  = 0.9
            elif state.prWin_adj >= 0.8:
                pr_bet    = 0.25
                pr_allin  = 0.7
            elif state.prWin_adj >= 0.7:
                pr_bet    = 0.5
                pr_allin  = 0.2
            elif state.prWin_adj >= 0.6:
                pr_bet    = 0.4
                pr_allin  = 0.05
            else:
                pr_bet    = 0.3
                pr_allin  = 0
            if state.roundName=='deal': pr_allin = max(pr_allin - 0.1,0)
            return [0,1-pr_bet-pr_allin,pr_bet,bet_amt]
    else: # B0 == 0, i.e. can stay in the game for free
        if state.prWin_adj >= 0.5:
            state['play']  = 'value'
            if state.prWin_adj >= 0.9:
                pr_bet    = 0.15
                pr_allin  = 0.85
            elif state.prWin_adj >= 0.8:
                pr_bet    = 0.25
                pr_allin  = 0.65
            elif state.prWin_adj >= 0.7:
                pr_bet    = 0.55
                pr_allin  = 0.25
            elif state.prWin_adj >= 0.6:
                pr_bet    = 0.45
                pr_allin  = 0.1
            else:
                pr_bet    = 0.35
                pr_allin  = 0.05
            if state.roundName=='deal': pr_allin = max(pr_allin - 0.1,0)
            return [0,1-pr_bet-pr_allin,pr_bet,bet_amt]
        else: # state.prWin_adj < 0.5
            state['play']  = 'bluff'
            pr_bluff  = 0 if state.Nallin==0 else 0
            return [0,1-pr_bluff,pr_bluff,bet_amt]

MODEL_PRWINMONEY  = joblib.load('pkrprb_winMoney_rf.pkl')

def michael3_logic(state,prev_state=None):
    global MODEL_PRWINMONEY
    #
    if prev_state is not None:
        state['prev_prWin']  = prev_state.prWin if prev_state.roundName!=state.roundName else prev_state.prev_prWin
        state['prWin_delta'] = state.prWin - state.prev_prWin if state.prev_prWin is not None else 0
    else:
        state['prev_prWin']  = None
        state['prWin_delta'] = 0
    #
    if state.prWin > 0.9:
        return [0,0,0.1,'raise']
    elif state.prWin > 0.1:
        ['N', 'Nnf', 'Nallin', 'NMaxBet', 'Nfold', 'Ncall', 'Nraise', 'self_Ncall', 'self_Nraise', 'NRfold', 'NRcall', 'NRraise', 'cards_rank1', 'cards_rank2', 'cards_aces', 'cards_faces', 'cards_pair', 'cards_suit', 'cards_conn', 'hand_score0', 'hand_score1', 'prWin', 'Deal', 'Flop', 'Turn', 'River', 'pos=E', 'pos=M', 'pos=L', 'pos=B', 'op_resp=none', 'op_resp=all_folded', 'op_resp=any_called', 'op_resp=any_raised', 'op_resp=any_reraised', 'pot_P', 'pot_SB', 'bet_P', 'bet_SB', 'bet_sum_P', 'bet_sum_SB', 'minBet_P', 'minBet_SB', 'prev=none', 'prev=check/call', 'prev=bet/raise/allin', 'action=check/call', 'action=bet/raise/allin', 'amount_P', 'amount_SB']
        return [0.2,0.4,0.35,'raise']
    else:
        return [1,0,0,0]
