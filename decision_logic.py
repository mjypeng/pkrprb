from agent_common import *

def decision_logic(state):
    """
    state.smallBlind
    state.forced_bet
    state.roundName - data['game']['roundName'].lower()
    state.N      - number of surviving players
    state.Nnf    - number of non-folded players
    state.Nallin - number of all in players
    state.first  - event == '__bet'
    state.hole   - data['self']['cards']
    state.board  - data['game']['board']
    state.Nsim   - number of Monte Carlo samples
    state.prWin  - hand win probability
    state.prWinStd - hand win probability St.D.
    state.chips  - data['self']['chips']
    state.reloadCount - data['self']['reloadCount']
    state.pot    - self.roundBet
    state.bet    - self.bet
    state.minBet
    state.cost_to_call - min(state.minBet,state.chips)
    state.pot_sum - players.roundBet.sum()
    state.bet_sum - np.minimum(players.bet,state.bet + state.cost_to_call).sum()
    state.NMaxBet - number of players that matched largest bet
    """
    return [prFold,prCheck/Call,prBet/Raise,BetAmount] # prAllIn = 1 - prFold - prCheck - prBet

def basic_logic(state):
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

def player4_logic(state):
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

def michael_logic(state):
    C   = state.pot + state.bet
    P   = state.pot_sum + state.bet_sum
    B0  = state.cost_to_call
    state['thd_call']  = (B0 - state.forced_bet)/(P + B0)
    bet_amt_range      = np.array([0.25*P,0.5*P,0.75*P,P,1.25*P,1.5*B0,2*B0,2.5*B0]).round().astype(int)
    bet_amt_range      = bet_amt_range[(bet_amt_range>B0)&(bet_amt_range<state.chips)]
    bet_amt  = np.random.choice(bet_amt_range) if len(bet_amt_range) > 0 else 0
    state['bet_amt']  = bet_amt
    #
    if B0 > 0:
        # Need to pay "cost_to_call" to stay in game
        if state.prWin_adj < state.thd_call:
            pr_bluff  = 0.1 if state.bluff_freq>0 else 0
            return [1-pr_bluff,0,pr_bluff,bet_amt]
        else:
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
            return [0,1-pr_bet-pr_allin,pr_bet,bet_amt]
    else: # B0 == 0, i.e. can stay in the game for free
        if state.prWin_adj >= 0.5:
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
            return [0,1-pr_bet-pr_allin,pr_bet,bet_amt]
        else: # state.prWin_adj < 0.5
            pr_bluff  = 0.1 if state.bluff_freq>0 else 0
            return [0,1-pr_bluff,pr_bluff,bet_amt]
