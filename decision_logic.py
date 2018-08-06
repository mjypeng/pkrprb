from common import *
from expert_rules import *
import os
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
    state.blind_level                      1
    state.game_N - number of players
    state.N      - number of surviving players
    state.game_phase  - 'Early': blind_level 
    state.Nnf    - number of non-folded players
    state.Nallin - number of all in players
    state.first  - event == '__bet', can check
    -- Current Hand --
    state.hole                         7C 4D
    state.board               5S 8C AC 4S 7C
    state.cards_rank1                      7
    state.cards_rank2                      4
    state.cards_rank_sum                  20
    state.cards_aces                       0
    state.cards_faces                      0
    state.cards_pair                       0
    state.cards_suit                       0
    state.cards_conn                       0
    state.cards_conn2                      1
    state.cards_category                   8
    state.hand_score0                      0
    state.hand_score1                      7
    state.hand_score2                      9
    -- Betting State --
    state.chips       - data['self']['chips']
    state.position    - 1~7 or D,SB,BB
    state.position_feature  - E,M,L or B
    state.pot    - self.roundBet
    state.bet    - self.bet
    state.minBet
    state.cost_to_call - min(state.minBet,state.chips)
    state.pot_sum - players.roundBet.sum()
    state.bet_sum - np.minimum(players.bet,state.bet + state.cost_to_call).sum()
    state.maxBet  - maximum of players bet
    state.NMaxBet - number of players that matched largest bet
    state.op_chips_max  - largest stack of non-folded opponent
    state.op_chips_min  - smallest stack of non-folded opponent
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
    state.op_resp - 'any_reraised', 'any_raised', 'any_called', 'all_folded', or 'none'
    -- Monte Carlo Simulation --
    state.Nsim     - number of Monte Carlo samples
    state.prWin    - hand win probability
    state.prWinStd - hand win probability St.D.
    state.prWin_delta  - prWin difference since last betting round
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

MODEL_PRWINMONEY  = {
    '20180727': joblib.load('pkrprb_winMoney_rf_20180727.pkl') if os.path.isfile('pkrprb_winMoney_rf_20180727.pkl') else None,
    '20180731': joblib.load('pkrprb_winMoney_rf_20180731.pkl') if os.path.isfile('pkrprb_winMoney_rf_20180731.pkl') else None,
    }

def michael3_logic(state,prev_state=None):
    global MODEL_PRWINMONEY
    model_prWinMoney  = MODEL_PRWINMONEY['20180727']
    #
    BET_MINIMUM  = {
        'Deal':  state.cost_to_call + 2*state.smallBlind,
        'Flop':  state.cost_to_call + 2*state.smallBlind,
        'Turn':  state.cost_to_call + 2*state.smallBlind,
        'River': state.cost_to_call + 2*state.smallBlind,
        }
    BET_LIMIT  = {
        'Deal':  state.chips*(0.1+state.prWin),
        'Flop':  state.chips*(state.prWin>0.5),
        'Turn':  state.chips,
        'River': state.chips,
        }
    state['bet_minimum']  = BET_MINIMUM[state.roundName]
    state['bet_limit']    = BET_LIMIT[state.roundName]
    #
    state['thd_call']  = (state.cost_to_call - state.forced_bet)/(state.pot_sum + state.bet_sum + state.cost_to_call) # This value <= 50%
    #
    if state.roundName == 'Deal' and state.cards_category > 8:
        #-- Don't Play Trash Hands Pre-Flop --#
        state['play']  = 'trash'
        return [0,1,0,0] if state.cost_to_call<=state.smallBlind else [1,0,0,0]
    elif (state.roundName == 'Deal' and state.cards_category == 1) or state.prWin > 0.9:
        state['play']  = 'allin'
        return [0,0,0.2,'raise']
    elif state.roundName == 'Deal' or state.prWin > state.thd_call:
        #-- Counterfactual Variables --#
        bets  = [BET_MINIMUM[state.roundName],BET_LIMIT[state.roundName]]
        bets  = np.arange(bets[0],bets[1],np.abs(bets[1]-bets[0])/20).round()
        CF    = pd.concat([
            pd.DataFrame({'action':'check/call','amount':[state.cost_to_call]}),
            pd.DataFrame({'action':'bet/raise/allin','amount':bets}),
            ],0,ignore_index=True)
        #
        #-- Compile Game State into Model Features --#
        X0  = pd.DataFrame(0.,index=CF.index,columns=model_prWinMoney['col'])
        for col in ('N','Nnf','Nallin','NMaxBet','Nfold','Ncall','Nraise','self_Ncall','self_Nraise','NRfold','NRcall','NRraise','cards_rank1','cards_rank2','cards_rank_sum','cards_aces','cards_faces','cards_pair','cards_suit','cards_conn','cards_conn2','cards_category','hand_score0','hand_score1','board_rank1','board_rank2','board_aces','board_faces','board_kind','board_kind_rank','board_suit','board_suit_rank','board_conn','board_conn_rank','prWin',):
            X0[col]  = state[col] if col in state else 0
        for roundName in ('Deal','Flop','Turn','River',):
            X0[roundName]  = state.roundName==roundName
        for pos in ('E','M','L','B',):
            X0['pos='+pos]  = state.position_feature==pos
        for op_resp in ('none','all_folded','any_called','any_raised','any_reraised',):
            X0['op_resp='+op_resp]  = state.op_resp==op_resp
        #
        P  = state.pot_sum + state.bet_sum
        X0['chips_SB']  = state.chips / state.smallBlind
        X0['chips_P']   = state.chips / P
        X0['pot_P']     = state.pot / P
        X0['pot_SB']    = state.pot / state.smallBlind
        X0['bet_P']     = state.bet / P
        X0['bet_SB']    = state.bet / state.smallBlind
        X0['bet_sum_P'] = state.bet_sum / P
        X0['bet_sum_SB'] = state.bet_sum / state.smallBlind
        X0['minBet_P']  = state.cost_to_call / P
        X0['minBet_SB'] = state.cost_to_call / state.smallBlind
        for act in ('none','check/call','bet/raise/allin',):
            X0['prev='+act]  = state.prev_action==act
        #
        #-- Compile Counterfactual Actions into Model Features --#
        X0['action=check/call']      = CF.action=='check/call'
        X0['action=bet/raise/allin'] = CF.action=='bet/raise/allin'
        X0['amount_P']  = CF.amount / P
        X0['amount_SB'] = CF.amount / state.smallBlind
        #
        #-- Predict prWinMoney --#
        CF['Nnf']         = state.Nnf
        CF['prWin']       = state.prWin
        CF['prWinMoney']  = model_prWinMoney['model'].predict_proba(X0)[:,1]
        CF['return']      = CF.prWinMoney*(P + CF.amount)/CF.amount - 1
        print(CF)
        print()
        #
        #-- Choose Best Action --#
        resp  = CF.loc[CF.prWinMoney.idxmax()]
        print(resp)
        if resp.prWinMoney < 0.55 or (state.cost_to_call>BET_LIMIT[state.roundName] and np.random.random()<0.5):
            state['play']  = 'ml_fold'
            return [1,0,0,0] if state.cost_to_call>0 else [0,1,0,0]
        elif resp.action == 'check/call':
            state['play']  = 'ml_call'
            return [0,1,0,0]
        else:
            state['play']  = 'ml_raise'
            return [0,0,1,int(resp.amount)]
    else:
        state['play']  = 'fold'
        return [1,0,0,0] if state.cost_to_call>0 else [0,1,0,0]

def michael4_logic(state,prev_state=None):
    global MODEL_PRWINMONEY
    model_prWinMoney  = MODEL_PRWINMONEY['20180731']
    #
    LIMP_AMOUNT = {
        'Deal':  4*state.smallBlind,
        'Flop':  2*state.smallBlind,
        'Turn':  0,
        'River': 0,
        }
    BET_LIMIT  = {
        'Deal':  state.chips*(0.1+state.prWin),
        'Flop':  state.chips*(state.prWin>0.5),
        'Turn':  state.chips*(state.prWin>0.5),
        'River': state.chips*(state.prWin>0.5),
        }
    #------------------------#
    #-- Predict prWinMoney --#
    #------------------------#
    #-- Counterfactual Variables --#
    bets  = [state.cost_to_call + 2*state.smallBlind,BET_LIMIT[state.roundName]]
    if bets[1] - bets[0] > 2*state.smallBlind:
        bets  = np.arange(bets[0],bets[1]+state.smallBlind,np.abs(bets[1]-bets[0])/20).round()
    else:
        bets  = [state.cost_to_call]
    #
    CF    = pd.concat([
        pd.DataFrame({'action':'check/call','amount':[state.cost_to_call]}),
        pd.DataFrame({'action':'bet/raise/allin','amount':bets}),
        ],0,ignore_index=True)
    #
    #-- Compile Game State into Model Features --#
    X0  = pd.DataFrame(0.,index=CF.index,columns=model_prWinMoney['col'])
    for col in ('blind_level','N','Nnf','Nallin','NMaxBet','Nfold','Ncall','Nraise','self_Ncall','self_Nraise','NRfold','NRcall','NRraise','op_chips_max','op_chips_min','cards_rank1','cards_rank2','cards_rank_sum','cards_aces','cards_faces','cards_pair','cards_suit','cards_conn','cards_conn2','cards_category','hand_score0','hand_score1','hand_score2','board_rank1','board_rank2','board_aces','board_faces','board_kind','board_kind_rank','board_suit','board_suit_rank','board_conn','board_conn_rank','prWin','prWin_delta',):
        X0[col]  = state[col] if col in state else 0
    for phase in ('Early','Middle','Late',):
        X0['phase='+phase]  = state.game_phase==phase
    for roundName in ('Deal','Flop','Turn','River',):
        X0[roundName]  = state.roundName==roundName
    for pos in ('E','M','L','B',):
        X0['pos='+pos]  = state.position_feature==pos
    for op_resp in ('none','all_folded','any_called','any_raised','any_reraised',):
        X0['op_resp='+op_resp]  = state.op_resp==op_resp
    #
    P  = state.pot_sum + state.bet_sum
    X0['chips_SB']  = state.chips / state.smallBlind
    X0['chips_P']   = state.chips / P
    X0['pot_P']     = state.pot / P
    X0['pot_SB']    = state.pot / state.smallBlind
    X0['bet_P']     = state.bet / P
    X0['bet_SB']    = state.bet / state.smallBlind
    X0['bet_sum_P'] = state.bet_sum / P
    X0['bet_sum_SB'] = state.bet_sum / state.smallBlind
    X0['minBet_P']  = state.cost_to_call / P
    X0['minBet_SB'] = state.cost_to_call / state.smallBlind
    X0['minBet_chips'] = state.cost_to_call / state.chips
    for act in ('none','check/call','bet/raise/allin',):
        X0['prev='+act]  = state.prev_action==act
    #
    #-- Compile Counterfactual Actions into Model Features --#
    X0['action=check/call']      = CF.action=='check/call'
    X0['action=bet/raise/allin'] = CF.action=='bet/raise/allin'
    X0['amount_P']  = CF.amount / P
    X0['amount_SB'] = CF.amount / state.smallBlind
    X0['amount_chips'] = CF.amount / state.chips
    #
    #-- Predict prWinMoney --#
    CF  = pd.concat([
        pd.DataFrame({'action':'fold','amount':[0]}),
        CF,
        ],0,ignore_index=True)
    CF['Nnf']         = state.Nnf
    CF['prWin']       = state.prWin
    CF['prWin_delta'] = state.prWin_delta
    CF['prWinMoney']  = np.r_[0,model_prWinMoney['model'].predict_proba(X0)[:,1]]
    CF['EV']          = CF.prWinMoney*(P + CF.amount) - np.minimum(CF.amount,state.op_chips_max)
    CF['EV_implied']  = CF.prWinMoney*(P + CF.amount*(state.Nnf - state.Nallin)*(1-0.3)) - np.minimum(CF.amount,state.op_chips_max)
    CF['ROI']         = CF.EV / CF.amount
    CF['ROI_implied'] = CF.EV_implied / CF.amount
    print(CF)
    print()
    #
    #-- Choose Best Action --#
    resp  = CF.loc[CF[CF.prWinMoney>CF.prWinMoney.max()-0.05].amount.idxmin()]
    # resp  = CF.loc[CF.prWinMoney.idxmax()]
    # resp  = CF.loc[CF.ROI.idxmax()]
    print(resp)
    #
    #-------------------#
    #-- Make Decision --#
    #-------------------#
    state['tight']     = state.game_phase == 'Early' and state.chips > 60*state.smallBlind
    state['thd_call']  = (state.cost_to_call - state.forced_bet)/(state.pot_sum + state.bet_sum + state.cost_to_call) # This value <= 50%
    #
    if state.roundName == 'Deal' and (state.cards_category>7 or (state.tight and state.cards_category>5)):
        #-- Don't Play Trash Hands Pre-Flop --#
        state['play']  = 'trash'
        return [0,1,0,0] if state.cost_to_call<=LIMP_AMOUNT[state.roundName] else [1,0,0,0]
    elif state.prWin > 0.9:
        state['play']  = 'allin'
        return [0,0,0.2,'raise']
    elif state.roundName == 'Deal' or state.prWin > state.thd_call:
        if resp.prWinMoney < 0.55 or state.cost_to_call > BET_LIMIT[state.roundName]:
            state['play']  = 'ml_fold'
            return [0,1,0,0] if state.cost_to_call<=LIMP_AMOUNT[state.roundName] else [1,0,0,0]
        elif resp.action == 'check/call' or state.prWin < 0.5:
            state['play']  = 'ml_call'
            return [0,1,0,0]
        else:
            state['play']  = 'ml_raise'
            return [0,0,1,int(resp.amount)]
    else:
        state['play']  = 'fold'
        return [0,1,0,0] if state.cost_to_call<=LIMP_AMOUNT[state.roundName] else [1,0,0,0]

MODEL_WIN  = {
    'Flop_20180805':  joblib.load('pkrprb_win_Flop_rf_20180805.pkl') if os.path.isfile('pkrprb_win_Flop_rf_20180805.pkl') else None,
    'Turn_20180805':  joblib.load('pkrprb_win_Turn_rf_20180805.pkl') if os.path.isfile('pkrprb_win_Turn_rf_20180805.pkl') else None,
    'River_20180805': joblib.load('pkrprb_win_River_rf_20180805.pkl') if os.path.isfile('pkrprb_win_River_rf_20180805.pkl') else None,
    }

def michael5_logic(state,prev_state=None):
    global MODEL_WIN
    #
    if state.roundName == 'Deal':
        state['stt_early_preflop']      = stt_early_preflop(state)
        state['stt_preflop_pairs']      = stt_preflop_pairs(state)
        state['stt_middle_preflop']     = stt_middle_preflop(state)
        state['stt_late_preflop_allin'] = stt_late_preflop_allin(state)
        #
        if state.game_phase == 'Early' and state.blind_level < 3:
            play,bet_amt  = state['stt_preflop_pairs'] if state['stt_early_preflop'][0]=='fold' else state['stt_early_preflop']
        elif state.game_phase != 'Late':
            play,bet_amt  = state['stt_preflop_pairs'] if state['stt_early_preflop'][0]=='fold' else state['stt_middle_preflop']
        else:
            if state['stt_late_preflop_allin'][0] == 'allin':
                play,bet_amt  = state['stt_late_preflop_allin']
                bet_amt       = 4*max(state.minBet,2*state.smallBlind)
                if bet_amt > state.chips/3:  bet_amt = state.chips
            else:
                play,bet_amt  = state['stt_preflop_pairs'] if state['stt_early_preflop'][0]=='fold' else state['stt_middle_preflop']
        #
        if play == 'fold':
            if state.cards_category <= 6 or ((state.game_phase!='Early' or state.blind_level>=3) and state.cards_category<=7):
                return [0,1,0,0] if state.cost_to_call<=2*state.smallBlind else [1,0,0,0]
            else:
                return [0,1,0,0] if state.cost_to_call<=state.smallBlind else [1,0,0,0]
        elif play == 'call':
            return [0,1,0,0]
        else:
            return [0,0,1,bet_amt]
    elif state.roundName == 'Flop':
        #-------------------------#
        #-- Flop Decision Logic --#
        #-------------------------#
        model_winNow  = MODEL_WIN['Flop_20180805']
        #
        X  = compile_features(state,model_winNow['feat'])
        state['prWinCond'] = model_winNow['model'].predict_proba(X[None,:])[0,1]
        #
        state['tight']     = state.game_phase == 'Early' and state.chips > 60*state.smallBlind
        state['thd_call']  = (state.cost_to_call - state.forced_bet)/(state.pot_sum + state.bet_sum + state.cost_to_call) # This value <= 50%
        #
        # TODO: More conservative if against Big Stacks
        #
        if state.prWinCond>0.95 and state.prWin>0.7 and state.prWin_delta>0 and state.hand_score0>=2:
            state['play']  = 'allin' # Call Any
            return [0,0,0.2,4*max(state.minBet,2*state.smallBlind)]
            #
        elif state.prWinCond>0.85 and state.prWin>0.6 and state.prWin_delta>0 and state.hand_score0>=2:
            state['play']  = 'raise/allin' # Call Any
            return [0,0,0.5,4*max(state.minBet,2*state.smallBlind)]
            #
        elif state.prWinCond>0.75 and state.prWin>0.5 and state.prWin_delta>0 and state.hand_score0>=2:
            state['play']  = 'raise/allin' # Call Any
            self_minBet    = 4*max(state.minBet,2*state.smallBlind)
            return [0,1,0,0] if self_minBet>state.chips/3 else [0,0,0.8,self_minBet]
            #
        elif state.prWinCond>0.55 and state.prWin>0.4 and state.prWin_delta>0 and state.hand_score0>=1:
            state['play']  = 'call/raise' # Call Any
            self_minBet    = 2*max(state.minBet,2*state.smallBlind)
            return [0,1,0,0] if self_minBet>state.chips/3 else [0,0.2,0.8,self_minBet]
            #
        elif state.prWinCond > max(state.thd_call,0.1) and state.prWin_delta > -0.05:
            #-- Do not pot commit --#
            if state.prWinCond > 0.4 or state.prWin_delta > 0.35 or state.hand_score0 >= 2:
                self_minBet  = 2*max(state.minBet,2*state.smallBlind)
                if state.minBet > state.chips/3:
                    state['play']  = 'fold/allin'
                    return [0.8,0,0,0]
                elif self_minBet > state.chips/3:
                    state['play']  = 'call/allin'
                    return [0,0.8,0,0]
                else:
                    state['play']  = 'call/raise'
                    return [0,0.2,0.8,self_minBet]
            elif state.minBet > state.chips/3:
                state['play']  = 'fold'
                return [1,0,0,0]
            else:
                state['play']  = 'check/call'
                return [0,1,0,0]
        else:
            state['play']  = 'fold/check'
            return [0,1,0,0] if state.cost_to_call<=0 else [1,0,0,0]
    elif state.roundName == 'Turn':
        #-------------------------#
        #-- Turn Decision Logic --#
        #-------------------------#
        model_winNow  = MODEL_WIN['Turn_20180805']
        #
        X  = compile_features(state,model_winNow['feat'])
        state['prWinCond'] = model_winNow['model'].predict_proba(X[None,:])[0,1]
        #
        state['tight']     = state.game_phase == 'Early' and state.chips > 60*state.smallBlind
        state['thd_call']  = (state.cost_to_call - state.forced_bet)/(state.pot_sum + state.bet_sum + state.cost_to_call) # This value <= 50%
        #
        if state.prWinCond > 0.98 and state.prWin > 0.7 and state.prWin_delta>0 and state.hand_score0>=3:
            state['play']  = 'allin'
            return [0,0,0.2,4*max(state.minBet,2*state.smallBlind)]
            #
        elif state.prWinCond > 0.88 and state.prWin > 0.6 and state.prWin_delta>0 and state.hand_score0>=3:
            state['play']  = 'raise/allin' # Call Any
            return [0,0,0.5,4*max(state.minBet,2*state.smallBlind)]
            #
        elif state.prWinCond > 0.75 and state.prWin > 0.5 and state.prWin_delta>0 and state.hand_score0>=3:
            state['play']  = 'raise/allin' # Call Any
            self_minBet    = 4*max(state.minBet,2*state.smallBlind)
            return [0,1,0,0] if self_minBet>state.chips/5 else [0,0,0.8,self_minBet]
        elif state.prWinCond > 0.57 and state.prWin > 0.4 and state.prWin_delta>0 and state.hand_score0>=2:
            state['play']  = 'call/raise' # Call Any
            self_minBet    = 2*max(state.minBet,2*state.smallBlind)
            return [0,1,0,0] if self_minBet>state.chips/5 else [0,0.2,0.8,self_minBet]
        elif state.prWinCond > max(state.thd_call,0.1) and state.prWin_delta > -0.1:
            #-- Do not pot commit --#
            if state.prWinCond > 0.4 or state.prWin_delta > 0.1 or state.hand_score0 >= 3:
                self_minBet  = 2*max(state.minBet,2*state.smallBlind)
                if state.minBet > state.chips/5:
                    state['play']  = 'fold/allin'
                    return [0.8,0,0,0]
                elif self_minBet > state.chips/5:
                    state['play']  = 'call/allin'
                    return [0,0.8,0,0]
                else:
                    state['play']  = 'call/raise'
                    return [0,0.2,0.8,self_minBet]
            elif state.minBet > state.chips/5:
                state['play']  = 'fold'
                return [1,0,0,0]
            else:
                state['play']  = 'check/call'
                return [0,1,0,0]
        else:
            state['play']  = 'fold/check'
            return [0,1,0,0] if state.cost_to_call<=0 else [1,0,0,0]
    elif state.roundName == 'River':
        #--------------------------#
        #-- River Decision Logic --#
        #--------------------------#
        model_winNow  = MODEL_WIN['River_20180805']
        #
        X  = compile_features(state,model_winNow['feat'])
        state['prWinCond'] = model_winNow['model'].predict_proba(X[None,:])[0,1]
        #
        state['tight']     = state.game_phase == 'Early' and state.chips > 60*state.smallBlind
        state['thd_call']  = (state.cost_to_call - state.forced_bet)/(state.pot_sum + state.bet_sum + state.cost_to_call) # This value <= 50%
        #
        if state.prWinCond>0.96 and state.prWin>0.7 and state.prWin_delta>0 and state.hand_score0>=4:
            state['play']  = 'allin'
            return [0,0,0.2,4*max(state.minBet,2*state.smallBlind)]
        elif state.prWinCond>0.88 and state.prWin>0.6 and state.prWin_delta>0 and state.hand_score0>=4:
            state['play']  = 'raise/allin' # Call Any
            return [0,0,0.5,4*max(state.minBet,2*state.smallBlind)]
        elif state.prWinCond>0.76 and state.prWin>0.5 and state.prWin_delta>0 and state.hand_score0>=3:
            state['play']  = 'raise/allin' # Call Any
            self_minBet    = 4*max(state.minBet,2*state.smallBlind)
            return [0,1,0,0] if self_minBet>state.chips/7 else [0,0,0.8,self_minBet]
        elif state.prWinCond>0.57 and state.prWin>0.4 and state.prWin_delta>0 and state.hand_score0>=3:
            state['play']  = 'call/raise' # Call Any
            self_minBet    = 2*max(state.minBet,2*state.smallBlind)
            return [0,1,0,0] if self_minBet>state.chips/7 else [0,0,0.8,self_minBet]
        elif state.prWinCond>max(state.thd_call,0.1) and state.prWin_delta>-0.2:
            if state.prWinCond>0.4 or state.prWin_delta>0.05 or state.hand_score0>=4:
                self_minBet  = 2*max(state.minBet,2*state.smallBlind)
                if state.minBet > state.chips/7:
                    state['play']  = 'fold/allin'
                    return [0.8,0,0,0]
                elif self_minBet > state.chips/7:
                    state['play']  = 'call/allin'
                    return [0,0.8,0,0]
                else:
                    state['play']  = 'call/raise'
                    return [0,0.2,0.8,self_minBet]
            elif state.minBet > state.chips/7:
                state['play']  = 'fold'
                return [1,0,0,0]
            else:
                state['play']  = 'check/call'
                return [0,1,0,0]
        else:
            state['play']  = 'fold/check'
            return [0,1,0,0] if state.cost_to_call<=0 else [1,0,0,0]
