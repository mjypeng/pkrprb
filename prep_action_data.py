from agent_common import *
import os

batch_name = sys.argv[1]
game_id    = sys.argv[2]

actions = pd.read_csv('game_records' + os.sep + batch_name + os.sep + 'game_' + game_id + '_actions.csv')
rounds  = pd.read_csv('game_records' + os.sep + batch_name + os.sep + 'game_' + game_id + '_rounds.csv')

#-- Preprocess Rounds Data --#
# Assign positions
for i in rounds.round_id.unique():
    rounds_i = rounds[(rounds.round_id==i)&((rounds.chips>0)|(rounds.reloadCount<2))]
    idx_list = rounds_i.index
    SB_idx  = (rounds_i.position=='SB').idxmax()
    BB_idx  = (rounds_i.position=='BB').idxmax()
    D_idx   = SB_idx - 1 if SB_idx > idx_list.min() else idx_list.max()
    while D_idx not in idx_list and D_idx != BB_idx:
        D_idx = D_idx - 1 if D_idx > idx_list.min() else idx_list.max()
    if D_idx != BB_idx:
        rounds.loc[D_idx,'position']  = 'D'
    idx  = BB_idx + 1 if BB_idx < idx_list.max() else idx_list.min()
    pos  = 1
    while idx not in (SB_idx,BB_idx,D_idx):
        if idx in idx_list:
            rounds.loc[idx,'position'] = pos
            pos += 1
        idx  = idx + 1 if idx < idx_list.max() else idx_list.min()
# Number of players
rounds['N'] = rounds[rounds.position.notnull()].groupby('round_id').playerName.nunique().loc[rounds.round_id].values
returns  = rounds.groupby(['round_id','playerName'])[['winMoney']].max()

#-- Compile Training Data --#
# Prepare training data columns
actions.drop('position','columns',inplace=True)
actions = actions.merge(rounds[['round_id','N','playerName','position','cards','hand','allIn','folded','rank','message','winMoney']],how='left',on=['round_id','playerName'],copy=False)
actions['game_id'] = pd.to_datetime(actions.game_id.astype(str))
actions['Nnf']     = 0 # Number of players not folded
actions['pot_sum'] = 0 # Sum of the whole pot for previous rounds
actions['bet_sum'] = 0 # Sum of all bets on current round
actions['pot'] = 0 # Contribution to the pot from current player for previous rounds
actions['bet'] = 0 # Bet on current round from current player
actions['cost_to_call'] = 0 # Minimum bet to match to stay in game
actions['NMaxBet']  = 0
actions['Nsim']     = 0
actions['prWin']    = 0
actions['prWinStd'] = 0
actions.amount.fillna(0,inplace=True)
actions.rename(columns={'cards':'hole','hand':'board'},inplace=True)
actions['board'] = actions.board.str.split().str[2:].str.join(' ')
actions.loc[actions.roundName=='Deal','board'] = ''
actions.loc[actions.roundName=='Flop','board'] = actions.loc[actions.roundName=='Flop','board'].str.split().str[:3].str.join(' ')
actions.loc[actions.roundName=='Turn','board'] = actions.loc[actions.roundName=='Turn','board'].str.split().str[:4].str.join(' ')
actions = actions[['game_id','round_id','turn_id','roundName','playerName','chips','reloadCount','position','N','Nnf','hole','board','Nsim','prWin','prWinStd','pot_sum','bet_sum','pot','bet','cost_to_call','NMaxBet','action','amount','allIn','folded','rank','message','winMoney']]
actions['chips'] += actions.amount # Modify chips to reflect chips *before* action

# Simulate Play and fill necessary columns
players    = pd.DataFrame(0,columns=('SB','chips','reld','pot','bet','folded','isSurvive'),index=actions.playerName.drop_duplicates())
players['SB']  = 10 # smallBlind starts from 10 and doubles every N(=number of surviving players) rounds
round_id   = 0
roundName  = ''
smallBlind = 10
SB_player  = rounds[rounds.position=='SB'][['round_id','playerName']].drop_duplicates().set_index('round_id').playerName
BB_player  = rounds[rounds.position=='BB'][['round_id','playerName']].drop_duplicates().set_index('round_id').playerName
for idx,row in actions.iterrows():
    #
    hole  = pd.DataFrame([((x[0],ordermap[x[1:]]),x[0],ordermap[x[1:]]) for x in row.hole.split()],columns=('c','s','o'))
    board = pd.DataFrame([((x[0],ordermap[x[1:]]),x[0],ordermap[x[1:]]) for x in row.board.split()],columns=('c','s','o'))
    if len(board) > 0:
        t0 = time.time()
        calculate_win_prob_mp_start(row.N,hole,board,n_jobs=2)
    else:
        actions.loc[idx,'Nsim'],actions.loc[idx,'prWin'],actions.loc[idx,'prWinStd'] = read_win_prob(row.N,hole)
    #
    # Update player state
    players.loc[row.playerName,'chips'] = row.chips
    players.loc[row.playerName,'reld']  = row.reloadCount
    #
    if row.round_id != round_id:
        if round_id > 0:
            # Record cost and returns
            returns.loc[round_id,'cost']   = pd.concat([players.pot + players.bet],keys=(round_id,))
            returns.loc[round_id,'profit'] = returns.loc[[round_id],'winMoney'] - returns.loc[[round_id],'cost']
        # new round
        round_id  = row.round_id
        roundName = row.roundName
        players['pot'] = 0
        players['bet'] = 0
        players['folded']    = False
        players['isSurvive'] = rounds[rounds.round_id==round_id].set_index('playerName').position.notnull()
        #
        smallBlind  = players.loc[SB_player[round_id],'SB']
        players.loc[SB_player[round_id],'SB'] *= 2
        players.loc[SB_player[round_id],'bet'] = smallBlind
        players.loc[BB_player[round_id],'bet'] = 2*smallBlind
    elif row.roundName != roundName:
        roundName = row.roundName
        players['pot'] += players.bet
        players['bet']  = 0
    #
    actions.loc[idx,'Nnf']     = row.N - players.folded.sum()
    actions.loc[idx,'pot_sum'] = players.pot.sum()
    actions.loc[idx,'bet_sum'] = players.bet.sum()
    actions.loc[idx,'pot']     = players.loc[row.playerName,'pot']
    actions.loc[idx,'bet']     = players.loc[row.playerName,'bet']
    actions.loc[idx,'cost_to_call'] = players.bet.max() - players.loc[row.playerName,'bet']
    actions.loc[idx,'NMaxBet'] = (players.bet==players.bet.max()).sum()
    #
    # Update game state for next turn
    players.loc[row.playerName,'chips'] -= row.amount
    players.loc[row.playerName,'bet']   += row.amount
    players.loc[row.playerName,'folded'] = row.action=='fold'
    #
    if len(board) > 0:
        res  = calculate_win_prob_mp_get()
        # while len(res) < 100:
        #     time.sleep(2)
        #     res = calculate_win_prob_mp_get()
        calculate_win_prob_mp_stop()
        res  = [x['prWin'] for x in res]
        actions.loc[idx,'Nsim']     = len(res)
        actions.loc[idx,'prWin']    = np.mean(res)
        actions.loc[idx,'prWinStd'] = np.std(res)
        print(time.time()-t0)
    #
    print(round_id,row.turn_id)
    print(players)
    print()

# Record cost and returns
returns.loc[round_id,'cost']   = pd.concat([players.pot + players.bet],keys=(round_id,))
returns.loc[round_id,'profit'] = returns.loc[[round_id],'winMoney'] - returns.loc[[round_id],'cost']
actions = actions.merge(returns[['cost','profit']],how='left',left_on=['round_id','playerName'],right_index=True,copy=False)

actions.to_csv("training_%s.csv"%game_id,index=False,encoding='utf-8-sig')
