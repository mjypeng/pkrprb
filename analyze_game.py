from agent_common import *
import os

batch_name = sys.argv[1] #'basic_vs_human' #'basic_1+twice'
game_id    = sys.argv[2]

actions = pd.read_csv('game_records' + os.sep + batch_name + os.sep + 'game_' + game_id + '_actions.csv')
rounds  = pd.read_csv('game_records' + os.sep + batch_name + os.sep + 'game_' + game_id + '_rounds.csv')

actions = actions.merge(rounds[['round_id','playerName','cards','hand','rank','message','winMoney']],how='left',on=['round_id','playerName'],copy=False)
actions['game_id'] = pd.to_datetime(actions.game_id.astype(str))
actions['chips0'] = 0
actions['pot'] = 0
actions['bet'] = 0
actions['cost_to_call'] = 0
actions['Nsim']     = 0
actions['prWin']    = 0
actions['prWinStd'] = 0
actions.amount.fillna(0,inplace=True)
actions = actions[['game_id','round_id','turn_id','roundName','playerName','chips','chips0','reloadCount','cards','hand','Nsim','prWin','prWinStd','pot','bet','cost_to_call','position','action','amount','rank','message','winMoney']]
actions.rename(columns={'cards':'hole','hand':'board'},inplace=True)
actions['board'] = actions.board.str.split().str[2:].str.join(' ')
actions.loc[actions.roundName=='Deal','board'] = ''
actions.loc[actions.roundName=='Flop','board'] = actions.loc[actions.roundName=='Flop','board'].str.split().str[:3].str.join(' ')
actions.loc[actions.roundName=='Turn','board'] = actions.loc[actions.roundName=='Turn','board'].str.split().str[:4].str.join(' ')

# smallBlind starts from 10 and doubles every N(=number of surviving players) rounds
actions['chips'] += actions.amount
winners    = rounds.groupby(['round_id','playerName']).winMoney.mean().astype(int)
players    = pd.DataFrame(0,columns=('smallBlind','chips','reld','pot','bet'),index=actions.playerName.drop_duplicates())
players['chips']      = 1000
players['smallBlind'] = 10
round_id   = 0
roundName  = ''
smallBlind = 10
SB_player = actions[actions.position=='SB'][['round_id','playerName']].drop_duplicates().set_index('round_id')
BB_player = actions[actions.position=='BB'][['round_id','playerName']].drop_duplicates().set_index('round_id')
for idx,row in actions.iterrows():
    #
    N     = len(players) - ((players.chips==0)&(players.reld==2)&(players.pot==0)&(players.bet==0)).sum()
    hole  = pd.DataFrame([((x[0],ordermap[x[1:]]),x[0],ordermap[x[1:]]) for x in row.hole.split()],columns=('c','s','o'))
    board = pd.DataFrame([((x[0],ordermap[x[1:]]),x[0],ordermap[x[1:]]) for x in row.board.split()],columns=('c','s','o'))
    if len(board) > 0:
        t0 = time.time()
        calculate_win_prob_mp_start(N,hole,board,n_jobs=2)
    else:
        actions.loc[idx,'Nsim'] = 20000
        actions.loc[idx,'prWin'],actions.loc[idx,'prWinStd'] = read_win_prob(N,hole)
    #
    # Update player state
    players.loc[row.playerName,'chips'] = row.chips
    players.loc[row.playerName,'reld']  = row.reloadCount
    #
    if row.round_id != round_id:
        # new round
        if round_id > 0:
            players['chips'] += winners.loc[round_id]
        #
        round_id  = row.round_id
        players['pot'] = 0
        players['bet'] = 0
        #
        smallBlind = players.loc[SB_player.loc[round_id],'smallBlind'][0]
    if row.roundName != roundName:
        roundName = row.roundName
        players['pot'] += players.bet
        players['bet']  = 0
    # if row.reloadCount > players.loc[row.playerName,'reld']:
    #     players.loc[row.playerName,'chips'] += 1000*(row.reloadCount - players.loc[row.playerName,'reld'])
    #     players.loc[row.playerName,'reld']   = row.reloadCount
    #
    if roundName == 'Deal':
        if row.position == 'SB' and players.loc[row.playerName,'bet'] == 0:
            players.loc[row.playerName,'bet'] = min(smallBlind,players.loc[row.playerName,'chips'])
            # players.loc[row.playerName,'chips'] -= players.loc[row.playerName,'bet']
            players.loc[row.playerName,'smallBlind'] *= 2
        elif row.position == 'BB' and players.loc[row.playerName,'bet'] == 0:
            players.loc[row.playerName,'bet'] = min(2*smallBlind,players.loc[row.playerName,'chips'])
            # players.loc[row.playerName,'chips'] -= players.loc[row.playerName,'bet']
    #
    actions.loc[idx,'chips0'] = players.loc[row.playerName,'chips']
    actions.loc[idx,'pot']    = players.loc[row.playerName,'pot']
    actions.loc[idx,'bet']    = players.loc[row.playerName,'bet']
    actions.loc[idx,'cost_to_call'] = players.bet.max() - players.loc[row.playerName,'bet']
    #
    players.loc[row.playerName,'bet']   += row.amount
    players.loc[row.playerName,'chips'] -= row.amount
    #
    if len(board) > 0:
        res  = calculate_win_prob_mp_get()
        while len(res) < 1000:
            time.sleep(2)
            res = calculate_win_prob_mp_get()
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

actions.to_csv("training_%s.csv"%game_id,index=False,encoding='utf-8-sig')
