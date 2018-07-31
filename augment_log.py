from common import *

pd.set_option('display.max_rows',120)
pd.set_option('display.max_columns',None)
pd.set_option('display.width',90)

#-- Read Data --#
dt      = sys.argv[1]
action  = pd.read_csv('data/target_action_'+dt+'.gz')

#-- Hand Win Probability @River --#
w  = pd.read_csv('precal_win_prob.gz',index_col='hashkey')
action['hashkey']  = action[['N','cards','board']].fillna('').apply(lambda x:pkr_to_hash(x.N,x.cards,x.board),axis=1)
action  = action.merge(w,how='left',left_on='hashkey',right_index=True,copy=False)

#-- Append prWin_delta --#
t0  = time.clock()
cur_round_id  = None
for idx,row in action.iterrows():
    if cur_round_id is None or cur_round_id != row.round_id:
        players  = pd.DataFrame(columns=('roundName','prWin','prWin_prev',))
        cur_round_id = row.round_id
    #
    if row.playerName not in players.index:
        action.loc[idx,'prWin_delta']  = 0
        players.loc[row.playerName,'prWin_prev'] = None
        players.loc[row.playerName,'roundName']  = row.roundName
        players.loc[row.playerName,'prWin']      = row.prWin
    elif row.roundName == players.loc[row.playerName,'roundName']:
        action.loc[idx,'prWin_delta']  = row.prWin - players.loc[row.playerName,'prWin_prev'] if players.loc[row.playerName,'prWin_prev'] is not None else 0
        players.loc[row.playerName,'prWin']      = row.prWin
    else:
        action.loc[idx,'prWin_delta']  = row.prWin - players.loc[row.playerName,'prWin']
        players.loc[row.playerName,'prWin_prev'] = players.loc[row.playerName,'prWin']
        players.loc[row.playerName,'roundName']  = row.roundName
        players.loc[row.playerName,'prWin']      = row.prWin

print(time.clock() - t0)

action.to_csv('target_action_'+dt+'.gz',index=False,compression='gzip')
