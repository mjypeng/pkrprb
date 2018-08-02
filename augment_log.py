from common import *

pd.set_option('display.max_rows',120)
pd.set_option('display.max_columns',None)
pd.set_option('display.width',90)

#-- Read Data --#
dt      = sys.argv[1]
action  = pd.read_csv('data/action_proc_'+dt+'.gz')

#-- Hand Win Probability @River --#
w  = pd.read_csv('precal_win_prob.gz',index_col='hashkey')
action['hashkey']  = action[['N','cards','board']].fillna('').apply(lambda x:pkr_to_hash(x.N,x.cards,x.board),axis=1)
action  = action.merge(w,how='left',left_on='hashkey',right_index=True,copy=False).drop('hashkey','columns')

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

#-- Validation Check --#
if (action.Nsim>0).all():  print(dt + ' Nsim OK')
if action.columns.tolist() == ['timestamp','tableNumber','roundCount','game_id','round_id','smallBlind','roundName','raiseCount','betCount','totalBet','playerName','isHuman','isOnline','chips','reloadCount','position','cards','board','pot','bet','N','Nnf','Nallin','pot_sum','bet_sum','maxBet','NMaxBet','opponents','action','amount','rank','message','winMoney','chips_final','blind_level','game_N','game_phase','pos','op_raiser','Nfold','Ncall','Nraise','self_Ncall','self_Nraise','prev_action','NRfold','NRcall','NRraise','op_resp','cards_rank1','cards_rank2','cards_rank_sum','cards_aces','cards_faces','cards_pair','cards_suit','cards_conn','cards_conn2','cards_category','hand_score','board_rank1','board_rank2','board_aces','board_faces','board_kind','board_kind_rank','board_suit','board_suit_rank','board_conn','board_conn_rank','Nsim','prWin','prWinStd','prWin_delta',]:
    print(dt + ' Columns OK')

action.to_csv('action_proc_'+dt+'.gz',index=False,compression='gzip')

# Final columns: [
#     'timestamp','tableNumber','roundCount',
#     'game_id','round_id','smallBlind','roundName',
#     'raiseCount','betCount','totalBet',
#     'playerName','isHuman','isOnline','chips','reloadCount','position',
#     'cards','board','pot','bet',
#     'N','Nnf','Nallin','pot_sum','bet_sum','maxBet','NMaxBet','opponents',
#     'action','amount','rank','message','winMoney','chips_final',
#     'blind_level','game_N','game_phase','pos',
#     'op_raiser','Nfold','Ncall','Nraise','self_Ncall','self_Nraise',
#     'prev_action','NRfold','NRcall','NRraise','op_resp',
#     'cards_rank1','cards_rank2','cards_rank_sum','cards_aces','cards_faces',
#     'cards_pair','cards_suit','cards_conn','cards_conn2','cards_category',
#     'hand_score',
#     'board_rank1','board_rank2','board_aces','board_faces',
#     'board_kind','board_kind_rank','board_suit','board_suit_rank',
#     'board_conn','board_conn_rank',
#     'Nsim','prWin','prWinStd','prWin_delta',
#     ]
