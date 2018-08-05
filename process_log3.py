from common import *
import json

pd.set_option('display.max_rows',120)
pd.set_option('display.max_columns',None)
pd.set_option('display.width',85)

#---------------#
#-- Read Data --#
#---------------#
dt      = sys.argv[1] #'20180716'
rnd     = pd.read_csv('data/round_log_'+dt+'.gz')
action  = pd.read_csv('data/action_proc_'+dt+'.gz')

#-- Get Opponent playerName --#
t0  = time.clock()
action['opponents']  = action.opponents.apply(eval)
action['op_playerName']  = action.opponents.apply(lambda x:[y['playerName'] for y in x])
print(time.clock() - t0)

#-- Round Hand Score Lookup Table --#
rnd.dropna(subset=['round_id','cards'],inplace=True)
rnd.drop_duplicates(subset=['round_id','playerName','cards','board'],inplace=True)
rnd.set_index(['round_id','playerName'],inplace=True)
score_lut  = pd.concat([rnd.score_Deal,rnd.score_Flop,rnd.score_Turn,rnd.score_River],0,keys=('Deal','Flop','Turn','River',)).apply(eval)

t0  = time.clock()
action['win']  = action.apply(lambda x:score_lut.loc[x.roundName,x.round_id,x.playerName]>=max([score_lut.loc[(x.roundName,x.round_id,y)] for y in x.op_playerName]) if len(x.op_playerName) else True,axis=1)
action['winRiver']  = action.apply(lambda x:score_lut.loc['River',x.round_id,x.playerName]>=max([score_lut.loc[('River',x.round_id,y)] for y in x.op_playerName]) if len(x.op_playerName) else True,axis=1)
print(time.clock() - t0)

#-- Hand Texture --#
t0  = time.clock()
mask  = action.roundName == 'Deal'
action['hand']  = action.cards + ' ' + action.board.fillna('')
hand_tt         = pd.DataFrame(action.loc[~mask,'hand'].unique(),columns=('hand',))
temp     = hand_tt.hand.apply(hand_texture)
hand_tt  = pd.concat([hand_tt,temp],1).set_index('hand')
action   = action.merge(hand_tt,how='left',left_on='hand',right_index=True,copy=False)
action.drop('hand','columns',inplace=True)
print(time.clock() - t0)

action.loc[mask,'hand_suit']      = action.loc[mask,'cards_suit'] + 1
action.loc[mask,'hand_suit_rank'] = action.loc[mask,'cards_rank1']
action.loc[mask,'hand_conn']      = action.loc[mask,'cards_conn'] + 1
action.loc[mask,'hand_conn_rank'] = action.loc[mask,'cards_rank1']

#-- Output --#
action  = action[['timestamp',
    'tableNumber','roundCount','game_id','round_id','smallBlind','roundName',
    'raiseCount','betCount','totalBet',
    'playerName','isHuman','isOnline','chips','reloadCount','position',
    'cards','board','pot','bet',
    'N','Nnf','Nallin',
    'pot_sum','bet_sum','maxBet','NMaxBet',
    'opponents',
    'action','amount','rank','message','win','winRiver','winMoney','chips_final',
    'blind_level','game_N','game_phase','pos',
    'op_raiser','Nfold','Ncall','Nraise','self_Ncall','self_Nraise',
    'prev_action','NRfold','NRcall','NRraise','op_resp',
    'cards_rank1','cards_rank2','cards_rank_sum','cards_aces','cards_faces',
    'cards_pair','cards_suit','cards_conn','cards_conn2','cards_category',
    'hand_score','hand_suit','hand_suit_rank','hand_conn','hand_conn_rank',
    'board_rank1','board_rank2','board_aces','board_faces',
    'board_kind','board_kind_rank','board_suit','board_suit_rank',
    'board_conn','board_conn_rank',
    'Nsim','prWin','prWinStd','prWin_delta',
    ]]
action.to_csv('action_proc_'+dt+'.gz',index=False,compression='gzip')
