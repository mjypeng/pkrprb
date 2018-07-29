from common import *

pd.set_option('display.max_rows',120)
pd.set_option('display.max_columns',None)
pd.set_option('display.width',90)

#-- Read Data --#
dt      = sys.argv[1] #'20180716'
rnd     = pd.read_csv('data/round_log_'+dt+'.gz')
action  = pd.read_csv('data/target_action_'+dt+'.gz')

#-- Game Phase --#
action['blind_level'] = (np.log2(action.smallBlind/10) + 1).astype(int)
game_N  = rnd.groupby('round_id').playerName.nunique()
action['game_N']  = game_N.loc[action.round_id].values
action['game_phase']  = np.where(action.blind_level<4,'Early','Middle')
action.loc[action.N<=np.ceil((action.game_N+1)/2),'game_phase'] = 'Late'

#-- Table Position --#
action['pos']     = position_feature_batch(action.N,action.position)

#-- Opponent Response --#
action['op_resp'] = opponent_response_code_batch(action)

#-- Hole Cards Texture --#
action  = pd.concat([action,hole_texture_batch(action.cards)],1)
action['cards_category']  = hole_texture_to_category_batch(action)

#-- Player Hand --#
action['hand']  = action.cards + ' ' + action.board.fillna('')

#-- Deal hand score --#
mask  = action.roundName == 'Deal'
action.loc[mask,'hand_score0']  = action.loc[mask,'cards_pair'].astype(int)
action.loc[mask,'hand_score1']  = action.loc[mask,'cards_rank1']
action.loc[mask,'hand_score2']  = np.where(action.loc[mask,'cards_pair'],0,action.loc[mask,'cards_rank2'])

#-- Flop hand score --#
t0  = time.clock()
mask  = action.roundName == 'Flop'
temp  = action[mask].hand.str.split().apply(lambda x:pd.Series(score_hand5(pkr_to_cards(x))))
action.loc[mask,'hand_score0']  = temp[0]
action.loc[mask,'hand_score1']  = temp[1]
action.loc[mask,'hand_score2']  = temp[2]
print(time.clock() - t0)

#-- Turn/River hand score --#
t0  = time.clock()
mask  = action.roundName.isin(('Turn','River'))
temp  = action[mask].hand.str.split().apply(lambda x:pd.Series(score_hand(pkr_to_cards(x))))
action.loc[mask,'hand_score0']  = temp[0]
action.loc[mask,'hand_score1']  = temp[1]
action.loc[mask,'hand_score2']  = temp[2]
print(time.clock() - t0)

action[['hand_score0','hand_score1','hand_score2']]  = action[['hand_score0','hand_score1','hand_score2']].fillna(0).astype(int)

#-- Board Texture --#
t0  = time.clock()
board_tt  = action[['board']].dropna().drop_duplicates()
temp      = board_tt.board.apply(board_texture)
board_tt  = pd.concat([board_tt,temp],1).set_index('board')
print(time.clock() - t0)

action  = action.merge(board_tt,how='left',left_on='board',right_index=True,copy=False)
action[board_tt.columns] = action[board_tt.columns].fillna(0).astype(int)

action.to_csv('target_action_'+dt+'.gz',index=False,compression='gzip')
