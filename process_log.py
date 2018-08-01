from common import *
import json

pd.set_option('display.max_rows',120)
pd.set_option('display.max_columns',None)
pd.set_option('display.width',120)

#---------------#
#-- Read Data --#
#---------------#
dt      = sys.argv[1] #'20180716'
game    = pd.read_csv('data/game_log_'+dt+'.gz')
rnd     = pd.read_csv('data/round_log_'+dt+'.gz')
action  = pd.read_csv('data/action_log_'+dt+'.gz')

#-------------------------#
#-- Get Player Rankings --#
#-------------------------#
game.dropna(subset=['game_id'],inplace=True)
game.sort_values(['tableNumber','game_id'],inplace=True)
game['survive']  = game.chips>0
game_ranking = game.groupby('playerName')[['survive','chips']].agg(['count','mean']).iloc[:,[0,1,3]].sort_values([('chips','mean')],ascending=False)

rnd.dropna(subset=['game_id','round_id'],inplace=True)
rnd.sort_values(['tableNumber','game_id','roundCount'],inplace=True)
rnd['win']    = rnd.winMoney>0
round_ranking = rnd.groupby('playerName')[['win','winMoney']].agg(['count','mean']).iloc[:,[0,1,3]].sort_values(('winMoney','mean'),ascending=False)

#----------------------------------------------------#
#-- Get Round Action Data involving Target Players --#
#----------------------------------------------------#
mask  = (round_ranking[('win','mean')]>0.33) | (round_ranking[('winMoney','mean')]>750)
target_players = round_ranking[mask].index.values
target_rounds  = rnd[rnd.playerName.isin(target_players)].round_id
action  = action[action.round_id.isin(target_rounds)]
action.sort_values(['tableNumber','game_id','roundCount','timestamp'],inplace=True)

#-----------------------------------------#
#-- Refine and Consolidate Action Types --#
#-----------------------------------------#
action.loc[(action.action=='call')&(action.amount==0),'action']  = 'check'
action.loc[(action.action=='allin')&(action.bet+action.amount<action.maxBet),'action']  = 'call'
action.loc[action.action.isin(('check','call')),'action'] = 'check/call'
action.loc[action.action.isin(('bet','raise','allin')),'action'] = 'bet/raise/allin'

#----------------#
#-- Game Phase --#
#----------------#
action['blind_level'] = (np.log2(action.smallBlind/10) + 1).astype(int)
game_N  = rnd.groupby('round_id').playerName.nunique()
action['game_N']  = game_N.loc[action.round_id].values
action['game_phase']  = np.where(action.blind_level<4,'Early','Middle')
action.loc[action.N<=np.ceil((action.game_N+1)/2),'game_phase'] = 'Late'

#--------------------#
#-- Table Position --#
#--------------------#
action['pos']     = position_feature_batch(action.N,action.position)

#-------------------------------------------#
#-- Re-trace and Append Opponent Response --#
#-------------------------------------------#
t0  = time.clock()
cur_round_id  = None
cur_roundName = None
for idx,row in action.iterrows():
    if pd.isnull(row.round_id): continue
    if cur_round_id is None or cur_round_id != row.round_id:
        players  = pd.DataFrame(columns=('prev_action','Nfold','Ncall','Nraise','NRfold','NRcall','NRraise'))
        raiser   = None
        cur_round_id = row.round_id
    if cur_roundName is None or cur_roundName != row.roundName:
        players['prev_action'] = 'none'
        players['Nfold']    = 0
        players['Ncall']    = 0
        players['Nraise']   = 0
        players['NRfold']   = 0
        players['NRcall']   = 0
        players['NRraise']  = 0
        raiser              = None
        cur_roundName       = row.roundName
    #
    action.loc[idx,'op_raiser'] = raiser
    action.loc[idx,'Nfold']    = players.Nfold.sum()
    action.loc[idx,'Ncall']    = players.Ncall.sum()
    action.loc[idx,'Nraise']   = players.Nraise.sum()
    if row.playerName in players.index:
        action.loc[idx,'self_Ncall']  = players.loc[row.playerName,'Ncall']
        action.loc[idx,'self_Nraise'] = players.loc[row.playerName,'Nraise']
        action.loc[idx,'prev_action'] = players.loc[row.playerName,'prev_action']
        action.loc[idx,'NRfold']  = players.loc[row.playerName,'NRfold']
        action.loc[idx,'NRcall']  = players.loc[row.playerName,'NRcall']
        action.loc[idx,'NRraise'] = players.loc[row.playerName,'NRraise']
    else:
        action.loc[idx,'self_Ncall']  = 0
        action.loc[idx,'self_Nraise'] = 0
        action.loc[idx,'prev_action'] = 'none'
        action.loc[idx,'NRfold']  = action.loc[idx,'Nfold']
        action.loc[idx,'NRcall']  = action.loc[idx,'Ncall']
        action.loc[idx,'NRraise'] = action.loc[idx,'Nraise']
    #
    if row.playerName in players.index:
        players.loc[row.playerName,'NRfold']  = 0
        players.loc[row.playerName,'NRcall']  = 0
        players.loc[row.playerName,'NRraise'] = 0
    else:
        players.loc[row.playerName]  = 0
    #
    if row.action=='bet/raise/allin': raiser = row.playerName
    players.loc[row.playerName,'prev_action']  = row.action
    players.loc[row.playerName,'Nfold']       += row.action=='fold'
    players.loc[row.playerName,'Ncall']       += row.action=='check/call'
    players.loc[row.playerName,'Nraise']      += row.action=='bet/raise/allin'
    #
    players.loc[players.index!=row.playerName,'NRfold']  += row.action=='fold'
    players.loc[players.index!=row.playerName,'NRcall']  += row.action=='check/call'
    players.loc[players.index!=row.playerName,'NRraise'] += row.action=='bet/raise/allin'

int_cols  = [x+y+z for x in ('','self_') for y in ('N','NR') for z in ('fold','call','raise') if x+y+z in action]
action[int_cols] = action[int_cols].astype(int)
print(time.clock() - t0)

#-----------------------#
#-- Opponent Response --#
#-----------------------#
action['op_resp'] = opponent_response_code_batch(action)

#------------------------#
#-- Hole Cards Texture --#
#------------------------#
action  = pd.concat([action,hole_texture_batch(action.cards)],1)
action['cards_category']  = hole_texture_to_category_batch(action)

#------------------------#
#-- Player Hand Scores --#
#------------------------#
hand  = action.cards + ' ' + action.board.fillna('')

#-- Deal --#
mask  = action.roundName == 'Deal'
action.loc[mask,'hand_score']  = action[mask].apply(lambda x:json.dumps((int(x.cards_pair),x.cards_rank1,0 if x.cards_pair else x.cards_rank2,)).replace(' ',''),axis=1)

#-- Flop --#
t0  = time.clock()
mask  = action.roundName == 'Flop'
action.loc[mask,'hand_score']  = hand[mask].str.split().apply(lambda x:json.dumps(score_hand5(pkr_to_cards(x))).replace(' ',''))
print(time.clock() - t0)

#-- Turn/River --#
t0  = time.clock()
mask  = action.roundName.isin(('Turn','River'))
action.loc[mask,'hand_score']  = hand[mask].str.split().apply(lambda x:json.dumps(score_hand(pkr_to_cards(x))).replace(' ',''))
print(time.clock() - t0)

#-- Board Texture --#
t0  = time.clock()
board_tt  = action[['board']].dropna().drop_duplicates()
temp      = board_tt.board.apply(board_texture)
board_tt  = pd.concat([board_tt,temp],1).set_index('board')
print(time.clock() - t0)

action  = action.merge(board_tt,how='left',left_on='board',right_index=True,copy=False)
action[board_tt.columns] = action[board_tt.columns].fillna(0).astype(int)

#-- Output --#
action.to_csv('action_proc_'+dt+'.gz',index=False,compression='gzip')
