from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

pd.set_option('display.max_rows',120)
pd.set_option('display.max_columns',None)
pd.set_option('display.width',120)

#-- Read Data --#
dt      = sys.argv[1] #'20180716'
game    = pd.read_csv('data/game_log_'+dt+'.gz')
rnd     = pd.read_csv('data/round_log_'+dt+'.gz')
action  = pd.read_csv('data/action_log_'+dt+'.gz')

#-- Get Player Rankings --#
game.dropna(subset=['game_id'],inplace=True)
game.sort_values(['tableNumber','game_id'],inplace=True)
game['survive']  = game.chips>0
game_ranking = game.groupby('playerName')[['survive','chips']].agg(['count','mean']).iloc[:,[0,1,3]].sort_values([('chips','mean')],ascending=False)

rnd.dropna(subset=['game_id','round_id'],inplace=True)
rnd.sort_values(['tableNumber','game_id','roundCount'],inplace=True)
rnd['win']    = rnd.winMoney>0
round_ranking = rnd.groupby('playerName')[['win','winMoney']].agg(['count','mean']).iloc[:,[0,1,3]].sort_values(('winMoney','mean'),ascending=False)

#-- Get Target Players --#
mask  = (round_ranking[('win','mean')]>0.33) | (round_ranking[('winMoney','mean')]>750)
target_players = round_ranking[mask].index.values
target_rounds  = rnd[rnd.playerName.isin(target_players)].round_id
target_action  = action[action.round_id.isin(target_rounds)].copy()
target_action.sort_values(['tableNumber','game_id','roundCount','timestamp'],inplace=True)

#-- Refine and Consolidate Action Types --#
target_action.loc[(target_action.action=='call')&(target_action.amount==0),'action']  = 'check'
target_action.loc[(target_action.action=='allin')&(target_action.bet+target_action.amount<target_action.maxBet),'action']  = 'call'
target_action.loc[target_action.action.isin(('check','call')),'action'] = 'check/call'
target_action.loc[target_action.action.isin(('bet','raise','allin')),'action'] = 'bet/raise/allin'

t0  = time.clock()
cur_round_id  = None
cur_roundName = None
for idx,row in target_action.iterrows():
    if pd.isnull(row.round_id): continue
    if cur_round_id is None or cur_round_id != row.round_id:
        players  = pd.DataFrame(columns=('prev_action','Nfold','Ncall','Nraise','NRfold','NRcall','NRraise'))
        cur_round_id = row.round_id
    if cur_roundName is None or cur_roundName != row.roundName:
        players['prev_action'] = 'none'
        players['Nfold']    = 0
        players['Ncall']    = 0
        players['Nraise']   = 0
        players['NRfold']   = 0
        players['NRcall']   = 0
        players['NRraise']  = 0
        cur_roundName       = row.roundName
    #
    target_action.loc[idx,'Nfold']    = players.Nfold.sum()
    target_action.loc[idx,'Ncall']    = players.Ncall.sum()
    target_action.loc[idx,'Nraise']   = players.Nraise.sum()
    if row.playerName in players.index:
        target_action.loc[idx,'self_Ncall']  = players.loc[row.playerName,'Ncall']
        target_action.loc[idx,'self_Nraise'] = players.loc[row.playerName,'Nraise']
        target_action.loc[idx,'prev_action'] = players.loc[row.playerName,'prev_action']
        target_action.loc[idx,'NRfold']  = players.loc[row.playerName,'NRfold']
        target_action.loc[idx,'NRcall']  = players.loc[row.playerName,'NRcall']
        target_action.loc[idx,'NRraise'] = players.loc[row.playerName,'NRraise']
    else:
        target_action.loc[idx,'self_Ncall']  = 0
        target_action.loc[idx,'self_Nraise'] = 0
        target_action.loc[idx,'prev_action'] = 'none'
        target_action.loc[idx,'NRfold']  = target_action.loc[idx,'Nfold']
        target_action.loc[idx,'NRcall']  = target_action.loc[idx,'Ncall']
        target_action.loc[idx,'NRraise'] = target_action.loc[idx,'Nraise']
    #
    if row.playerName in players.index:
        players.loc[row.playerName,'NRfold']  = 0
        players.loc[row.playerName,'NRcall']  = 0
        players.loc[row.playerName,'NRraise'] = 0
    else:
        players.loc[row.playerName]  = 0
    #
    players.loc[row.playerName,'prev_action']  = row.action
    players.loc[row.playerName,'Nfold']       += row.action=='fold'
    players.loc[row.playerName,'Ncall']       += row.action=='check/call'
    players.loc[row.playerName,'Nraise']      += row.action=='bet/raise/allin'
    #
    players.loc[players.index!=row.playerName,'NRfold']  += row.action=='fold'
    players.loc[players.index!=row.playerName,'NRcall']  += row.action=='check/call'
    players.loc[players.index!=row.playerName,'NRraise'] += row.action=='bet/raise/allin'

int_cols  = [x+y+z for x in ('','self_') for y in ('N','NR') for z in ('fold','call','raise') if x+y+z in target_action]
target_action[int_cols] = target_action[int_cols].astype(int)
print(time.clock() - t0)

target_action.to_csv('target_action_'+dt+'.gz',index=False,compression='gzip')
