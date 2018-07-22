from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

pd.set_option('display.max_rows',120)
pd.set_option('display.max_columns',None)
pd.set_option('display.width',90)

dt      = '20180716'
game    = pd.read_csv('data/game_log_'+dt+'.gz')
rnd     = pd.read_csv('data/round_log_'+dt+'.gz')
action  = pd.read_csv('data/action_log_'+dt+'.gz')

game.dropna(subset=['game_id'],inplace=True)
game.sort_values(['tableNumber','game_id'],inplace=True)
game['survive']  = game.chips>0
game_ranking = game.groupby('playerName')[['survive','chips']].agg(['count','mean']).iloc[:,[0,1,3]].sort_values([('chips','mean')],ascending=False)

rnd.dropna(subset=['game_id','round_id'],inplace=True)
rnd.sort_values(['tableNumber','game_id','roundCount'],inplace=True)
rnd['win']    = rnd.winMoney>0
round_ranking = rnd.groupby('playerName')[['win','winMoney']].agg(['count','mean']).iloc[:,[0,1,3]].sort_values(('winMoney','mean'),ascending=False)


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
players       = None
prev_action   = 'none'
for idx,row in target_action.iterrows():
    if pd.isnull(row.round_id): continue
    if cur_round_id is None or cur_round_id != row.round_id:
        players  = pd.DataFrame(columns=('prev_action','Nfold','Ncall','Nraise','NRfold','NRcall','NRraise'))
        prev_action  = 'none'
    if cur_roundName is None or cur_roundName != row.roundName:
        players['NRfold']   = 0
        players['NRcall']   = 0
        players['NRraise']  = 0
    #
    target_action.loc[idx,'prev_action']  = prev_action
    target_action.loc[idx,'Nfold']    = players.Nfold.sum()
    target_action.loc[idx,'Ncall']    = players.Ncall.sum()
    target_action.loc[idx,'Nraise']   = players.Nraise.sum()
    target_action.loc[idx,'NRfold']   = players.NRfold.sum()
    target_action.loc[idx,'NRcall']   = players.NRcall.sum()
    target_action.loc[idx,'NRraise']  = players.NRraise.sum()
    target_action.loc[idx,'self_prev_action']  = players.loc[row.playerName,'prev_action'] if row.playerName in players.index else 'none'
    target_action.loc[idx,'self_Nfold']    = players.loc[row.playerName,'Nfold'] if row.playerName in players.index else 0
    target_action.loc[idx,'self_Ncall']    = players.loc[row.playerName,'Ncall'] if row.playerName in players.index else 0
    target_action.loc[idx,'self_Nraise']   = players.loc[row.playerName,'Nraise'] if row.playerName in players.index else 0
    target_action.loc[idx,'self_NRfold']   = players.loc[row.playerName,'NRfold'] if row.playerName in players.index else 0
    target_action.loc[idx,'self_NRcall']   = players.loc[row.playerName,'NRcall'] if row.playerName in players.index else 0
    target_action.loc[idx,'self_NRraise']  = players.loc[row.playerName,'NRraise'] if row.playerName in players.index else 0
    #
    if row.playerName not in players.index:
        players.loc[row.playerName]  = 0
    players.loc[row.playerName,'prev_action']  = row.action
    players.loc[row.playerName,'Nfold']       += row.action=='fold'
    players.loc[row.playerName,'Ncall']       += row.action=='check/call'
    players.loc[row.playerName,'Nraise']      += row.action=='bet/raise/allin'
    players.loc[row.playerName,'NRfold']      += row.action=='fold'
    players.loc[row.playerName,'NRcall']      += row.action=='check/call'
    players.loc[row.playerName,'NRraise']     += row.action=='bet/raise/allin'
    prev_action  = row.action

target_action[['NRfold','NRcheck','NRcall','NRbet','NRraise','NRallincall','NRallin']] = target_action[['NRfold','NRcheck','NRcall','NRbet','NRraise','NRallincall','NRallin']].astype(int)
print(time.clock() - t0)

mask  = target_action.playerName.isin(target_players) & (target_action.roundName=='Deal')
target_action[mask].action.value_counts()

temp  = target_action[mask].board.str.split().apply(lambda x:pd.Series(score_hand5(pkr_to_cards(x))[:2]))
target_action.loc[mask,'score0']  = temp[0]
target_action.loc[mask,'score1']  = temp[1]

X  = target_action.loc[mask,['smallBlind','chips','position','pot','bet','N','Nnf','Nallin','pot_sum','bet_sum','maxBet','NMaxBet','NRfold','NRcheck','NRcall','NRbet','NRraise','NRallincall','NRallin']].copy() #,'score0','score1'
y  = target_action.loc[mask,['action','amount']].copy()

y.loc[(y.action=='call')&(y.amount==0),'action']  = 'check'
y.loc[(y.action=='allin')&(X.bet+y.amount<X.maxBet),'action']  = 'call'

y.loc[y.action.isin(('check','call')),'action']        = 'check/call'
y.loc[y.action.isin(('bet','raise','allin')),'action'] = 'bet/raise/allin'

P  = X.pot_sum + X.bet_sum
X['chips_SB']  = X.chips / X.smallBlind
X['chips_P']   = X.chips / P
X  = pd.concat([X,pd.get_dummies(X.position,prefix='pos',prefix_sep='=')],1)
X['pot_P']     = X.pot / P
X['pot_SB']    = X.pot / X.smallBlind
X['bet_P']     = X.bet / P
X['bet_SB']    = X.bet / X.smallBlind
X['bet_sum_P'] = X.bet_sum / P
X['bet_sum_SB'] = X.bet_sum / X.smallBlind
X['minBet_P']  = (X.maxBet - X.bet) / P
X['minBet_SB'] = (X.maxBet - X.bet) / X.smallBlind
X.drop(['smallBlind','chips','position','pot','bet','pot_sum','bet_sum','maxBet',],'columns',inplace=True)

rf  = RandomForestClassifier(n_estimators=20,max_depth=None,min_samples_leaf=1,oob_score=False,n_jobs=1,random_state=0,verbose=2,warm_start=False,class_weight=None)
rf.fit(X,y.action)
yhat  = rf.predict(X)
accuracy_score(y.action,yhat)

kf  = StratifiedKFold(n_splits=4,shuffle=True,random_state=0)
rf  = RandomForestClassifier(n_estimators=100,max_depth=None,min_samples_leaf=1,oob_score=False,n_jobs=1,random_state=0,verbose=2,warm_start=False,class_weight=None)
results  = pd.DataFrame(columns=('tr','tt'))
for i,(idx_tr,idx_tt) in enumerate(kf.split(X,y.action)):
    X_tr  = X.iloc[idx_tr]
    y_tr  = y.iloc[idx_tr]
    X_tt  = X.iloc[idx_tt]
    y_tt  = y.iloc[idx_tt]
    #
    rf.fit(X_tr,y_tr.action)
    yhat_tr  = rf.predict(X_tr)
    yhat_tt  = rf.predict(X_tt)
    #
    results.loc[i,'tr']  = accuracy_score(y_tr.action,yhat_tr)
    results.loc[i,'tt']  = accuracy_score(y_tt.action,yhat_tt)

print(results.mean(0))
accuracy_score(y_tt.action,yhat_tt,labels=['fold','check/call','bet/raise/allin'])
f1_score(y_tt.action,yhat_tt,labels=['fold','check/call','bet/raise/allin'],average=None)


# X:
# 'smallBlind','chips','position','board','pot','bet','N','Nnf','Nallin','pot_sum','bet_sum','maxBet','NMaxBet','NRfold','NRcheck','NRcall','NRbet','NRraise','NRallincall','NRallin','score0','score1'
# y:
# ,'action','amount'
