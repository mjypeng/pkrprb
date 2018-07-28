from common import *
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.externals import joblib

pd.set_option('display.max_rows',120)
pd.set_option('display.max_columns',None)
pd.set_option('display.width',90)

#-- Read Data --#
# dt      = sys.argv[1] #'20180716'
# game    = pd.concat([pd.read_csv(f) for f in glob.glob('data/game_log_*.gz')],0)
rnd     = pd.concat([pd.read_csv(f) for f in glob.glob('data/round_log_*.gz')],0)
action  = pd.concat([pd.read_csv(f) for f in glob.glob('data/target_action_*.gz')],0)

#-- Game Phase --#
action['blind_level'] = np.log2(action.smallBlind/10) + 1
game_N  = rnd.groupby('round_id').playerName.nunique()
action['game_N']  = game_N.loc[action.round_id].values
action['game_phase']  = np.where(action.blind_level<4,'Early','Middle')
action.loc[action.N<=np.ceil((action.game_N+1)/2),'game_phase'] = 'Late'

#-- Hole Cards Texture --#
action  = pd.concat([action,hole_texture_batch(action.cards)],1)
action['cards_category']  = hole_texture_to_category_batch(action)

#-- Table Position --#
action['pos']     = position_feature_batch(action.N,action.position)

#-- Opponent Response --#
action['op_resp'] = opponent_response_code_batch(action)

#-- Hand Win Probability @River --#
w  = pd.read_csv('precal_win_prob.gz',index_col='hashkey')
action['hashkey']  = action[['N','cards','board']].fillna('').apply(lambda x:pkr_to_hash(x.N,x.cards,x.board),axis=1)
action  = action.merge(w,how='left',left_on='hashkey',right_index=True,copy=False)
del w

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

#-- Board Texture --#
t0 = time.clock()
board_tt  = action[['board']].dropna().drop_duplicates()
temp      = board_tt.board.apply(board_texture)
board_tt  = pd.concat([board_tt,temp],1).set_index('board')
print(time.clock() - t0)

action    = action.merge(board_tt,how='left',left_on='board',right_index=True,copy=False)
action[board_tt.columns] = action[board_tt.columns].fillna(0)

exit(0)

# Predict winMoney>0 given game state + hole cards + action
#                   Deal     Flop     Turn    River
# tr  acc          79.46    86.93    88.23    89.89
#     f1           61.88    85.03    87.58    89.44
#     precision    87.91    89.76    90.39    91.04
#     recall       47.74    80.78    84.95    87.90
# tt  acc          70.46    69.01    69.27    74.35
#     f1           42.57    63.57    66.99    72.85
#     precision    66.23    69.15    70.52    75.14
#     recall       31.36    58.83    63.80    70.71

# Predict winMoney>0 given game state + hole cards + prWin + action
#                   Deal     Flop     Turn    River
# tr  acc          80.25    89.09    90.19    90.82
#     f1           63.77    87.62    89.70    90.42
#     precision    88.64    91.53    92.12    91.91
#     recall       49.80    84.03    87.41    88.98
# tt  acc          70.42    70.20    71.37    78.23
#     f1           42.81    65.37    69.01    77.00
#     precision    65.85    70.17    73.28    79.31
#     recall       31.71    61.19    65.21    74.82

# Predict winMoney>0 given game state + hole cards + prWin + action (One Model)
#                    Deal     Flop     Turn    River
# tr  acc           80.92    91.78    93.05    93.95
#     f1            65.14    90.57    92.71    93.71
#     precision     88.95    94.38    94.29    95.01
#     recall        51.38    87.06    91.19    92.46
# tt  acc           70.78    71.14    73.88    79.77
#     f1            43.73    65.41    71.52    78.40
#     precision     65.87    71.68    75.82    81.79
#     recall        32.72    60.15    67.71    75.28

mask  = (action.Nsim>0) & (action.action!='fold') #(action.roundName=='Deal') &  ##& (action.winMoney>0) # & target_action.playerName.isin(target_players) #
print(action[mask].action.value_counts())

X  = action.loc[mask,[
    'game_phase','blind_level','smallBlind','roundName','chips','position','pot','bet','N','Nnf','Nallin','pot_sum','bet_sum','maxBet','NMaxBet','Nfold','Ncall','Nraise','self_Ncall','self_Nraise','prev_action','NRfold','NRcall','NRraise','pos','op_resp',
    'cards','cards_rank1','cards_rank2','cards_rank_sum','cards_aces','cards_faces','cards_pair','cards_suit','cards_conn','cards_conn2','cards_category',
    'hand_score0','hand_score1',
    'board','board_rank1','board_rank2','board_aces','board_faces','board_kind','board_kind_rank','board_suit','board_suit_rank','board_conn','board_conn_rank',
    'prWin',
    'action','amount',]].copy() #
y  = action.loc[mask,['winMoney','chips_final']].copy()

#-- Preprocess Features --#
P  = X.pot_sum + X.bet_sum
X  = pd.concat([X,
    pd.get_dummies(X.roundName).reindex(columns=['Deal','Flop','Turn','River']).fillna(0),
    pd.get_dummies(X.pos,prefix='pos',prefix_sep='=')[['pos='+x for x in ('E','M','L','B')]],
    pd.get_dummies(X.op_resp,prefix='op_resp',prefix_sep='=')[['op_resp='+x for x in ('none','all_folded','any_called','any_raised','any_reraised')]],
    ],1)
X['chips_SB']  = X.chips / X.smallBlind
X['chips_P']   = X.chips / P
X['pot_P']     = X.pot / P
X['pot_SB']    = X.pot / X.smallBlind
X['bet_P']     = X.bet / P
X['bet_SB']    = X.bet / X.smallBlind
X['bet_sum_P'] = X.bet_sum / P
X['bet_sum_SB'] = X.bet_sum / X.smallBlind
minBet  = np.minimum(X.maxBet - X.bet,X.chips)
X['minBet_P']  = minBet / P
X['minBet_SB'] = minBet / X.smallBlind
X  = pd.concat([X,
    pd.get_dummies(X.prev_action,prefix='prev',prefix_sep='=')[['prev='+x for x in ('none','check/call','bet/raise/allin')]],
    pd.get_dummies(X.action,prefix='action',prefix_sep='=')[['action='+x for x in ('check/call','bet/raise/allin',)]],
    ],1)
X['amount_P']  = X.amount / P
X['amount_SB'] = X.amount / X.smallBlind
X['amount_chips']  = X.amount / X.chips
X.drop(['smallBlind','roundName','chips','position','board','pot','bet','pot_sum','bet_sum','maxBet','prev_action','pos','op_resp',
    'cards',
    'action','amount',
    ],'columns',inplace=True)

gbc = GradientBoostingClassifier(loss='deviance',learning_rate=0.1,n_estimators=100,subsample=1.0,criterion='friedman_mse',min_samples_leaf=4,max_depth=3,min_impurity_decrease=0.0,min_impurity_split=None,init=None,random_state=0,max_features=None,verbose=2,max_leaf_nodes=None,warm_start=False,presort='auto')
rf  = RandomForestClassifier(n_estimators=100,max_depth=None,min_samples_leaf=4,oob_score=False,n_jobs=1,random_state=0,verbose=2,warm_start=False,class_weight=None)
lr  = LogisticRegression(penalty='l2',dual=False,tol=0.0001,C=1.0,fit_intercept=True,intercept_scaling=1,class_weight=None,random_state=0,solver='liblinear',max_iter=100,multi_class='ovr',verbose=0,warm_start=False,n_jobs=1)
model = rf

model.fit(X,y.winMoney>0)#y.action=='fold')
joblib.dump({'col':X.columns.tolist(),'model':model},'pkrprb_winMoney_rf2.pkl')
out  = joblib.load('pkrprb_winMoney_rf2.pkl')
t0  = time.clock()
yhat  = model.predict(X)
print(time.clock() - t0)
accuracy_score(y.winMoney>0,yhat)# y.action=='fold',yhat)

kf  = StratifiedKFold(n_splits=4,shuffle=True,random_state=0)
results_tr  = pd.DataFrame()#columns=('acc','f1'))
results_tt  = pd.DataFrame()#columns=('acc','f1'))
feat_rank   = []
for i,(idx_tr,idx_tt) in enumerate(kf.split(X,y.winMoney>0)): #y.action=='fold')):
    X_tr  = X.iloc[idx_tr]
    y_tr  = y.iloc[idx_tr]
    X_tt  = X.iloc[idx_tt]
    y_tt  = y.iloc[idx_tt]
    # #
    # X_tr  = X_tr.sample(2*(y_tr.action=='fold').sum())
    # y_tr  = y_tr.loc[X_tr.index]
    #
    model.fit(X_tr,y_tr.winMoney>0)#.action=='fold')
    yhat_tr  = model.predict_proba(X_tr)[:,1]>0.5
    yhat_tt  = model.predict_proba(X_tt)[:,1]>0.5
    #
    for roundName in ('Deal','Flop','Turn','River'):
        results_tr.loc[i,roundName+'_'+'acc'] = accuracy_score(y_tr[X_tr[roundName]>0].winMoney>0,yhat_tr[X_tr[roundName]>0])#.action=='fold',yhat_tr)
        results_tr.loc[i,roundName+'_'+'f1']  = f1_score(y_tr[X_tr[roundName]>0].winMoney>0,yhat_tr[X_tr[roundName]>0])#.action=='fold',yhat_tr)
        results_tr.loc[i,roundName+'_'+'precision'] = precision_score(y_tr[X_tr[roundName]>0].winMoney>0,yhat_tr[X_tr[roundName]>0])#.action=='fold',yhat_tr)
        results_tr.loc[i,roundName+'_'+'recall']    = recall_score(y_tr[X_tr[roundName]>0].winMoney>0,yhat_tr[X_tr[roundName]>0])#.action=='fold',yhat_tr)
        results_tt.loc[i,roundName+'_'+'acc'] = accuracy_score(y_tt[X_tt[roundName]>0].winMoney>0,yhat_tt[X_tt[roundName]>0])#.action=='fold',yhat_tt)
        results_tt.loc[i,roundName+'_'+'f1']  = f1_score(y_tt[X_tt[roundName]>0].winMoney>0,yhat_tt[X_tt[roundName]>0])#.action=='fold',yhat_tt)
        results_tt.loc[i,roundName+'_'+'precision'] = precision_score(y_tt[X_tt[roundName]>0].winMoney>0,yhat_tt[X_tt[roundName]>0])#.action=='fold',yhat_tt)
        results_tt.loc[i,roundName+'_'+'recall']    = recall_score(y_tt[X_tt[roundName]>0].winMoney>0,yhat_tt[X_tt[roundName]>0])#.action=='fold',yhat_tt)
    #
    if isinstance(model,RandomForestClassifier):
        feat_rank.append(pd.Series(model.feature_importances_,index=X_tr.columns))
    elif isinstance(model,LogisticRegression):
        feat_rank.append(pd.Series(np.r_[model.intercept_,model.coef_[0,:]],index=['intercept_']+X_tr.columns.tolist()))

results  = pd.concat([results_tr,results_tt],1,keys=('tr','tt'))
feat_rank = pd.concat(feat_rank,1).mean(1)
print(feat_rank.sort_values(ascending=False))
print((100*results.mean(0)).round(2))
confusion_matrix(y_tt.winMoney>0,yhat_tt)#.action=='fold',yhat_tt)#,labels=['fold','check/call','bet/raise/allin'])


# X:
# 'smallBlind','chips','position','board','pot','bet','N','Nnf','Nallin','pot_sum','bet_sum','maxBet','NMaxBet','NRfold','NRcheck','NRcall','NRbet','NRraise','NRallincall','NRallin','score0','score1'
# y:
# ,'action','amount'
