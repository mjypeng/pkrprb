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

def board_texture(board):
    c    = board.lower().split()
    o,s  = zip(*[(rankmap[cc[0]],cc[1]) for cc in c])
    o,s  = np.asarray(o),np.asarray(s)
    #
    X    = pd.Series()
    X['board_rank']      = max(o)
    X['board_faces']     = sum(o>=10)
    order,counts = np.unique(o,return_counts=True)
    order,counts = order[::-1],counts[::-1]
    idx  = counts.argmax()
    X['board_kind']      = counts[idx]
    X['board_kind_rank'] = order[idx]
    suit,counts  = np.unique(s,return_counts=True)
    idx  = counts.argmax()
    X['board_suit']      = counts[idx]
    X['board_suit_rank'] = o[s==suit[idx]].max()
    X['board_conn']      = 0
    X['board_conn_rank'] = 0
    #
    temp = np.sort(np.unique(o))
    if 14 in temp: temp = np.r_[1,temp]
    dtemp = np.diff(temp)
    if (dtemp==1).all():
        X['board_conn']      = len(temp)
        X['board_conn_rank'] = temp[-1]
    elif (dtemp[1:]==1).all():
        X['board_conn']      = len(temp)-1
        X['board_conn_rank'] = temp[-1]
    elif (dtemp[:-1]==1).all():
        X['board_conn']      = len(temp)-1
        X['board_conn_rank'] = temp[-2]
    elif len(temp) >= 4:
        if (dtemp[2:]==1).all():
            X['board_conn']      = len(temp)-2
            X['board_conn_rank'] = temp[-1]
        elif (dtemp[1:-1]==1).all():
            X['board_conn']      = len(temp)-2
            X['board_conn_rank'] = temp[-2]
        elif (dtemp[:-2]==1).all():
            X['board_conn']      = len(temp)-2
            X['board_conn_rank'] = temp[-3]
        elif len(temp) >= 5:
            if (dtemp[3:]==1).all():
                X['board_conn']      = len(temp)-3
                X['board_conn_rank'] = temp[-1]
            elif (dtemp[2:-1]==1).all():
                X['board_conn']      = len(temp)-3
                X['board_conn_rank'] = temp[-2]
            elif (dtemp[1:-2]==1).all():
                X['board_conn']      = len(temp)-3
                X['board_conn_rank'] = temp[-3]
            elif (dtemp[:-3]==1).all():
                X['board_conn']      = len(temp)-3
                X['board_conn_rank'] = temp[-4]
    #
    return X

def position_feature(N,pos):
    pos = pos.copy()
    pos[pos.isin(('1','2'))] = 'E'
    pos[pos=='D']  = 'L'
    pos[pos.isin(('SB','BB'))] = 'B'
    mask = ~pos.isin(('E','L','B'))
    pos[mask] = pos[mask].astype(int)
    pos[pos==(N-3)] = 'L'
    pos[~pos.isin(('E','L','B'))] = 'M'
    return pos

def opponent_response(prev_action,NRfold,NRcall,NRraise):
    return np.where(prev_action=='bet/raise/allin','any_reraised',
            np.where(NRraise>0,'any_raised',
                np.where(NRcall>0,'any_called',
                    np.where(NRfold>0,'all_folded','none'))))

#-- Read Data --#
# dt      = sys.argv[1] #'20180716'
action  = pd.concat([pd.read_csv(f) for f in glob.glob('data/target_action_*.gz')],0)

#-- Hole Cards Texture --#
action  = pd.concat([action,hole_texture_batch(action.cards)],1)

#-- Table Position --#
action['pos']     = position_feature(action.N,action.position)

#-- Opponent Response --#
action['op_resp'] = opponent_response(action.prev_action,action.NRfold,action.NRcall,action.NRraise)

#-- Hand Win Probability @River --#
w  = pd.read_csv('precal_win_prob.gz',index_col='hashkey')
action['hashkey']  = action[['N','cards','board']].fillna('').apply(lambda x:pkr_to_hash(x.N,x.cards,x.board),axis=1)
action  = action.merge(w,how='left',left_on='hashkey',right_index=True,copy=False)
del w

#-- Deal hand score --#
mask  = action.roundName == 'Deal'
action.loc[mask,'hand_score0']  = action.loc[mask,'cards_pair'].astype(int)
action.loc[mask,'hand_score1']  = action.loc[mask,'cards_rank1']
action.loc[mask,'board_score0'] = 0
action.loc[mask,'board_score1'] = 0

#-- Flop hand score --#
t0  = time.clock()
mask  = action.roundName == 'Flop'
temp  = (action[mask].cards + ' ' + action[mask].board).str.split().apply(lambda x:pd.Series(score_hand5(pkr_to_cards(x))[:2]))
action.loc[mask,'hand_score0']  = temp[0]
action.loc[mask,'hand_score1']  = temp[1]
print(time.clock() - t0)

#-- Turn/River hand score --#
t0  = time.clock()
mask  = action.roundName.isin(('Turn','River'))
temp  = (action[mask].cards + ' ' + action[mask].board.fillna('')).str.split().apply(lambda x:pd.Series(score_hand(pkr_to_cards(x))[:2]))
action.loc[mask,'hand_score0']  = temp[0]
action.loc[mask,'hand_score1']  = temp[1]
temp  = action[mask].board.str.split().apply(lambda x:pd.Series(score_hand(pkr_to_cards(x))[:2]))
action.loc[mask,'board_score0'] = temp[0]
action.loc[mask,'board_score1'] = temp[1]
print(time.clock() - t0)

#-- Board Texture --#
t0 = time.clock()
board_tt  = action[['board']].dropna().drop_duplicates()
temp      = board_tt.board.apply(board_texture)
board_tt  = pd.concat([board_tt,temp],1).set_index('board')
action    = action.merge(board_tt,how='left',left_on='board',right_index=True,copy=False)
print(time.clock() - t0)

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
# tt  acc           70.53    70.60    72.79    79.28
#     f1            43.01    65.41    70.78    77.93
#     precision     65.47    71.19    74.45    80.97
#     recall        32.02    60.51    67.48    75.11

action = action[action.Nsim>0]

mask  = (action.action!='fold') #(action.roundName=='Deal') &  ##& (action.winMoney>0) # & target_action.playerName.isin(target_players) #
print(action[mask].action.value_counts())

# # Board score
# temp  = action[mask].board.str.split().apply(lambda x:pd.Series(score_hand5(pkr_to_cards(x))[:2]))
# action.loc[mask,'board_score0']  = temp[0]
# action.loc[mask,'board_score1']  = temp[1]

X  = action.loc[mask,[
    'smallBlind','roundName','chips','position','board','pot','bet','N','Nnf','Nallin','pot_sum','bet_sum','maxBet','NMaxBet','Nfold','Ncall','Nraise','self_Ncall','self_Nraise','prev_action','NRfold','NRcall','NRraise','pos','op_resp',
    'cards','cards_rank1','cards_rank2','cards_aces','cards_faces','cards_pair','cards_suit','cards_conn',
    'hand_score0','hand_score1',
    'prWin',
    'action','amount',]].copy() #
y  = action.loc[mask,['winMoney','chips_final']].copy()

#-- Preprocess Features --#
P  = X.pot_sum + X.bet_sum
X  = pd.concat([X,
    pd.get_dummies(X.roundName)[['Deal','Flop','Turn','River']],
    pd.get_dummies(X.pos,prefix='pos',prefix_sep='=')[['pos='+x for x in ('E','M','L','B')]],
    pd.get_dummies(X.op_resp,prefix='op_resp',prefix_sep='=')[['op_resp='+x for x in ('none','all_folded','any_called','any_raised','any_reraised')]],
    ],1)
# X['chips_SB']  = X.chips / X.smallBlind
# X['chips_P']   = X.chips / P
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
X.drop(['smallBlind','roundName','chips','position','board','pot','bet','pot_sum','bet_sum','maxBet','prev_action','pos','op_resp',
    'cards',
    'action','amount',
    ],'columns',inplace=True)

gbc = GradientBoostingClassifier(loss='deviance',learning_rate=0.1,n_estimators=100,subsample=1.0,criterion='friedman_mse',min_samples_leaf=4,max_depth=3,min_impurity_decrease=0.0,min_impurity_split=None,init=None,random_state=0,max_features=None,verbose=2,max_leaf_nodes=None,warm_start=False,presort='auto')
rf  = RandomForestClassifier(n_estimators=100,max_depth=None,min_samples_leaf=4,oob_score=False,n_jobs=1,random_state=0,verbose=2,warm_start=False,class_weight=None)
lr  = LogisticRegression(penalty='l2',dual=False,tol=0.0001,C=1.0,fit_intercept=True,intercept_scaling=1,class_weight=None,random_state=0,solver='liblinear',max_iter=100,multi_class='ovr',verbose=0,warm_start=False,n_jobs=1)
model = rf

model.fit(X,y.winMoney>0)#y.action=='fold')
joblib.dump({'col':X.columns.tolist(),'model':model},'pkrprb_winMoney_rf.pkl')
out  = joblib.load('pkrprb_winMoney_rf.pkl')
t0  = time.clock()
yhat  = out['model'].predict(X)
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
