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
pd.set_option('display.width',80)

#-- Read Data --#
DATE_START  = '20180723'
DATE_END    = '20180803'
DATE_TEST   = '20180803'
dt_range    = pd.date_range(DATE_START,DATE_END,freq='B').strftime('%Y%m%d')
rnd     = pd.concat([pd.read_csv('data/round_log_'+dt+'.gz') for dt in dt_range],0,ignore_index=True,sort=False)
action  = pd.concat([pd.read_csv('data/action_proc_'+dt+'.gz') for dt in dt_range],0,ignore_index=True,sort=False)
action['timestamp']  = pd.to_datetime(action.timestamp)
action_tt  = pd.read_csv('data/action_proc_'+DATE_TEST+'.gz')

#-- Extract Opponents Features --#
t0  = time.clock()
action['opponents']  = action.opponents.apply(eval)
print(time.clock() - t0)

t0  = time.clock()
action['op_playerName']  = action.opponents.apply(lambda x:[y['playerName'] for y in x])
action['op_chips'] = action.opponents.apply(lambda x:[y['chips'] for y in x])
action['op_pot']   = action.opponents.apply(lambda x:[y['roundBet'] for y in x])
action['op_bet']   = action.opponents.apply(lambda x:[y['bet'] for y in x])
print(time.clock() - t0)

action['op_chips_max']  = action.op_chips.apply(lambda x:max(x) if len(x) else 0)
action['op_chips_min']  = action.op_chips.apply(lambda x:min(x) if len(x) else 0)
action['op_chips_mean'] = (action.op_chips.apply(lambda x:sum(x) if len(x) else 0) / (action.Nnf - 1)).fillna(0)

def get_op_raiser_idx(x):
    for i,y in enumerate(x.op_playerName):
        if y == x.op_raiser: return i

t0  = time.clock()
action['op_raiser_idx']  = action.apply(lambda x:get_op_raiser_idx(x) if pd.notnull(x.op_raiser) else -1,axis=1)
print(time.clock() - t0)

t0  = time.clock()
mask  = action.op_raiser_idx >= 0
action.loc[mask,'op_raiser_chips'] = action[mask].apply(lambda x:x.op_chips[x.op_raiser_idx],axis=1)
action.loc[mask,'op_raiser_pot']   = action[mask].apply(lambda x:x.op_pot[x.op_raiser_idx],axis=1)
action.loc[mask,'op_raiser_bet']   = action[mask].apply(lambda x:x.op_bet[x.op_raiser_idx],axis=1)
print(time.clock() - t0)

#-- Hand Score --#
action['hand_score']  = action.hand_score.apply(eval)
action['hand_score0'] = action.hand_score.str[0]
action['hand_score1'] = action.hand_score.str[1]
action['hand_score2'] = action.hand_score.str[2].fillna(0).astype(int)

action_tt['hand_score']  = action_tt.hand_score.apply(eval)
action_tt['hand_score0'] = action_tt.hand_score.str[0]
action_tt['hand_score1'] = action_tt.hand_score.str[1]
action_tt['hand_score2'] = action_tt.hand_score.str[2].fillna(0).astype(int)

# #-- Baseline Accuracy for Predicting Hand Score Win --#
# target_col  = 'winRiver'
# thd_range   = np.arange(0.1,1,0.1)
# results_acc = pd.DataFrame(index=('Deal','Flop','Turn','River'),columns=thd_range)
# results_f1  = pd.DataFrame(index=('Deal','Flop','Turn','River'),columns=thd_range)
# results_precision = pd.DataFrame(index=('Deal','Flop','Turn','River'),columns=thd_range)
# results_recall    = pd.DataFrame(index=('Deal','Flop','Turn','River'),columns=thd_range)
# for thd in thd_range:
#     results_acc[thd]  = action.groupby('roundName')[['prWin',target_col]].agg(lambda x:accuracy_score(x[target_col],x.prWin>thd))[target_col].values
#     results_f1[thd]   = action.groupby('roundName')[['prWin',target_col]].agg(lambda x:f1_score(x[target_col],x.prWin>thd))[target_col].values
#     results_precision[thd]  = action.groupby('roundName')[['prWin',target_col]].agg(lambda x:precision_score(x[target_col],x.prWin>thd))[target_col].values
#     results_recall[thd]     = action.groupby('roundName')[['prWin',target_col]].agg(lambda x:recall_score(x[target_col],x.prWin>thd))[target_col].values

# results  = pd.concat([results_acc,results_f1,results_precision,results_recall],0,keys=('acc','f1','precision','recall',))
# results['max']  = results.max(1)

exit(0)

def cross_validation(kfold,model,X,y):
    results_tr  = pd.DataFrame()
    results_tt  = pd.DataFrame()
    feat_rank   = []
    for i,(idx_tr,idx_tt) in enumerate(kfold.split(X,y)):
        X_tr  = X.iloc[idx_tr]
        y_tr  = y.iloc[idx_tr]
        X_tt  = X.iloc[idx_tt]
        y_tt  = y.iloc[idx_tt]
        #
        model.fit(X_tr,y_tr)
        yhat_tr  = model.predict_proba(X_tr)[:,1]>0.5
        yhat_tt  = model.predict_proba(X_tt)[:,1]>0.5
        #
        results_tr.loc[i,'N']   = len(X_tr)
        results_tr.loc[i,'acc'] = 100*accuracy_score(y_tr,yhat_tr)
        results_tr.loc[i,'f1']  = 100*f1_score(y_tr,yhat_tr)
        results_tr.loc[i,'precision'] = 100*precision_score(y_tr,yhat_tr)
        results_tr.loc[i,'recall']    = 100*recall_score(y_tr,yhat_tr)
        results_tt.loc[i,'N']   = len(X_tt)
        results_tt.loc[i,'acc'] = 100*accuracy_score(y_tt,yhat_tt)
        results_tt.loc[i,'f1']  = 100*f1_score(y_tt,yhat_tt)
        results_tt.loc[i,'precision'] = 100*precision_score(y_tt,yhat_tt)
        results_tt.loc[i,'recall']    = 100*recall_score(y_tt,yhat_tt)
        #
        if isinstance(model,RandomForestClassifier):
            feat_rank.append(pd.Series(model.feature_importances_,index=X_tr.columns))
        elif isinstance(model,LogisticRegression):
            feat_rank.append(pd.Series(np.r_[model.intercept_,model.coef_[0,:]],index=['intercept_']+X_tr.columns.tolist()))
    #
    results   = pd.concat([results_tr,results_tt],1,keys=('tr','cv'))
    feat_rank = pd.concat(feat_rank,1)
    print(feat_rank.mean(1).sort_values(ascending=False))
    print(results.mean().round(2))
    print(confusion_matrix(y_tt,yhat_tt))
    #
    return results,feat_rank

rf  = RandomForestClassifier(n_estimators=100,max_depth=None,min_samples_leaf=5,oob_score=False,n_jobs=1,random_state=0,verbose=2,warm_start=False,class_weight=None)
model = rf

#-- Train Agent Model --#
mask = action.roundName == 'River'
feat = ['N','prWin','cards','hand','board','Naction','minBet',]
X    = compile_features_batch(action[mask],feat)
y    = action.loc[mask,'win'].copy()
model.fit(X,y)
y_hat = model.predict_proba(X)[:,1]
print(accuracy_score(y,y_hat>0.5),f1_score(y,y_hat>0.5),precision_score(y,y_hat>0.5),recall_score(y,y_hat>0.5))
joblib.dump({'feat':feat,'col':X.columns.tolist(),'model':model},'pkrprb_win_rf_temp.pkl')

#-- Temporal Train Test Split --#
mask     = action.roundName == 'River'
mask_tt  = action_tt.roundName == 'River'
feat     = ['N','prWin','cards','hand','board','Naction','minBet',] #,'pos','op_resp','op_chips','prev'
targets  = ['win','winRiver','winMoney',]

X    = compile_features_batch(action[mask],feat)
y    = action.loc[mask,targets].copy()
acc  = pd.DataFrame(index=['N','acc','f1','precision','recall'],columns=targets)
X_tt    = compile_features_batch(action_tt[mask_tt],feat)
y_tt    = action_tt.loc[mask_tt,targets].copy()
acc_tt  = pd.DataFrame(index=['N','acc','f1','precision','recall'],columns=targets)
for col in targets:
    model.fit(X,y[col]>0)
    y[col+'_hat']    = model.predict_proba(X)[:,1]
    y_tt[col+'_hat'] = model.predict_proba(X_tt)[:,1]
    acc.loc['N',col]   = len(X)
    acc.loc['acc',col] = accuracy_score(y[col]>0,y[col+'_hat']>0.5)
    acc.loc['f1',col]  = f1_score(y[col]>0,y[col+'_hat']>0.5)
    acc.loc['precision',col] = precision_score(y[col]>0,y[col+'_hat']>0.5)
    acc.loc['recall',col]    = recall_score(y[col]>0,y[col+'_hat']>0.5)
    acc_tt.loc['N',col]   = len(X_tt)
    acc_tt.loc['acc',col] = accuracy_score(y_tt[col]>0,y_tt[col+'_hat']>0.5)
    acc_tt.loc['f1',col]  = f1_score(y_tt[col]>0,y_tt[col+'_hat']>0.5)
    acc_tt.loc['precision',col] = precision_score(y_tt[col]>0,y_tt[col+'_hat']>0.5)
    acc_tt.loc['recall',col]    = recall_score(y_tt[col]>0,y_tt[col+'_hat']>0.5)

results  = pd.concat([acc,acc_tt],0,keys=('tr','tt',))

kf  = StratifiedKFold(n_splits=4,shuffle=True,random_state=0)
results  = {}
for col in targets:
    res,feat_rank  = cross_validation(kf,model,X,y[col]>0)
    results[col]  = res.mean(0)

results  = pd.concat(results,1)[targets]
print(results.round(2))
results.round(2).to_clipboard('\t')

# River              prWin     +N/Nnf +Nfold/call/raise  +board      +hand
# tr  N            4406.25    4406.25      4406.25      4406.25    4406.25
#     acc            82.43      83.58        83.44        87.65      88.51
#     f1             79.03      80.34        80.29        85.34      86.31
#     precision      81.33      82.92        82.35        87.34      88.63
#     recall         76.85      77.93        78.35        83.44      84.12
# tt  N            1468.75    1468.75      1468.75      1468.75    1468.75
#     acc            73.92      76.65        77.16        77.58      78.18
#     f1             68.69      71.99        72.96        73.31      73.80
#     precision      71.09      74.42        74.41        75.22      76.36
#     recall         66.44      69.72        71.58        71.50      71.42

# River       +prWin_delta     +cards
# tr  N            4406.25    4406.25
#     acc            89.43      90.48
#     f1             87.40      88.68
#     precision      89.83      90.85
#     recall         85.10      86.63
# tt  N            1468.75    1468.75
#     acc            78.11      78.42
#     f1             73.80      74.22
#     precision      76.16      76.33
#     recall         71.58      72.26

#                       Deal        Flop        Turn       River
# tr  N            208956.75    98355.75    46203.00    30080.25
#     acc              77.75       86.58       87.65       89.92
#     f1               31.41       76.26       83.43       87.63
#     precision        70.43       90.96       90.25       90.31
#     recall           20.21       65.65       77.57       85.11
# tt  N             69652.25    32785.25    15401.00    10026.75
#     acc              75.65       76.28       75.88       79.77
#     f1               24.59       55.94       66.72       74.93
#     precision        56.06       71.69       74.63       78.00
#     recall           15.75       45.87       60.32       72.09


mask  = (action.Nsim>0) & (action.action!='fold') #(action.roundName=='Deal') &  ##& (action.winMoney>0) # & target_action.playerName.isin(target_players) #
print(action[mask].action.value_counts())

X  = action.loc[mask,[
    'game_phase','blind_level','smallBlind','roundName',
    'chips','position','pos','pot','bet','N','Nnf','Nallin','pot_sum','bet_sum','maxBet','NMaxBet',
    'Nfold','Ncall','Nraise','self_Ncall','self_Nraise',
    'prev_action','NRfold','NRcall','NRraise','op_resp',
    'op_chips_max','op_chips_min','op_chips_mean',
    'op_raiser_chips','op_raiser_pot','op_raiser_bet',
    'cards_rank1','cards_rank2','cards_rank_sum','cards_aces','cards_faces','cards_pair','cards_suit','cards_conn','cards_conn2','cards_category',
    'hand_score0','hand_score1','hand_score2',
    'board_rank1','board_rank2','board_aces','board_faces','board_kind','board_kind_rank','board_suit','board_suit_rank','board_conn','board_conn_rank',
    'prWin','prWin_delta',
    'action','amount',
    ]].copy()
y  = action.loc[mask,[
    'winMoney','chips_final',
    ]].copy()

#-- Preprocess Features --#
P  = X.pot_sum + X.bet_sum
X  = pd.concat([X,
    pd.get_dummies(X.game_phase,prefix='phase',prefix_sep='=')[['phase='+x for x in ('Early','Middle','Late',)]],
    pd.get_dummies(X.roundName).reindex(columns=['Deal','Flop','Turn','River']).fillna(0),
    pd.get_dummies(X.pos,prefix='pos',prefix_sep='=')[['pos='+x for x in ('E','M','L','B')]],
    pd.get_dummies(X.op_resp,prefix='op_resp',prefix_sep='=')[['op_resp='+x for x in ('none','all_folded','any_called','any_raised','any_reraised')]],
    ],1)
X['op_chips_max']    /= X.chips
X['op_chips_min']    /= X.chips
X['op_chips_mean']   /= X.chips
X['op_raiser_chips'] /= X.chips
X['op_raiser_pot']   /= X.pot_sum
X['op_raiser_bet']   /= X.bet_sum
X['op_raiser_chips_max']  = X.op_raiser_chips / X.op_chips_max
X['chips_SB']  = X.chips / X.smallBlind
X['chips_P']   = X.chips / P
X['pot_P']     = X.pot / P
X['pot_SB']    = X.pot / X.smallBlind
X['bet_P']     = X.bet / P
X['bet_SB']    = X.bet / X.smallBlind
X['bet_sum_P'] = X.bet_sum / P
X['bet_sum_SB'] = X.bet_sum / X.smallBlind
cost_to_call  = np.minimum(X.maxBet - X.bet,X.chips)
X['minBet_P']  = cost_to_call / P
X['minBet_SB'] = cost_to_call / X.smallBlind
X['minBet_chips']  = cost_to_call / X.chips
X  = pd.concat([X,
    pd.get_dummies(X.prev_action,prefix='prev',prefix_sep='=')[['prev='+x for x in ('none','check/call','bet/raise/allin')]],
    pd.get_dummies(X.action,prefix='action',prefix_sep='=')[['action='+x for x in ('check/call','bet/raise/allin',)]],
    ],1)
X['amount_P']  = X.amount / P
X['amount_SB'] = X.amount / X.smallBlind
X['amount_chips']  = X.amount / X.chips
X.drop(['game_phase','smallBlind','roundName',
    'chips','position','pot','bet','pot_sum','bet_sum','maxBet','prev_action','pos','op_resp',
    'action','amount',
    ],'columns',inplace=True)
X.fillna(0,inplace=True)

gbc = GradientBoostingClassifier(loss='deviance',learning_rate=0.1,n_estimators=100,subsample=1.0,criterion='friedman_mse',min_samples_leaf=4,max_depth=3,min_impurity_decrease=0.0,min_impurity_split=None,init=None,random_state=0,max_features=None,verbose=2,max_leaf_nodes=None,warm_start=False,presort='auto')
rf  = RandomForestClassifier(n_estimators=100,max_depth=None,min_samples_leaf=4,oob_score=False,n_jobs=1,random_state=0,verbose=2,warm_start=False,class_weight=None)
lr  = LogisticRegression(penalty='l2',dual=False,tol=0.0001,C=1.0,fit_intercept=True,intercept_scaling=1,class_weight=None,random_state=0,solver='liblinear',max_iter=100,multi_class='ovr',verbose=0,warm_start=False,n_jobs=1)
model = rf

t0    = time.clock()
model.fit(X,y.winMoney>0) #y.action=='fold') #
# joblib.dump({'col':X.columns.tolist(),'model':model},'pkrprb_action_rf_temp.pkl')
# out  = joblib.load('pkrprb_winMoney_rf2.pkl')
yhat  = model.predict(X)
feat_rank = pd.Series(model.feature_importances_,index=X.columns)
print(time.clock() - t0)
print(accuracy_score(y.winMoney>0,yhat)) #y.action=='fold',yhat)) #

exit(0)

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
        results_tr.loc[i,'N_'+roundName]      = (X_tr[roundName]>0).sum()
        results_tr.loc[i,'acc'+'_'+roundName] = 100*accuracy_score(y_tr[X_tr[roundName]>0].winMoney>0,yhat_tr[X_tr[roundName]>0])#.action=='fold',yhat_tr)
        results_tr.loc[i,'f1'+'_'+roundName]  = 100*f1_score(y_tr[X_tr[roundName]>0].winMoney>0,yhat_tr[X_tr[roundName]>0])#.action=='fold',yhat_tr)
        results_tr.loc[i,'precision'+'_'+roundName] = 100*precision_score(y_tr[X_tr[roundName]>0].winMoney>0,yhat_tr[X_tr[roundName]>0])#.action=='fold',yhat_tr)
        results_tr.loc[i,'recall'+'_'+roundName]    = 100*recall_score(y_tr[X_tr[roundName]>0].winMoney>0,yhat_tr[X_tr[roundName]>0])#.action=='fold',yhat_tr)
        results_tt.loc[i,'N_'+roundName]      = (X_tt[roundName]>0).sum()
        results_tt.loc[i,'acc'+'_'+roundName] = 100*accuracy_score(y_tt[X_tt[roundName]>0].winMoney>0,yhat_tt[X_tt[roundName]>0])#.action=='fold',yhat_tt)
        results_tt.loc[i,'f1'+'_'+roundName]  = 100*f1_score(y_tt[X_tt[roundName]>0].winMoney>0,yhat_tt[X_tt[roundName]>0])#.action=='fold',yhat_tt)
        results_tt.loc[i,'precision'+'_'+roundName] = 100*precision_score(y_tt[X_tt[roundName]>0].winMoney>0,yhat_tt[X_tt[roundName]>0])#.action=='fold',yhat_tt)
        results_tt.loc[i,'recall'+'_'+roundName]    = 100*recall_score(y_tt[X_tt[roundName]>0].winMoney>0,yhat_tt[X_tt[roundName]>0])#.action=='fold',yhat_tt)
    #
    if isinstance(model,RandomForestClassifier):
        feat_rank.append(pd.Series(model.feature_importances_,index=X_tr.columns))
    elif isinstance(model,LogisticRegression):
        feat_rank.append(pd.Series(np.r_[model.intercept_,model.coef_[0,:]],index=['intercept_']+X_tr.columns.tolist()))

results  = pd.concat([results_tr,results_tt],1,keys=('tr','tt'))
results.columns  = pd.MultiIndex.from_tuples([tuple([x[0]]+x[1].split('_')) for x in results.columns])
results  = pd.concat([results.loc[:,(x,y)] for x in ('tr','tt',) for y in ('N','acc','f1','precision','recall',)],0,keys=[(x,y) for x in ('tr','tt',) for y in ('N','acc','f1','precision','recall',)])
feat_rank = pd.concat(feat_rank,1).mean(1)
print(feat_rank.sort_values(ascending=False))
print((100*results.groupby(level=[0,1]).mean()).round(2))
confusion_matrix(y_tt.winMoney>0,yhat_tt) #y.action=='fold',yhat_tt)# ,labels=['fold','check/call','bet/raise/allin'])
