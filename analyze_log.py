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
dt      = sys.argv[1] if len(sys.argv)>1 else '*'#'20180716'
action  = pd.concat([pd.read_csv(f) for f in glob.glob('data/action_proc_'+dt+'.gz')],0,ignore_index=True,sort=False)

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

exit(0)

mask  = (action.Nsim>0) & (action.action!='fold') #(action.roundName=='Deal') &  ##& (action.winMoney>0) # & target_action.playerName.isin(target_players) #
print(action[mask].action.value_counts())

X  = action.loc[mask,[
    'game_phase','blind_level','smallBlind','roundName',
    'chips','position','pos','pot','bet','N','Nnf','Nallin','pot_sum','bet_sum','maxBet','NMaxBet',
    'Nfold','Ncall','Nraise','self_Ncall','self_Nraise',
    'prev_action','NRfold','NRcall','NRraise','op_resp',
    # 'op_chips_max','op_chips_min','op_chips_mean',
    # 'op_raiser_chips','op_raiser_pot','op_raiser_bet',
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
# X['op_chips_max']    /= X.chips
# X['op_chips_min']    /= X.chips
# X['op_chips_mean']   /= X.chips
# X['op_raiser_chips'] /= X.chips
# X['op_raiser_pot']   /= X.pot_sum
# X['op_raiser_bet']   /= X.bet_sum
# X['op_raiser_chips_max']  = X.op_raiser_chips / X.op_chips_max
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
# X['amount_P']  = X.amount / P
# X['amount_SB'] = X.amount / X.smallBlind
# X['amount_chips']  = X.amount / X.chips
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
print((results.groupby(level=[0,1]).mean()).round(2))
print(confusion_matrix(y_tt.winMoney>0,yhat_tt))

# 20180716
#                Deal   Flop   Turn  River
# tr acc        86.71  93.81  95.38  95.90
#    f1         78.63  92.85  95.21  95.73
#    precision  95.52  96.11  96.22  96.16
#    recall     66.81  89.81  94.23  95.29
# tt acc        70.17  71.45  75.93  80.96
#    f1         45.66  65.05  73.82  79.28
#    precision  68.45  71.93  78.55  83.55
#    recall     34.26  59.38  69.64  75.45

# 20180717
#                Deal   Flop   Turn  River
# tr acc        87.55  94.59  95.84  96.04
#    f1         81.87  94.12  95.69  95.88
#    precision  94.59  96.31  96.28  96.65
#    recall     72.16  92.04  95.11  95.12
# tt acc        68.83  70.29  74.99  79.94
#    f1         50.05  66.22  73.03  78.07
#    precision  66.58  71.25  76.68  82.88
#    recall     40.11  61.86  69.73  73.78

# 20180718
#                Deal   Flop   Turn  River
# tr acc        86.24  94.16  95.95  96.49
#    f1         75.73  93.64  95.90  96.45
#    precision  95.32  95.54  96.40  96.83
#    recall     62.82  91.81  95.42  96.07
# tt acc        71.06  72.11  74.99  80.60
#    f1         42.54  68.07  73.78  79.60
#    precision  66.19  73.26  77.12  83.24
#    recall     31.37  63.59  70.72  76.29

# 20180719
#                Deal   Flop   Turn  River
# tr acc        86.98  93.70  95.02  95.98
#    f1         76.19  92.72  94.76  95.85
#    precision  96.31  96.01  95.95  96.45
#    recall     63.03  89.65  93.60  95.26
# tt acc        71.79  72.51  75.35  79.96
#    f1         40.87  66.84  73.02  78.59
#    precision  66.49  72.68  77.09  82.03
#    recall     29.51  61.88  69.38  75.47

# 20180720
#                Deal   Flop   Turn  River
# tr acc        86.83  93.58  95.45  96.31
#    f1         76.32  92.62  95.12  96.30
#    precision  96.03  95.95  96.54  96.51
#    recall     63.32  89.52  93.73  96.10
# tt acc        72.52  72.26  74.71  79.78
#    f1         44.61  66.01  71.43  78.99
#    precision  68.77  73.69  76.74  82.25
#    recall     33.01  59.79  66.83  75.99

# 20180723
#                Deal   Flop   Turn  River
# tr acc        87.49  94.37  96.03  96.48
#    f1         77.81  93.33  95.70  96.33
#    precision  96.28  96.62  96.69  97.17
#    recall     65.29  90.27  94.72  95.51
# tt acc        71.39  72.70  76.28  81.40
#    f1         41.39  65.03  73.00  79.70
#    precision  66.40  73.82  77.75  84.42
#    recall     30.09  58.16  68.85  75.57

# 20180724
#                Deal   Flop   Turn  River
# tr acc        86.75  93.76  95.59  96.26
#    f1         75.33  92.63  95.17  96.18
#    precision  96.69  96.03  96.81  96.26
#    recall     61.71  89.47  93.59  96.11
# tt acc        73.70  72.86  77.18  80.45
#    f1         45.07  65.70  73.88  79.25
#    precision  71.46  73.61  78.62  82.67
#    recall     32.92  59.33  69.73  76.18

# All
#                Deal   Flop   Turn  River
# tr acc        85.61  93.17  94.23  94.84
#    f1         74.96  92.17  93.88  94.69
#    precision  94.63  95.61  95.36  95.52
#    recall     62.06  88.97  92.45  93.88
# tt acc        72.02  72.14  74.24  79.94
#    f1         46.23  66.39  71.54  78.84
#    precision  69.43  73.04  75.95  81.64
#    recall     34.65  60.85  67.61  76.23
