from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

pd.set_option('display.max_rows',120)
pd.set_option('display.max_columns',None)
pd.set_option('display.width',90)

def hole_texture_(x):
    o  = [rankmap[xx[0]] for xx in x]
    s  = [xx[1] for xx in x]
    o1 = max(o)
    o2 = min(o)
    d  = o1 - o2
    y  = pd.Series()
    y['cards_rank1'] = o1
    y['cards_rank2'] = o2
    y['cards_aces']  = o1 == 14
    y['cards_faces'] = o2 >= 10
    y['cards_pair']  = d == 0
    y['cards_suit']  = s[0] == s[1]
    y['cards_conn']  = (d==1) & (o1<=12) & (o2>=4)
    return y

def hole_texture(cards):
    X  = pd.DataFrame(index=cards.index)
    c  = cards.str.lower().str.split()
    c1 = c.str[0]
    c2 = c.str[1]
    o  = np.c_[c1.str[0].apply(lambda x:rankmap[x]),c2.str[0].apply(lambda x:rankmap[x])]
    s1 = c1.str[1]
    s2 = c2.str[1]
    o_max = o.max(1)
    o_min = o.min(1)
    #
    X['cards_rank1'] = o_max
    X['cards_rank2'] = o_min
    X['cards_aces']  = o_max == 14
    X['cards_faces'] = o_min >= 10
    X['cards_pair']  = o_max == o_min
    X['cards_suit']  = s1 == s2
    X['cards_conn']  = ((o_max-o_min)==1) & (o_max<=12) & (o_min>=4)
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
dt      = sys.argv[1] #'20180716'
action  = pd.read_csv('data/target_action_'+dt+'.gz')

temp  = hole_texture(action.cards)
action  = pd.concat([action,temp],1)
action['pos']  = position_feature(action.N,action.position)
action['op_resp'] = opponent_response(action.prev_action,action.NRfold,action.NRcall,action.NRraise)

exit(0)

mask  = (action.roundName=='Deal') & (action.action!='fold') ##& (action.winMoney>0) # & target_action.playerName.isin(target_players) #
print(action[mask].action.value_counts())

# temp  = target_action[mask].board.str.split().apply(lambda x:pd.Series(score_hand5(pkr_to_cards(x))[:2]))
# target_action.loc[mask,'score0']  = temp[0]
# target_action.loc[mask,'score1']  = temp[1]

X  = action.loc[mask,[
    'smallBlind','roundName','chips','position','board','pot','bet','N','Nnf','Nallin','pot_sum','bet_sum','maxBet','NMaxBet','Nfold','Ncall','Nraise','self_Ncall','self_Nraise','prev_action','NRfold','NRcall','NRraise','pos','op_resp',
    # 'score0','score1',
    'cards','cards_rank1','cards_rank2','cards_aces','cards_faces',       'cards_pair','cards_suit','cards_conn',
    'action','amount',]].copy() #
y  = action.loc[mask,['winMoney','chips_final']].copy()

#-- Preprocess Features --#
P  = X.pot_sum + X.bet_sum
X  = pd.concat([X,
    pd.get_dummies(X.roundName),
    pd.get_dummies(X.pos,prefix='pos',prefix_sep='='),
    pd.get_dummies(X.op_resp,prefix='op_resp',prefix_sep='='),
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
    pd.get_dummies(X.prev_action,prefix='prev',prefix_sep='='),
    pd.get_dummies(X.action,prefix='action',prefix_sep='='),
    ],1)
X['amount_P']  = X.amount / P
X['amount_SB'] = X.amount / X.smallBlind
X.drop(['smallBlind','roundName','chips','position','board','pot','bet','pot_sum','bet_sum','maxBet','prev_action','pos','op_resp',
    'cards',
    'action','amount',
    ],'columns',inplace=True)

rf  = RandomForestClassifier(n_estimators=100,max_depth=None,min_samples_leaf=4,oob_score=False,n_jobs=1,random_state=0,verbose=2,warm_start=False,class_weight=None)
lr  = LogisticRegression(penalty='l2',dual=False,tol=0.0001,C=1.0,fit_intercept=True,intercept_scaling=1,class_weight=None,random_state=0,solver='liblinear',max_iter=100,multi_class='ovr',verbose=0,warm_start=False,n_jobs=1)
model = lr

model.fit(X,y.winMoney>0)#y.action=='fold')
yhat  = model.predict(X)
accuracy_score(y.winMoney>0,yhat)# y.action=='fold',yhat)

kf  = StratifiedKFold(n_splits=4,shuffle=True,random_state=0)
results_tr  = pd.DataFrame(columns=('acc','f1'))
results_tt  = pd.DataFrame(columns=('acc','f1'))
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
    results_tr.loc[i,'acc'] = accuracy_score(y_tr.winMoney>0,yhat_tr)#.action=='fold',yhat_tr)
    results_tr.loc[i,'f1']  = f1_score(y_tr.winMoney>0,yhat_tr)#.action=='fold',yhat_tr)
    results_tr.loc[i,'precision'] = precision_score(y_tr.winMoney>0,yhat_tr)#.action=='fold',yhat_tr)
    results_tr.loc[i,'recall']    = recall_score(y_tr.winMoney>0,yhat_tr)#.action=='fold',yhat_tr)
    results_tt.loc[i,'acc'] = accuracy_score(y_tt.winMoney>0,yhat_tt)#.action=='fold',yhat_tt)
    results_tt.loc[i,'f1']  = f1_score(y_tt.winMoney>0,yhat_tt)#.action=='fold',yhat_tt)
    results_tt.loc[i,'precision'] = precision_score(y_tt.winMoney>0,yhat_tt)#.action=='fold',yhat_tt)
    results_tt.loc[i,'recall']    = recall_score(y_tt.winMoney>0,yhat_tt)#.action=='fold',yhat_tt)
    #
    if isinstance(model,RandomForestClassifier):
        feat_rank.append(pd.Series(model.feature_importances_,index=X_tr.columns))
    elif isinstance(model,LogisticRegression):
        feat_rank.append(pd.Series(np.r_[model.intercept_,model.coef_[0,:]],index=['intercept_']+X_tr.columns.tolist()))

results  = pd.concat([results_tr,results_tt],1,keys=('tr','tt'))
feat_rank = pd.concat(feat_rank,1).mean(1)
print(feat_rank.sort_values(ascending=False))
print(results.mean(0))
confusion_matrix(y_tt.winMoney>0,yhat_tt)#.action=='fold',yhat_tt)#,labels=['fold','check/call','bet/raise/allin'])


# X:
# 'smallBlind','chips','position','board','pot','bet','N','Nnf','Nallin','pot_sum','bet_sum','maxBet','NMaxBet','NRfold','NRcheck','NRcall','NRbet','NRraise','NRallincall','NRallin','score0','score1'
# y:
# ,'action','amount'
