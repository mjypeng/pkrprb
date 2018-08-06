from common import *
from expert_rules import *
from decision_logic import *
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
DATE_START  = '20180716'
DATE_END    = '20180803'
dt_range    = pd.date_range(DATE_START,DATE_END,freq='B').strftime('%Y%m%d')
rnd     = pd.concat([pd.read_csv('data/round_log_'+dt+'.gz') for dt in dt_range],0,ignore_index=True,sort=False)
rnd['timestamp']  = pd.to_datetime(rnd.timestamp)
action  = pd.concat([pd.read_csv('data/action_proc_'+dt+'.gz') for dt in dt_range],0,ignore_index=True,sort=False)
action['timestamp']  = pd.to_datetime(action.timestamp)

t0  = time.clock()
action['opponents']  = action.opponents.apply(eval)
action['op_chips'] = action.opponents.apply(lambda x:[y['chips'] for y in x])
print(time.clock() - t0)

action['op_chips_max']  = action.op_chips.apply(lambda x:max(x) if len(x) else 0)
action['op_chips_min']  = action.op_chips.apply(lambda x:min(x) if len(x) else 0)
action['op_chips_mean'] = (action.op_chips.apply(lambda x:sum(x) if len(x) else 0) / (action.Nnf - 1)).fillna(0)

#-- Hand Score --#
action['hand_score']  = action.hand_score.apply(eval)
action['hand_score0'] = action.hand_score.str[0]
action['hand_score1'] = action.hand_score.str[1]
action['hand_score2'] = action.hand_score.str[2].fillna(0).astype(int)

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

mask = (action.roundName=='Deal') & (action.action=='bet/raise/allin')
feat = ['N','pos','prWin','cards','hand','board','Naction','op_resp','minBet','op_chips','prev','action',] #
X    = compile_features_batch(action[mask],feat)
y    = action.loc[mask,'steal'].copy()

#-- Cross Validation --#
kf  = StratifiedKFold(n_splits=4,shuffle=True,random_state=0)
results,feat_rank  = cross_validation(kf,model,X,y)

results  = pd.concat(results.mean(0),1)
print(results.round(2))
results.round(2).to_clipboard('\t')










raises  = action[(action.roundName=='Deal')&(action.action=='bet/raise/allin')].copy()
raises['allin']  = raises.amount==raises.chips
raises['op_chips_max']  /= raises.chips
raises[raises.allin].groupby('steal').op_chips_max.mean()


action[action.roundName=='Deal'].groupby(['position']).steal.mean()


#-- Quality of Hands Received --#
temp  = rnd[rnd.timestamp.dt.strftime('%Y%m%d')=='20180803'].groupby('playerName')['rank'].agg(['count','mean','std'])
results  = []
for dt in ('20180730','20180731','20180801','20180802','20180803',):
    results.append(rnd[rnd.timestamp.dt.strftime('%Y%m%d')==dt].groupby('playerName')['rank'].mean())

results  = pd.concat(results,1,keys=('20180730','20180731','20180801','20180802','20180803',))



temp  = action[(action.timestamp.dt.strftime('%Y%m%d')>'20180730')&(action.roundName=='Deal')&(action.action=='bet/raise/allin')].groupby(['playerName',]).cards_category.agg(['count','mean','std','min','max']).sort_values('mean')
temp[temp['count']>=50]

def simulate_rules(state,rules):
    results     = []
    results_act = []
    for i,rule in rules:
        temp   = state.apply(rule,axis=1)
        state['play']    = temp.str[0]
        state['bet_amt'] = temp.str[1]
        state['result']  = np.where(state.play=='fold',-state.bet,
            np.where(state.play.isin(('raise','reraise'))&state.win,
                state.pot_sum + state.bet_sum,
                np.where(state.winRiver,
                    state.pot_sum + state.bet_sum + state.bet_amt,
                    -state.bet - state.bet_amt)))
        results_act.append(state.groupby('play')[['win','winRiver','result']].agg(['count','sum','mean','std']).iloc[:,[0,1,2,5,6,9,10,11]])
        print(results_act[-1])
        results.append(state.result.apply(['count','sum','mean','std']))
    #
    results_act = pd.concat(results_act,0,keys=[x[0] for x in rules])
    results     = pd.concat(results,1,keys=[x[0] for x in rules]).T
    results['sharpe']  = results['mean']/results['std']
    return results,results_act

#-- Simulate stt_preflop_pairs Decision Logic --#
mask   = (action.roundName == 'Deal') & action.cards_pair & (action.cards_rank1<=10)
state  = action[mask].copy()
state.rename(columns={'pos':'position_feature'},inplace=True)
state['minBet']  = state.maxBet - state.bet
rules    = [(i,lambda x:stt_preflop_pairs(x,i)) for i in np.arange(0.5,3,0.1)]
results,results_act = simulate_rules(state,rules)
results.to_clipboard(sep='\t')

#-- Simulate stt_early_preflop Decision Logic --#
mask   = (action.roundName == 'Deal')
state  = action[mask].copy()
state.rename(columns={'pos':'position_feature'},inplace=True)
state['minBet']  = state.maxBet - state.bet
rules  = [((i,j),lambda x:stt_early_preflop(x,i,j)) for i in range(2,7) for j in range(i+1,7)]
results,results_act = simulate_rules(state,rules)
results.to_clipboard(sep='\t')



rules  = [('middle',stt_middle_preflop)]
results,results_act = simulate_rules(state,rules)
results.to_clipboard(sep='\t')



mask   = (action.roundName == 'Deal') & (action.game_phase == 'Late')
state  = action[mask].copy()
state.rename(columns={'pos':'position_feature'},inplace=True)
state['minBet']  = state.maxBet - state.bet
rules  = [((i,j),lambda x:stt_late_preflop_allin(x,raise_thd=i,call_thd=j)) for i in range(5,1,-1) for j in range(0,-4,-1)]
results,results_act = simulate_rules(state,rules)
results.to_clipboard(sep='\t')
