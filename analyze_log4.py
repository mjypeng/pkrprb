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
pd.set_option('display.width',120)

#-- Read Data --#
DATE_START  = '20180716'
DATE_END    = '20180803'
dt_range    = pd.date_range(DATE_START,DATE_END,freq='B').strftime('%Y%m%d')
rnd     = pd.concat([pd.read_csv('data/round_log_'+dt+'.gz') for dt in dt_range],0,ignore_index=True,sort=False)
action  = pd.concat([pd.read_csv('data/action_proc_'+dt+'.gz') for dt in dt_range],0,ignore_index=True,sort=False)
action['timestamp']  = pd.to_datetime(action.timestamp)

t0  = time.clock()
action['opponents']  = action.opponents.apply(eval)
action['op_chips'] = action.opponents.apply(lambda x:[y['chips'] for y in x])
print(time.clock() - t0)

action['op_chips_max']  = action.op_chips.apply(lambda x:max(x) if len(x) else 0)
action['op_chips_min']  = action.op_chips.apply(lambda x:min(x) if len(x) else 0)
action['op_chips_mean'] = (action.op_chips.apply(lambda x:sum(x) if len(x) else 0) / (action.Nnf - 1)).fillna(0)

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
