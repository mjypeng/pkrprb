from agent_common import *
import os,glob
from sklearn.metrics import log_loss

filelist  = pd.Series(glob.glob('game_records' + os.sep + '*' + os.sep + 'training_*.csv'))
results   = pd.concat([pd.concat([pd.read_csv(filename,index_col=('game_id','round_id','turn_id'))],keys=(filename.split(os.sep)[1],),names=['batch']) for filename in filelist])
results.reset_index(inplace=True)

temp    = results[results.turn_id==1].set_index(['game_id','round_id'])[['bet_sum']]/3
results = results.merge(temp,how='left',left_on=['game_id','round_id'],right_index=True,suffixes=('','_temp'),copy=False).rename(columns={'bet_sum_temp':'smallBlind'})
results['profit_mbb'] = 1000*results.profit/(2*results.smallBlind)
results['amount_mbb'] = 1000*results.amount/(2*results.smallBlind)
temp    = results.groupby(['game_id','round_id'])[['rank']].max()
results = results.merge(temp,how='left',left_on=['game_id','round_id'],right_index=True,suffixes=('','_temp'),copy=False).rename(columns={'rank_temp':'max_rank'})
results['winHand'] = results['rank']==results.max_rank

results.loc[(results.action=='call')&(results.amount==0)&(results.cost_to_call>0),'amount']  = results.loc[(results.action=='call')&(results.amount==0)&(results.cost_to_call>0),'cost_to_call']
results.loc[(results.action=='call')&(results.amount==0)&(results.cost_to_call==0),'action'] = 'check'
results.loc[results.action=='raise','action'] = 'bet'
results.loc[(results.action=='allin')&(results.chips<=results.cost_to_call),'action'] = 'call'

pot  = results.pot_sum + results.bet_sum
results['chips_mpot']   = results.chips / pot
results['bet_sum_mpot'] = results.bet_sum / pot
results['cost_to_call_mpot'] = results.cost_to_call / pot
results['amount_mpot']  = results.amount / pot
results['util_final_mpot']   = results.profit / pot
results['last_bet_mpot']  = (10*results.cost_to_call_mpot/(1-results.cost_to_call_mpot)).round()/10

deal  = results[results.roundName=='Deal']
flop  = results[results.roundName=='Flop']
turn  = results[results.roundName=='Turn']
river = results[results.roundName=='River']

exit(0)

bet_mpot  = np.array([0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.75,1,2,5,10])
next_call_mpot = bet_mpot/(1+bet_mpot)

pd.pivot_table(results[results.roundName=='Deal'],values='turn_id',index='last_bet_mpot',columns='action',aggfunc='count')


#-- Unconditional Action Probabilities --#
actprob    = []
decisions  = ('Check or Bet','Fold, Call or Raise')
roundNames = ('Deal','Flop','Turn','River')
for roundName in roundNames:
    mask0  = results.batch=='preliminary'
    mask1  = results.roundName==roundName
    rr     = []
    for decision in decisions:
        if decision == 'Check or Bet':
            mask2  = results.cost_to_call==0
        else:
            mask2  = results.cost_to_call>0
        rrr   = results[mask1&mask2].groupby('action').amount.count()
        rrr  /= rrr.sum()
        rrr_  = results[mask0&mask1&mask2].groupby('action').amount.count()
        rrr_ /= rrr_.sum()
        rrr   = (rrr.fillna(0) + rrr_.fillna(0))/2
        rr.append(rrr)
    rr     = pd.concat(rr,0,keys=decisions,names=['decision'])
    actprob.append(rr)

actprob  = pd.concat(actprob,0,keys=roundNames,names=['roundName']).fillna(0)
actprob.name  = 'prob'
pd.DataFrame(actprob).to_csv('prior_action_prob.csv',encoding='utf-8-sig')

#-- Weighted (Check or Bet) --#
# roundName      Deal      Flop      Turn     River
# action                                           
# allin           NaN  0.028960  0.038074  0.056872
# bet        0.198950  0.465680  0.639734  0.588925
# check      0.668109  0.256587  0.198569  0.203728
# fold       0.127596  0.248774  0.123624  0.150475

#-- Weighted (Fold, Call or Raise) --#
# roundName      Deal      Flop      Turn     River
# action                                           
# allin      0.004373  0.014884  0.026917  0.042864
# bet        0.086168  0.115709  0.135628  0.144023
# call       0.554475  0.536389  0.622873  0.637595
# fold       0.354983  0.333018  0.214582  0.175518

#-- prWin Distribution --#
N_range    = np.sort(results.N.unique())
roundNames = ('Deal','Flop','Turn','River')
temp  = results.groupby(['N','roundName']).prWin.quantile(np.arange(0,1.01,0.1))
temp  = pd.concat([pd.concat([temp.loc[(N,roundName)] for roundName in roundNames],1,keys=roundNames) for N in N_range],1,keys=N_range).T
temp.index.names = ('N','roundName')
temp.columns = (100*temp.columns.values).astype(int)
temp.to_csv('prWin_distribution.csv',encoding='utf-8-sig')

w         = temp.loc[temp.index.get_level_values('roundName')=='Deal',100]
wstd      = np.sqrt(w*(1-w))
tightness = (0.9 - w)/wstd
tightness.diff()[-1]

#-- prWin Distribution (N=10) --#
#          Deal      Flop      Turn     River
# 0.0  0.048172  0.001626  0.000000  0.000000
# 0.1  0.058843  0.022072  0.009189  0.000000
# 0.2  0.067704  0.035185  0.027423  0.000000
# 0.3  0.076698  0.046410  0.036315  0.000000
# 0.4  0.082383  0.056452  0.050499  0.001658
# 0.5  0.092190  0.075638  0.070091  0.016514
# 0.6  0.098230  0.091796  0.093023  0.055901
# 0.7  0.110591  0.113591  0.110766  0.068224
# 0.8  0.130101  0.155694  0.147047  0.114035
# 0.9  0.144585  0.194261  0.254160  0.195216
# 1.0  0.311741  0.751462  0.816568  0.989691

#-- prWin Distribution (N=10) --#



# Observed freq bet sizes in preliminary: mpot = 0.1, 0.15, 0.2, 0.3, 0.4
# Observed freq bet sizes overall: mpot = 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5
# Corresponding prWinThd: 1/12~=8%, 3/26~=12%, 1/7~=14%, 1/6~=17%, 3/16~=19%, 2/9~=22%, 1/4~=25%

results   = results.merge(results.groupby(['game_id','round_id'])[['playerName']].nunique(),how='left',left_on=['game_id','round_id'],right_index=True,suffixes=('','_temp'),copy=False).rename(columns={'playerName_temp':'N'})

results['amt_pot'] = results.amount / results.pot

mask  = results.action=='allin'
results.loc[mask,'action'] = np.where(results.loc[mask,'amount']>results.loc[mask,'cost_to_call'],'bet/raise','check/call')
results.loc[results.action.isin(('bet','raise')),'action']  = 'bet/raise'
results.loc[results.action.isin(('check','call')),'action'] = 'check/call'

results['util_call']  = results.prWin*(results.pot + 2*results.cost_to_call) - results.cost_to_call
results['util_raise'] = 2*results.prWin - 1

results[results.roundName=='Deal'].groupby(['playerName','roundName','N','action']).agg({'prWin':['count','mean'],'util_call':'mean','util_raise':'mean','amt_pot':'mean'})




