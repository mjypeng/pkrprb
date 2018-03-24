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

pot  = results.pot_sum + results.bet_sum
results['chips_mpot']   = results.chips / pot
results['bet_sum_mpot'] = results.bet_sum / pot
results['cost_to_call_mpot'] = results.cost_to_call / pot
results['amount_mpot']  = results.amount / pot
results['util_final_mpot']   = results.profit / pot


deal  = results[results.roundName=='Deal']
flop  = results[results.roundName=='Flop']
turn  = results[results.roundName=='Turn']
river = results[results.roundName=='River']

exit(0)

temp  = pd.pivot_table(results[results.batch=='preliminary'],values='amount',index='action',columns='roundName',aggfunc='count')
temp /= temp.sum(0)




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




