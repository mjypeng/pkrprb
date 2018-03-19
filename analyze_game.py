from agent_common import *
import os,glob

filelist  = glob.glob('opponent_data' + os.sep + '*.csv')
results   = pd.concat([pd.read_csv(filename) for filename in filelist],ignore_index=True)

results   = results.merge(results.groupby(['game_id','round_id'])[['playerName']].nunique(),how='left',left_on=['game_id','round_id'],right_index=True,suffixes=('','_temp'),copy=False).rename(columns={'playerName_temp':'N'})

results['amt_pot'] = results.amount / results.pot

mask  = results.action=='allin'
results.loc[mask,'action'] = np.where(results.loc[mask,'amount']>results.loc[mask,'cost_to_call'],'bet/raise','check/call')
results.loc[results.action.isin(('bet','raise')),'action']  = 'bet/raise'
results.loc[results.action.isin(('check','call')),'action'] = 'check/call'

results['util_call']  = results.prWin*(results.pot + 2*results.cost_to_call) - results.cost_to_call
results['util_raise'] = 2*results.prWin - 1

results[results.roundName=='Deal'].groupby(['playerName','roundName','N','action']).agg({'prWin':['count','mean'],'util_call':'mean','util_raise':'mean','amt_pot':'mean'})
