import os,sys,glob,time
import pandas as pd
import numpy as np
import json

# pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)

def get_player_info(x):
    for y in x.players:
        if y['playerName'] == x.playerName:
            return pd.Series(y)

def agg_player_info(x):
    x  = pd.DataFrame(x)
    y  = pd.Series()
    y['N']   = x.isSurvive.sum()
    x  = x[x.isSurvive]
    y['Nnf'] = y.N - x.folded.sum()
    y['Nallin'] = x.allIn.sum()
    y['pot_sum'] = x.roundBet.sum()
    y['bet_sum'] = x.bet.sum()
    y['maxBet']  = x.bet.max()
    y['NMaxBet'] = ((x.bet > 0) & ~x.folded & (x.allIn | (x.bet==y.maxBet))).sum()
    return y

dt     = sys.argv[1] #'20180716'
logs   = []
events = []
for f in glob.glob('user_logs_'+dt+'.log/*'):
    log  = pd.read_csv(f,sep='>>>',header=None,names=(0,1,2),engine='python')
    log.dropna(subset=[2],inplace=True)
    if len(log) > 0:
        event = log[log[1].str.split().str[0]=='table'].copy()
        event = event[event[2].str.strip().str.startswith('new round')]
        event['timestamp']   = event[0].str.split(n=1).str[0].str.strip().str[1:-1]
        event['tableNumber'] = event[1].str.split().str[-1].astype(int)
        events.append(event)
        log  = log[log[1].str.split().str[0]=='event']
        log['timestamp']  = log[0].str.split(n=1).str[0].str.strip().str[1:-1]
        logs.append(log)

logs  = pd.concat(logs,0,ignore_index=True)
logs.sort_values('timestamp',ascending=True,inplace=True)
events = pd.concat(events,0,ignore_index=True)
events['roundCount'] = events[2].str.rsplit(n=1).str[-1].astype(int)
events.drop([0,1,2],'columns',inplace=True)
events.sort_values(['tableNumber','timestamp','roundCount'],ascending=True,inplace=True)
events.rename(columns={'timestamp':'start_ts'},inplace=True)
events['start_ts']  = pd.to_datetime(events.start_ts)

logs[2]  = logs[2].apply(json.loads)
temp     = pd.DataFrame(logs[2].tolist(),index=logs.index)
logs.drop([0,1,2],'columns',inplace=True)
logs['eventName']  = temp.eventName

temp  = pd.DataFrame(temp.data.tolist(),index=logs.index)
for col in ('tableNumber','status','roundName','board','roundCount','raiseCount','betCount','totalBet','initChips','maxReloadCount',):
    logs[col]  = temp.table.apply(lambda x:x[col])

for col in ('smallBlind','bigBlind',):
    for col2 in ('playerName','amount',):
        logs[col+'_'+col2]  = temp.table.apply(lambda x:x[col][col2])

logs['players']    = temp.players
logs['action0']    = temp.action
logs['winners']    = temp.winners

#------------------------#
#-- Compile Action Log --#
#------------------------#
action  = logs[logs.eventName=='__show_action'].drop('winners','columns')
for col in ('playerName','action','amount'):
    action[col]  = action.action0.apply(lambda x:x[col] if col in x else None)

action.drop('action0','columns',inplace=True)

t0  = time.clock()
temp  = action[['playerName','players']].apply(get_player_info,axis=1)
print(time.clock() - t0)
for col in ('chips','folded','allIn','cards','isSurvive','reloadCount','roundBet','bet','isOnline','isHuman'):
    action[col]  = temp[col]

t0  = time.clock()
temp  = action.players.apply(agg_player_info)
print(time.clock() - t0)

action  = pd.concat([action,temp],1)

action['timestamp'] = pd.to_datetime(action.timestamp)
action['board'] = action.board.str.join(' ')
action['cards'] = action.cards.str.join(' ')
action['position']  = np.where(action.smallBlind_playerName==action.playerName,'SB',np.where(action.bigBlind_playerName==action.playerName,'BB',''))
action.rename(columns={'smallBlind_amount':'smallBlind','bigBlind_amount':'bigBlind','roundBet':'pot'},inplace=True)

action  = action[['timestamp','tableNumber','roundCount','smallBlind','bigBlind','roundName','raiseCount','betCount','totalBet','playerName','isHuman','isOnline','chips','reloadCount','folded','allIn','position','cards','board','pot','bet','N','Nnf','Nallin','pot_sum','bet_sum','maxBet','NMaxBet','action','amount',]]
action.to_csv('action_log_'+dt+'.gz',index=False,compression='gzip')

#------------------------------#
#-- Compile Round Winner Log --#
#------------------------------#
rnd  = logs[logs.eventName=='__round_end'].drop(['action0','winners'],'columns')
rnd  = pd.concat([
    pd.DataFrame(np.repeat(rnd.drop('players','columns').values,rnd.players.str.len(),axis=0),columns=rnd.drop('players','columns').columns),
    pd.DataFrame([y for x in rnd.players.tolist() for y in x])[['playerName','chips','folded','allIn','cards','isSurvive','reloadCount','roundBet','bet','isOnline','isHuman','hand','winMoney']],
    ],1)
for col in ('rank','message'):
    rnd[col]  = rnd.hand.apply(lambda x:x[col] if pd.notnull(x) else None)

rnd.drop('hand','columns',inplace=True)

rnd['timestamp'] = pd.to_datetime(rnd.timestamp)
rnd['board'] = rnd.board.str.join(' ')
rnd['cards'] = rnd.cards.str.join(' ')
rnd['position']  = np.where(rnd.smallBlind_playerName==rnd.playerName,'SB',np.where(rnd.bigBlind_playerName==rnd.playerName,'BB',''))
rnd.rename(columns={'smallBlind_amount':'smallBlind','bigBlind_amount':'bigBlind','roundBet':'pot'},inplace=True)

rnd  = rnd[['timestamp','tableNumber','status','roundCount','raiseCount','betCount','totalBet','smallBlind','bigBlind','playerName','isHuman','isOnline','isSurvive','chips','reloadCount','folded','allIn','position','cards','board','pot','bet','rank','message','winMoney',]]
rnd.to_csv('round_log_'+dt+'.gz',index=False,compression='gzip')

#-----------------------------#
#-- Compile Game Winner Log --#
#-----------------------------#
game    = logs[logs.eventName=='__game_over'].drop(['players','action0'],'columns')
game    = pd.concat([
    pd.DataFrame(np.repeat(game.drop('winners','columns').values,game.winners.str.len(),axis=0),columns=game.drop('winners','columns').columns),
    pd.DataFrame([y for x in game.winners.tolist() for y in x])[['playerName','chips']],
    ],1)
game.rename(columns={'smallBlind_amount':'smallBlind','bigBlind_amount':'bigBlind'},inplace=True)
game  = game[['timestamp','tableNumber','roundCount','smallBlind','bigBlind','playerName','chips']]
game.to_csv('game_log_'+dt+'.gz',index=False,compression='gzip')

#--------------------#
#-- Segment Rounds --#
#--------------------#
evente  = rnd[['timestamp','tableNumber','roundCount']].drop_duplicates().sort_values(['tableNumber','timestamp','roundCount'],ascending=True).rename(columns={'timestamp':'end_ts'})
evente['end_ts']  = pd.to_datetime(evente.end_ts)

t0  = time.clock()
for idx,row in events.iterrows():
    ee  = evente[(evente.tableNumber==row.tableNumber)&(evente.roundCount==row.roundCount)]
    temp = ee.end_ts - row.start_ts
    temp = temp[temp>pd.to_timedelta(0)]
    if len(temp) > 0:
        events.loc[idx,'end_ts']  = ee.loc[temp.idxmin(),'end_ts']

print(time.clock() - t0)
events.drop_duplicates(subset=['tableNumber','end_ts','roundCount'],keep='last',inplace=True)
events['round_id']  = events.tableNumber.astype(str) + '_' + events.start_ts.dt.strftime('%Y%m%d%H%M%S%f').str[:-3] + '_' + events.roundCount.astype(str)

#-- Assign Round ID to Round Log --#
t0  = time.clock()
rnd = rnd.merge(events.drop('start_ts','columns'),how='left',left_on=['tableNumber','roundCount','timestamp'],right_on=['tableNumber','roundCount','end_ts'],copy=False)
print(time.clock() - t0)

rnd = rnd[['timestamp','tableNumber','status','roundCount','round_id','raiseCount','betCount','totalBet','smallBlind','bigBlind','playerName','isHuman','isOnline','isSurvive','chips','reloadCount','folded','allIn','position','cards','board','rank','message','winMoney']]
rnd.to_csv('round_log_'+dt+'.gz',index=False,compression='gzip')

#-- Assign Round ID to Action Log --#
t0  = time.clock()
for idx,row in events.iterrows():
    mask  = (action.tableNumber==row.tableNumber) & (action.roundCount==row.roundCount) & (action.timestamp>=row.start_ts) & (action.timestamp<=row.end_ts)
    action.loc[mask,'round_id']  = row.round_id

print(time.clock() - t0)

#-- Calculate Player Position for each Round --#
pos  = action[['timestamp','round_id','playerName','position']].dropna(subset=['round_id','playerName']).drop_duplicates(subset=['round_id','playerName'],keep='first').sort_values(['round_id','timestamp'])
i  = 1
for idx,row in pos.iterrows():
    if pd.isnull(row.position) or row.position=='':
        pos.loc[idx,'position'] = i
        i += 1
        previdx = idx
    elif row.position == 'SB':
        pos.loc[previdx,'position'] = 'D'
    elif row.position == 'BB':
        i = 1

pos.set_index(['round_id','playerName'],inplace=True)
pos.drop('timestamp','columns',inplace=True)

#-- Assign Player Position to Round Log --#
rnd  = rnd.merge(pos,how='left',left_on=['round_id','playerName'],right_index=True,suffixes=('','_temp'))
rnd['position'] = np.where(rnd.position.notnull(),rnd.position,rnd.position_temp)
rnd.drop('position_temp','columns',inplace=True)
rnd.to_csv('round_log_'+dt+'.gz',index=False,compression='gzip')

#-- Assign Player Position to Action Log --#
action = action.merge(pos,how='left',left_on=['round_id','playerName'],right_index=True,suffixes=('','_temp'))
action['position'] = np.where(action.position.notnull(),action.position,action.position_temp)
action.drop('position_temp','columns',inplace=True)

#-- Merge Round Results into Action Log --#
action = action.merge(rnd.set_index(['round_id','playerName'])[['rank','message','winMoney']],how='left',left_on=['round_id','playerName'],right_index=True,copy=False)

action = action[['timestamp','tableNumber','roundCount','round_id','smallBlind','bigBlind','roundName','raiseCount','betCount','totalBet','playerName','isHuman','isOnline','chips','reloadCount','folded','allIn','position','cards','board','pot','bet','N','Nnf','Nallin','pot_sum','bet_sum','maxBet','NMaxBet','action','amount','rank','message','winMoney']]
action.to_csv('action_log_'+dt+'.gz',index=False,compression='gzip')

# if 'eventName' == '__show_action': (players, table, action)
# if 'eventName' == '__round_end':   (players, table)
# if 'eventName' == '__game_over':   (players, table, winners)

# Action Current: ['timestamp', 'tableNumber', 'roundCount', 'round_id', 'smallBlind', 'bigBlind', 'roundName', 'raiseCount', 'betCount', 'totalBet', 'playerName', 'isHuman', 'isOnline', 'chips', 'reloadCount', 'folded', 'allIn', 'position', 'cards', 'board', 'pot', 'bet', 'N', 'Nnf', 'Nallin', 'pot_sum', 'bet_sum', 'maxBet', 'NMaxBet', 'action', 'amount', 'rank', 'message', 'winMoney']
# TODO: Revert state before action, 'Nsim', 'prWin', 'prWinStd', 'cost_to_call', 'cost', 'profit'
