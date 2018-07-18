import os,sys,glob,time
import pandas as pd
import numpy as np
import json

logs   = []
events = []
for f in glob.glob('user_logs_20180716.log/*'):
    log  = pd.read_csv(f,sep='>>>',header=None,names=(0,1,2),engine='python')
    log.dropna(subset=[2],inplace=True)
    if len(log) > 0:
        event = log[log[1].str.split().str[0]=='table'].copy()
        event = event[event[2].str.strip().str.startswith('new round')|event[2].str.strip().str.startswith('game over')]
        event['timestamp']   = event[0].str.split(n=1).str[0].str.strip().str[1:-1]
        event['tableNumber'] = event[1].str.split().str[-1].astype(int)
        events.append(event)
        log  = log[log[1].str.split().str[0]=='event']
        log['timestamp']  = log[0].str.split(n=1).str[0].str.strip().str[1:-1]
        logs.append(log)

logs  = pd.concat(logs,0,ignore_index=True)
logs.sort_values('timestamp',ascending=True,inplace=True)
events = pd.concat(events,0,ignore_index=True)
events.sort_values('timestamp',ascending=True,inplace=True)

TABLES  = {}
for i,row in events.iterrows():
    if row.tableNumber not in TABLES or row[2].str.strip().str.startswith('new round : 1'):
        TABLES[row.tableNumber]  = {
            'GAMES': {},
            }




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

action  = logs[logs.eventName=='__show_action'].drop('winners','columns')
for col in ('playerName','action','amount'):
    action[col]  = action.action0.apply(lambda x:x[col] if col in x else None)

action.drop('action0','columns',inplace=True)

def get_player_info(x):
    for y in x.players:
        if y['playerName'] == x.playerName:
            return pd.Series(y)

t0  = time.clock()
temp  = action[['playerName','players']].apply(get_player_info,axis=1)
time.clock() - t0
for col in ('chips','folded','allIn','cards','isSurvive','reloadCount','roundBet','bet','isOnline','isHuman'):
    action[col]  = temp[col]

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

rnd  = logs[logs.eventName=='__round_end'].drop(['action0','winners'],'columns')
rnd  = pd.concat([
    pd.DataFrame(np.repeat(rnd.drop('players','columns').values,rnd.players.str.len(),axis=0),columns=rnd.drop('players','columns').columns),
    pd.DataFrame([y for x in rnd.players.tolist() for y in x])[['playerName','chips','folded','allIn','cards','isSurvive','reloadCount','roundBet','bet','isOnline','isHuman','hand','winMoney']],
    ],1)
for col in ('rank','message'):
    rnd[col]  = rnd.hand.apply(lambda x:x[col] if pd.notnull(x) else None)

rnd.drop('hand','columns',inplace=True)

[x for _,x in zip(rnd.drop('players','columns').iterrows(),rnd.players.str.len())]


game    = logs[logs.eventName=='__game_over'].drop('action0','columns')




logs['winners']    = logs[2].apply(lambda x:x['data']['winners'] if 'winners' in x['data'] else None)

# if 'eventName' == '__show_action': (players, table, action)
# if 'eventName' == '__round_end':   (players, table)
# if 'eventName' == '__game_over':   (players, table, winners)
