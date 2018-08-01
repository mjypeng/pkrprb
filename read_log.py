import os,sys,glob,time
import pandas as pd
import numpy as np
import json

pd.set_option('display.max_rows',120)
pd.set_option('display.max_columns',None)
pd.set_option('display.width',90)

def get_player_info(x):
    for i,y in enumerate(x.players):
        if y['playerName'] == x.playerName:
            y  = pd.Series(y)
            y['player_idx']  = i
            return y

def agg_player_info(x):
    players  = [xx for xx in x.players if xx['isSurvive']]
    y  = pd.Series()
    y['N']       = len(players)
    y['Nnf']     = y.N - len([xx for xx in players if xx['folded']])
    y['Nallin']  = len([xx for xx in players if xx['allIn']])
    y['pot_sum'] = sum([xx['roundBet'] for xx in players])
    bets  = [xx['bet'] for xx in players]
    y['bet_sum'] = sum(bets)
    y['maxBet']  = max(bets)
    y['NMaxBet'] = len([xx for xx in players if xx['bet']>0 and not xx['folded'] and (xx['allIn'] or (xx['bet']==y.maxBet))])
    #
    y['opponents']  = json.dumps([{col:xx[col] for col in ('playerName','chips','cards','roundBet','bet',)} for xx in players if xx['playerName']!=x.playerName and not xx['folded']]).replace(' ','')
    return y

#-------------------#
#-- Read Raw Logs --#
#-------------------#
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
logs['timestamp']  = pd.to_datetime(logs.timestamp)
logs.sort_values('timestamp',ascending=True,inplace=True)

events = pd.concat(events,0,ignore_index=True)
events['timestamp']  = pd.to_datetime(events.timestamp)
events['roundCount'] = events[2].str.rsplit(n=1).str[-1].astype(int)
events.drop([0,1,2],'columns',inplace=True)
events.sort_values(['tableNumber','timestamp','roundCount'],ascending=True,inplace=True)
events.rename(columns={'timestamp':'start_ts'},inplace=True)

#-- Parse JSON --#
logs[2]  = logs[2].apply(json.loads)
temp     = pd.DataFrame(logs[2].tolist(),index=logs.index)
logs.drop([0,1,2],'columns',inplace=True)
logs['eventName']  = temp.eventName

#-- Parse Table Info --#
temp  = pd.DataFrame(temp.data.tolist(),index=logs.index)
for col in ('tableNumber','status','roundName','board','roundCount','raiseCount','betCount','totalBet','initChips','maxReloadCount',):
    logs[col]  = temp.table.apply(lambda x:x[col])

for col in ('smallBlind','bigBlind',):
    for col2 in ('playerName','amount',):
        logs[col+'_'+col2]  = temp.table.apply(lambda x:x[col][col2])

logs['players']    = temp.players
logs['action0']    = temp.action
logs['winners']    = temp.winners

#--------------------#
#-- Segment Rounds --#
#--------------------#
evente  = logs.loc[logs.eventName=='__round_end',['timestamp','tableNumber','roundCount']].drop_duplicates().sort_values(['tableNumber','timestamp','roundCount'],ascending=True).rename(columns={'timestamp':'end_ts'})
t0  = time.clock()
for idx,row in events.iterrows():
    ee  = evente[(evente.tableNumber==row.tableNumber)&(evente.roundCount==row.roundCount)]
    temp = ee.end_ts - row.start_ts
    temp = temp[temp>pd.to_timedelta(0)]
    if len(temp) > 0:
        events.loc[idx,'end_ts']  = ee.loc[temp.idxmin(),'end_ts']

print(time.clock() - t0)
events.dropna(subset=['end_ts'],inplace=True)
events.drop_duplicates(subset=['tableNumber','end_ts','roundCount'],keep='last',inplace=True)
events['round_id']  = events.tableNumber.astype(str) + '_' + events.start_ts.dt.strftime('%Y%m%d%H%M%S%f').str[:-3] + '_' + events.roundCount.astype(str)

#-------------------#
#-- Segment Games --#
#-------------------#
events2 = events.loc[events.roundCount==1,['tableNumber','start_ts','round_id']].copy()
events2['game_id']  = events2.round_id.str[:-2]
events2.drop('round_id','columns',inplace=True)
evente2 = logs.loc[logs.eventName=='__game_over',['timestamp','tableNumber']].drop_duplicates().sort_values(['tableNumber','timestamp'],ascending=True).rename(columns={'timestamp':'end_ts'})
t0  = time.clock()
for idx,row in events2.iterrows():
    ee  = evente2[(evente2.tableNumber==row.tableNumber)]
    temp = ee.end_ts - row.start_ts
    temp = temp[temp>pd.to_timedelta(0)]
    if len(temp) > 0:
        events2.loc[idx,'end_ts']  = ee.loc[temp.idxmin(),'end_ts']

print(time.clock() - t0)
events2.dropna(subset=['end_ts'],inplace=True)
events2.drop_duplicates(subset=['tableNumber','end_ts'],keep='last',inplace=True)

for idx,row in events2.iterrows():
    mask  = (events.tableNumber==row.tableNumber) & (events.start_ts>=row.start_ts) & (events.start_ts<=row.end_ts)
    events.loc[mask,'game_id']  = row.game_id

#-----------------------------#
#-- Compile Game Winner Log --#
#-----------------------------#
game    = logs.loc[logs.eventName=='__game_over',['timestamp','tableNumber','roundCount','smallBlind_amount','players']].copy()
game    = pd.concat([
    pd.DataFrame(np.repeat(game.drop('players','columns').values,game.players.str.len(),axis=0),columns=game.drop('players','columns').columns),
    pd.DataFrame([y for x in game.players.tolist() for y in x])[['playerName','chips']],
    ],1)
game['timestamp']   = pd.to_datetime(game.timestamp)
game['tableNumber'] = game.tableNumber.astype(int)
game.rename(columns={'smallBlind_amount':'smallBlind',},inplace=True)

#-- Assign Game ID to Game Log --#
t0  = time.clock()
game = game.merge(events[['tableNumber','game_id','end_ts']],how='left',left_on=['tableNumber','timestamp'],right_on=['tableNumber','end_ts'],copy=False)
print(time.clock() - t0)

game  = game[['timestamp','tableNumber','game_id','roundCount','smallBlind','playerName','chips']]
game.to_csv('game_log_'+dt+'.gz',index=False,compression='gzip')

#------------------------------#
#-- Compile Round Winner Log --#
#------------------------------#
rnd  = logs[logs.eventName=='__round_end'].drop(['eventName','roundName','initChips','maxReloadCount','bigBlind_amount','raiseCount','betCount','totalBet','action0','winners'],'columns').copy()
rnd  = pd.concat([
    pd.DataFrame(np.repeat(rnd.drop('players','columns').values,rnd.players.str.len(),axis=0),columns=rnd.drop('players','columns').columns),
    pd.DataFrame([y for x in rnd.players.tolist() for y in x])[['playerName','isHuman','isOnline','isSurvive','chips','reloadCount','folded','allIn','cards','hand','winMoney']],
    ],1)
rnd['timestamp']    = pd.to_datetime(rnd.timestamp)
rnd['tableNumber']  = rnd.tableNumber.astype(int)
rnd['roundCount']   = rnd.roundCount.astype(int)

for col in ('rank','message'):
    rnd[col]  = rnd.hand.apply(lambda x:x[col] if pd.notnull(x) else None)

rnd['board'] = rnd.board.str.join(' ')
rnd['cards'] = rnd.cards.str.join(' ')
rnd['position']  = np.where(rnd.smallBlind_playerName==rnd.playerName,'SB',np.where(rnd.bigBlind_playerName==rnd.playerName,'BB',None))
rnd.rename(columns={'smallBlind_amount':'smallBlind',},inplace=True)

#-- Assign Game ID and Round ID to Round Log --#
t0  = time.clock()
rnd = rnd.merge(events.drop('start_ts','columns'),how='left',left_on=['tableNumber','roundCount','timestamp'],right_on=['tableNumber','roundCount','end_ts'],copy=False)
print(time.clock() - t0)

rnd = rnd[['timestamp','tableNumber','status','roundCount','game_id','round_id','smallBlind','playerName','isHuman','isOnline','isSurvive','chips','reloadCount','folded','allIn','position','cards','board','rank','message','winMoney',]]
rnd.to_csv('round_log_'+dt+'.gz',index=False,compression='gzip')

#------------------------#
#-- Compile Action Log --#
#------------------------#
action  = logs[logs.eventName=='__show_action'].drop(['eventName','initChips','maxReloadCount','bigBlind_amount','winners'],'columns').copy()
for col in ('playerName','action','amount'):
    action[col]  = action.action0.apply(lambda x:x[col] if col in x else None)

action.drop('action0','columns',inplace=True)
action['amount']  = action.amount.fillna(0).astype(int)

#-- Extract Action Player Info --#
t0  = time.clock()
temp  = action[['players','playerName']].apply(get_player_info,axis=1)
print(time.clock() - t0)
for col in ('player_idx','isHuman','isOnline','chips','reloadCount','cards','roundBet','bet',):
    action[col]  = temp[col]

action['board'] = action.board.str.join(' ')
action['cards'] = action.cards.str.join(' ')

#-- Revert Player State before Action --#
# 'raiseCount' is before action, but 'betCount' and 'totalBet' are after and need to be reverted
action.loc[action.action=='bet','betCount']  -= 1
action['totalBet'] -= action.amount
action['chips'] += action.amount
action['bet']   -= action.amount
t0  = time.clock()
for idx,row in action.iterrows():
    y  = row.players[row.player_idx]
    y['chips']  += row.amount
    y['folded']  = False
    y['allIn']   = False
    y['bet']    -= row.amount

print(time.clock() - t0)

#-- Aggregate Player Info --#
t0  = time.clock()
temp  = action[['players','playerName']].apply(agg_player_info,axis=1)
print(time.clock() - t0)
action  = pd.concat([action.drop(['players','player_idx'],'columns'),temp],1)

action['position']  = np.where(action.smallBlind_playerName==action.playerName,'SB',np.where(action.bigBlind_playerName==action.playerName,'BB',None))
action.rename(columns={'smallBlind_amount':'smallBlind','roundBet':'pot'},inplace=True)

#-- Assign Game ID and Round ID to Action Log --#
t0  = time.clock()
for idx,row in events.iterrows():
    mask  = (action.tableNumber==row.tableNumber) & (action.roundCount==row.roundCount) & (action.timestamp>=row.start_ts) & (action.timestamp<=row.end_ts)
    action.loc[mask,'game_id']   = row.game_id
    action.loc[mask,'round_id']  = row.round_id

print(time.clock() - t0)

#-- Merge Round Results into Action Log --#
action = action.merge(rnd.dropna(subset=['round_id']).set_index(['round_id','playerName'])[['rank','message','winMoney']],how='left',left_on=['round_id','playerName'],right_index=True,copy=False)

#-- Merge Game Results into Action Log --#
action = action.merge(game.dropna(subset=['game_id']).set_index(['game_id','playerName'])[['chips']],how='left',left_on=['game_id','playerName'],right_index=True,suffixes=('','_final'),copy=False)

action  = action[['timestamp','tableNumber','roundCount','game_id','round_id','smallBlind','roundName','raiseCount','betCount','totalBet','playerName','isHuman','isOnline','chips','reloadCount','position','cards','board','pot','bet','N','Nnf','Nallin','pot_sum','bet_sum','maxBet','NMaxBet','op_chips_max','op_chips_min','action','amount','rank','message','winMoney','chips_final']]
action.to_csv('action_log_'+dt+'.gz',index=False,compression='gzip')

#-------------------------------------------#
#-- Derive Player Position for each Round --#
#-------------------------------------------#
pos  = action[['timestamp','round_id','playerName','position']].dropna(subset=['round_id','playerName']).drop_duplicates(subset=['round_id','playerName'],keep='first').sort_values(['round_id','timestamp'])
pos.loc[~pos.position.isin(('SB','BB')),'position'] = np.nan
t0 = time.clock()
cur_round_id = None
for idx,row in pos.iterrows():
    if cur_round_id is None or cur_round_id != row.round_id:
        i  = 1
        cur_round_id  = row.round_id
    if pd.isnull(row.position) or row.position=='':
        pos.loc[idx,'position'] = i
        D_idx = idx
        i    += 1
    elif row.position in ('SB','BB'):
        pos.loc[D_idx,'position'] = 'D'

print(time.clock() - t0)
pos.set_index(['round_id','playerName'],inplace=True)
pos.drop('timestamp','columns',inplace=True)

#-----------------------------------------#
#-- Assign Player Position to Round Log --#
#-----------------------------------------#
rnd  = rnd.merge(pos,how='left',left_on=['round_id','playerName'],right_index=True,suffixes=('','_temp'))
rnd['position'] = np.where(rnd.position.notnull(),rnd.position,rnd.position_temp)
rnd.drop('position_temp','columns',inplace=True)
rnd.to_csv('round_log_'+dt+'.gz',index=False,compression='gzip')

#------------------------------------------#
#-- Assign Player Position to Action Log --#
#------------------------------------------#
action = action.merge(pos,how='left',left_on=['round_id','playerName'],right_index=True,suffixes=('','_temp'))
action['position'] = np.where(action.position.notnull(),action.position,action.position_temp)
action.drop('position_temp','columns',inplace=True)
action.to_csv('action_log_'+dt+'.gz',index=False,compression='gzip')
