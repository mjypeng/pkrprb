from common import *
from datetime import datetime
import player6 as pp

pd.set_option('display.max.rows',None)
pd.set_option('display.max_columns',None)
pd.set_option('display.width',90)
pd.set_option('display.unicode.east_asian_width',False)

pp.TABLE_STATE['name']     = 'p0'
pp.TABLE_STATE['name_md5'] = 'p0'

table    = {
    'tableNumber':    0,
    'roundName':      'Deal', #'Flop', #
    'board':          [], #['AD','5C','2D'], #
    'smallBlind':{
        'playerName': 'p0',
        'amount':     10,
        },
    'bigBlind':{
        'playerName': 'p1',
        'amount':     20,
        },
    }
players  = pd.DataFrame({'playerName':['p0','p1','p2','p3','p4',],'isHuman':False,'isOnline':True,'isSurvive':True,'chips':3000,'reloadCount':0,'cards':'','allIn':False,'folded':False,'roundBet':0,'bet':0})
players.loc[players.playerName==pp.TABLE_STATE['name_md5'],'cards'] = 'AS AH'
players['cards']  = players.cards.str.split()
players.loc[players.playerName==table['smallBlind']['playerName'],'bet']  = table['smallBlind']['amount']
players.loc[players.playerName==table['bigBlind']['playerName'],'bet']  = table['bigBlind']['amount']

#-- __game_start --#
event_name,data  = '__game_start',{'tableNumber':table['tableNumber']}
pp.TABLE_STATE['tableNumber']  = data['tableNumber']
pp.TABLE_STATE['game_id']  = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
pp.TABLE_STATE['round_id'] = 0
pp.GAME_STATE    = None
pp.ROUND_RESULTS = []
pp.GAME_ACTIONS  = None

#-- __new_round --#
event_name,data  = '__new_round',{'table':table.copy(),'players':[x.copy() for _,x in players.iterrows()]}
pp.TABLE_STATE['round_id'] += 1
pp.init_game_state(data['players'],data['table'])
if data['table']['smallBlind']['playerName']==pp.TABLE_STATE['name_md5']:
    pp.TABLE_STATE['forced_bet']  = data['table']['smallBlind']['amount']
elif data['table']['bigBlind']['playerName']==pp.TABLE_STATE['name_md5']:
    pp.TABLE_STATE['forced_bet']  = data['table']['bigBlind']['amount']
else:
    pp.TABLE_STATE['forced_bet']  = 0

pp.PREV_AGENT_STATE  = None

#-- __action --#
event_name,data  = '__action',{'game':table.copy(),}
data['self']             = players[players.playerName==pp.TABLE_STATE['name_md5']].iloc[0]
data['self']['minBet']   = 0
data['game']['players']  = [x.copy() for _,x in players.iterrows()]
t0    = time.time()
if pp.GAME_STATE is None:
    pp.init_game_state(data['game']['players'],data['game'])
else:
    pp.update_game_state(data['game']['players'],data['game'])

out   = pp.agent(event_name,data)
resp  = out[0]
log   = out[1]

log['cputime']  = time.time() - t0
pp.AGENT_LOG.append(log)
pp.TABLE_STATE['forced_bet']  = 0
pp.PREV_AGENT_STATE  = log
print(pp.PREV_AGENT_STATE)
print()

#-- __deal --#
table['roundName']  = 'Flop'
table['board']      = ['AD','5C','2D']
event_name,data  = '__deal',{'table':table.copy(),'players':[x.copy() for _,x in players.iterrows()]}

# New card is dealt, i.e. a new betting round
if pp.GAME_STATE is None:
    pp.init_game_state(data['players'],data['table'])
pp.update_game_state(data['players'],data['table'],'reset_action_count')

#-- Determine Effective Number of Players --#
if pp.TABLE_STATE['roundName'] == 'Flop':
    # Flop cards are just dealt, anyone that folded preflop can be considered out of the game and not figured in win probability calculation
    pp.TABLE_STATE['N_effective']  = (pp.GAME_STATE.isSurvive & ~pp.GAME_STATE.folded).sum()

agent(event_name,data)

#-- __action --#
event_name,data  = '__action',{'game':table.copy(),}
data['self']             = players[players.playerName==pp.TABLE_STATE['name_md5']].iloc[0]
data['self']['minBet']   = 0
data['game']['players']  = [x.copy() for _,x in players.iterrows()]
t0    = time.time()
if pp.GAME_STATE is None:
    pp.init_game_state(data['game']['players'],data['game'])
else:
    pp.update_game_state(data['game']['players'],data['game'])

out   = pp.agent(event_name,data)
resp  = out[0]
log   = out[1]

log['cputime']  = time.time() - t0
pp.AGENT_LOG.append(log)
pp.TABLE_STATE['forced_bet']  = 0
pp.PREV_AGENT_STATE  = log
print(pp.PREV_AGENT_STATE)
print()
