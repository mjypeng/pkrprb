from common import *
import json,hashlib
from websocket import create_connection
from datetime import datetime

pd.set_option('display.width',100)
pd.set_option('display.unicode.east_asian_width',True)

ws  = None
game_id    = None
round_id   = 0
turn_id    = 0
game_state = None
game_board = None
game_actions = None
player_stats = None
expsmo_alpha = 0.1

playerNames = ['jyp','jyp0','jyp1','jyp2','jyp3','jyp4','jyp5','twice','Samuel','steven','465fc773c4','basic1','basic2','basic3','basic4','random1','random2','random3','pot','fold','cat1','cat4','cat5','8+9','87-dawn-ape','87-rising-ape','87945','V.S.A.','basic65536','houtou_a','houtou_p','tomas','lefthand_cat','89','basic','cat0','cat1','cat2','cat3','cat4','cat5','cat6','cat7','cat8','cat9','basic_a','basic_b','basic_c','tomas2','fold1','pot1']
# cat4 => Leo
# cat9 => Teebone
# cat1 => jyp

playerMD5   = {}
for playerName in playerNames:
    m    = hashlib.md5()
    m.update(playerName.encode('utf8'))
    playerMD5[m.hexdigest()] = playerName

playerMD5['530ea4dde1e76f98c6a35459743033a9'] = '( ´_ゝ`) Dracarys'
playerMD5['ce324d61e647fa3eb81c23dec02e659c'] = '( Φ Ω Φ )'
playerMD5['a0eb6599677296b01f9300e77c8f0a2d'] = '(=^-Ω-^=)'
playerMD5['915ed2eeb968868a41e8701c25436842'] = '(=´ᴥ`)'
playerMD5['a7599eb7a3b0cb2e0cad8bc7241ba810'] = '8+9'
playerMD5['52518335a5521597e78ae1597d11f879'] = 'A_P_T^T'
playerMD5['5929f41996f4ce76a68a77ce153275a0'] = 'aaa'
playerMD5['56fc9be9944fc896dced79d7bde9a100'] = 'AllinWin'
playerMD5['b05db4c3533a17098b7bed4fd774a38f'] = 'BRIMS'
playerMD5['befa275bf113f179e653e3b63f0b7519'] = 'Chicken Dinner'
playerMD5['fcb0342a8968cbcd6faafc485ead4884'] = 'Commentallez-vous'
playerMD5['6196e9650cc56c70b3faee4e60663aaf'] = 'ERS yo'
playerMD5['b9573e57ac2c4cf458fffb47017d53b7'] = 'fatz'
playerMD5['1519630acb1b3569b8869d6e1568a08d'] = 'FourSwordsMan'
playerMD5['8f154c38bb3040b51d2c28ff1e6b19be'] = 'Hodor'
playerMD5['171311fb8b6fe8c975811367467b8eed'] = 'I DO GOGOGO'
playerMD5['ce36ca046a9423ca1cfa95c434f615cc'] = 'iCRC必勝'
playerMD5['f164881adcdc582c2c6ef9ffdc0bec25'] = 'Im Not Kao Chin'
playerMD5['37c755f40d65a4bc736dfab20e7144c2'] = 'Jee'
playerMD5['2817312802d1b71380bd058ad510a7f2'] = 'Joker'
playerMD5['dde0d676b3c85415d7725ad117fb7fb1'] = 'Minari'
playerMD5['11202e53d6e360903d1fe8293e3ad5f1'] = 'Orca Poker'
playerMD5['2c3e80a6aa48ea3e71262b66c3f7e0c8'] = 'Out Bluffed'
playerMD5['5963cc50c9574d3d35605ecda5f6e627'] = 'P.I.J'
playerMD5['41579b8940a2482619500991a7dd4bc5'] = 'PCB'
playerMD5['d8c5f9a3a9e26e83b4ee0afa5be7ccc3'] = 'Poker\'s Finger'
playerMD5['7ac37daa9667f89d8e383b826233e9df'] = 'poy and his good friends'
playerMD5['26c29e9949501822a57c991ce9960a6e'] = 'Red Sparrow'
playerMD5['345198c48c818615ce802cdd91d34ddd'] = 'S.U.N.S.A.'
playerMD5['67932ee63b2984a136db48eb9e994f8e'] = 'SML'
playerMD5['8c344848375fda783d24c92f7189ee39'] = 'TeamMustJoin'
playerMD5['973d1f9ad0a2bba82b6b3a5e4f27adb8'] = 'testing'
playerMD5['c767b3b24b29c5a270658fd690cd919e'] = 'TMBS=trend micro\'s best supreme'
playerMD5['cc916b8f0759af7958a8a927e191ac35'] = 'Tobacco AI'
playerMD5['8e85063af83bd95032e9c96919a14a80'] = 'V.S.A.'
playerMD5['31b65b0d46c540b5e27c55323e476e0f'] = 'Winner Winner'
playerMD5['a88dd117b68794c33519068b86ca7dda'] = 'Yeeeee'
playerMD5['312098e67bde366ca7a4f1b0148cb03d'] = 'ミ^・.・^彡'
playerMD5['a3d68e76ba28878bb2611c5e193e99d1'] = '柏林常勝王'
playerMD5['4811e13de92d629cf35235fc225dde86'] = '善於打牌的低能機器人'
playerMD5['aa5c076ff89ad5f2fe1c2574626f0a94'] = '惡意送頭'
playerMD5['eef2e0db65ec73153c303f455b3ead7e'] = '隨便'
playerMD5['9339e4be51695e365e9740fe6f34385b'] = '趨勢白牌娛樂城上線啦'
playerMD5['0d2463992a3cce805eb5b0adf7804594'] = 'ヽ(=^･ω･^=)丿'
playerMD5['465fc773c4'] = 'ヽ(=^･ω･^=)丿'

def takeAction(x):
    # x[0]: prob to fold
    # x[1]: prob to check/call
    # x[2]: prob to bet/raise
    # x[3]: amount to bet/raise, use string 'raise' to raise double last raise, use 0 to bet minimum amount
    # 1 - x[0] - x[1] - x[2] is prob to go allin
    samp  = np.random.random()
    if samp < x[0]:
        return ('fold',0)
    elif samp < x[0] + x[1]:
        return ('check',0)
    elif samp < x[0] + x[1] + x[2]:
        if x[3] == 'raise':
            return ('raise',0)
        else:
            return ('bet',x[3])
    else:
        return ('allin',0)

def num_rounds_to_bankrupt(A,N,SB,nSB=0):
    if A <= 2*(N - nSB)*SB:
        T_BB  = np.ceil(A/(2*SB))
    else:
        A    -= 2*(N - nSB)*SB
        T_BB  = N - nSB
        k     = 4
        while A > 0:
            if A > k*N*SB:
                A    -= k*N*SB
                T_BB += N
                k    *= 2
            else:
                T_BB += np.ceil(A/(k*SB))
                A     = 0
    return T_BB

def init_game_state(players,table,name_md5=None):
    global game_state
    game_state  = pd.DataFrame(players)
    game_state['cards']  = game_state.cards.fillna('').apply(pkr_to_str)
    game_state.set_index('playerName',inplace=True)
    game_state.loc[table['smallBlind']['playerName'],'position'] = 'SB'
    game_state.loc[table['bigBlind']['playerName'],'position']   = 'BB'
    #
    SB_idx = (game_state.position=='SB').values.argmax()
    BB_idx = (game_state.position=='BB').values.argmax()
    D_idx  = (SB_idx - 1) % len(players)
    while not game_state.loc[game_state.index[D_idx],'isSurvive'] and D_idx != BB_idx:
        D_idx = (D_idx - 1) % len(players)
    if D_idx not in (SB_idx,BB_idx):
        game_state.loc[game_state.index[D_idx],'position'] = 'D'
    idx  = (BB_idx + 1) % len(players)
    pos  = 1
    while idx not in (SB_idx,BB_idx,D_idx):
        if game_state.loc[game_state.index[idx],'isSurvive']:
            game_state.loc[game_state.index[idx],'position'] = pos
            pos  += 1
        idx  = (idx + 1) % len(players)
    #
    game_state['action'] = np.nan
    game_state['amount'] = np.nan
    game_state['me']     = game_state.index==name_md5

def update_game_state(players,table,action=None):
    global game_state
    for x in players:
        idx  = x['playerName']
        for col in ('allIn','bet','chips','folded','isHuman','isOnline','isSurvive','reloadCount','roundBet'):
            game_state.loc[idx,col]  = x[col]
        game_state.loc[idx,'cards'] = pkr_to_str(x['cards']) if 'cards' in x else ''
    if action is not None:
        idx  = action['playerName']
        game_state.loc[idx,'action'] = action['action']
        game_state.loc[idx,'amount'] = action['amount'] if 'amount' in action else np.nan

def get_game_state():
    global game_id
    global round_id
    global game_state
    return game_id,round_id,game_state

def get_game_actions():
    global game_id
    global round_id
    global turn_id
    global game_actions
    return game_id,round_id,turn_id,game_actions

def init_player_stats(table):
    global game_state
    global player_stats
    player_stats  = pd.DataFrame(0,columns=pd.MultiIndex.from_tuples([(x,y) for x in ('deal','flop','turn','river') for y in ('rounds','prBet','amtBet','prCall','amtCall','oddsCall','prFold','amtFold','lossFold')]),index=game_state.index)
    for rnd in ('deal','flop','turn','river'):
        player_stats[(rnd,'rounds')]  = 0
        player_stats[(rnd,'prBet')]   = 0.20
        player_stats[(rnd,'amtBet')]  = table['bigBlind']['amount']
        player_stats[(rnd,'prCall')]  = 0.60
        player_stats[(rnd,'amtCall')] = table['smallBlind']['amount']
        player_stats[(rnd,'oddsCall')] = 1/(2*game_state.isSurvive.sum()+1)
        player_stats[(rnd,'prFold')]  = 0.20
        player_stats[(rnd,'amtFold')] = table['smallBlind']['amount']
        player_stats[(rnd,'lossFold')] = table['smallBlind']['amount']

def get_player_stats():
    global player_stats
    return player_stats

def record_round_results(state,round_id,players):
    result  = state.copy()
    result.reset_index(drop=False,inplace=True)
    result.drop(['action','amount'],'columns',inplace=True)
    result['round_id'] = round_id
    result.set_index(['round_id','playerName'],drop=True,append=False,inplace=True)
    for x in players:
        idx  = (round_id,x['playerName'])
        if 'hand' in x:
            result.loc[idx,'hand']     = pkr_to_str(x['hand']['cards'])
            result.loc[idx,'rank']     = x['hand']['rank']
            result.loc[idx,'message']  = x['hand']['message']
        result.loc[idx,'winMoney'] = x['winMoney']
    #
    result  = result.reindex(columns=['chips','reloadCount','cards','hand','position','allIn','folded','rank','message','winMoney'])
    return result

def record_game_results(state,winners):
    result  = state.copy()
    result.drop(['action','amount'],'columns',inplace=True)
    for x in winners:
        idx  = x['playerName']
        result.loc[idx,'win']   = True
        result.loc[idx,'score'] = x['chips']
    return result

def pkr_to_str(pkr):
    # Trend micro poker platform format to string
    return ' '.join([suitmap[x[1].lower()]+(x[0] if x[0]!='T' else '10') for x in pkr])

def pkr_to_cards(pkr):
    # Trend micro poker platform format to pkrprb format
    cards  = [((suitmap[x[1].lower()],rankmap[x[0].lower()]),suitmap[x[1].lower()],rankmap[x[0].lower()]) for x in pkr]
    return pd.DataFrame(cards,columns=('c','s','o'))

#-- Agent Event Loop --#
def doListen(url,name,action,record=False):
    global playerMD5
    global ws
    global game_id
    global round_id
    global turn_id
    global game_state
    global game_board
    global game_actions
    global player_stats
    global expsmo_alpha
    #
    m   = hashlib.md5()
    m.update(name.encode('utf8'))
    name_md5   = m.hexdigest()
    decisions  = []
    round_results   = []
    round_decisions = {}
    while True:
        if ws is not None:
            try:
                msg  = ws.recv()
                msg  = json.loads(msg)
            except Exception as e:
                print(e)
                ws.close()
                ws   = None
                msg  = ''
        while ws is None or len(msg) == 0:
            try:
                time.sleep(3)
                print('Rejoining ...')
                ws   = create_connection(url)
                ws.send(json.dumps({
                    'eventName': '__join',
                    'data': {'playerName': name,},
                    }))
                msg  = ws.recv()
                msg  = json.loads(msg)
            except Exception as e:
                print(e)
                if ws is not None: ws.close()
                ws   = None
                msg  = ''
        #
        t0         = time.time()
        event_name = msg['eventName']
        data       = msg['data']
        if event_name in ('__action','__bet'):
            if game_board is None: game_board = data['game']['board']
            if game_state is None:
                init_game_state(data['game']['players'],data['game'],name_md5=name_md5)
            else:
                update_game_state(data['game']['players'],data['game'])
            #
            resp   = action(event_name,data)
            ws.send(json.dumps({
                'eventName': '__action',
                'data': {
                    'playerName': name,
                    'action': resp[0][0],
                    'amount': resp[0][1],
                    }
                }))
            resp[1]['cputime']  = time.time() - t0
            decisions.append(resp[1])
            #
            print("Action:")
            print(resp[1])
            print()
            #
        elif event_name == '__game_prepare':
            print("Table %s: Game starts in %d sec(s)"%(data['tableNumber'],data['countDown']))
        elif event_name == '__game_start':
            game_id      = datetime.now().strftime('%Y%m%d%H%M%S')
            game_state   = None
            game_actions = None
            round_id     = 0
            turn_id      = 0
            if record:
                print("Table %s: Game %s start!!!\n"%(data['tableNumber'],game_id))
            else:
                print("Table %s: Game start!!!\n"%data['tableNumber'])
        elif event_name == '__new_round':
            round_id   += 1
            turn_id     = 0
            game_board  = data['table']['board']
            init_game_state(data['players'],data['table'],name_md5=name_md5)
            #
            if player_stats is None:
                player_stats  = pd.DataFrame(0,columns=pd.MultiIndex.from_tuples([(x,y) for x in ('deal','flop','turn','river') for y in ('rounds','prBet','amtBet','prCall','amtCall','oddsCall','prFold','amtFold','lossFold')]),index=game_state.index)
                player_stats[('deal','prBet')]  = 0.33
                player_stats[('deal','amtBet')] = data['table']['bigBlind']['amount']
                player_stats[('deal','prCall')] = 0.33
                player_stats[('deal','amtCall')] = data['table']['smallBlind']['amount']
                player_stats[('deal','oddsCall')] = 1/(2*game_state.isSurvive.sum()+1)
                player_stats[('deal','prFold')] = 0.33
                player_stats[('deal','amtFold')] = data['table']['smallBlind']['amount']
                player_stats[('deal','lossFold')] = data['table']['smallBlind']['amount']
            else:
                for playerName in game_state.index:
                    if playerName not in player_stats.index:
                        player_stats.loc[playerName] = 0
                        player_stats.loc[playerName,('deal','prBet')]  = 0.33
                        player_stats.loc[playerName,('deal','amtBet')] = data['table']['bigBlind']['amount']
                        player_stats.loc[playerName,('deal','prCall')] = 0.33
                        player_stats.loc[playerName,('deal','amtCall')] = data['table']['smallBlind']['amount']
                        player_stats.loc[playerName,('deal','oddsCall')] = 1/(2*game_state.isSurvive.sum()+1)
                        player_stats.loc[playerName,('deal','prFold')] = 0.33
                        player_stats.loc[playerName,('deal','amtFold')] = data['table']['smallBlind']['amount']
                        player_stats.loc[playerName,('deal','lossFold')] = data['table']['smallBlind']['amount']
            #
            player_stats[('deal','rounds')] += 1
            #
            resp   = action(event_name,data)
        elif event_name == '__deal':
            # Deal hole cards
            game_board  = data['table']['board']
            if game_state is None:
                init_game_state(data['players'],data['table'],name_md5=name_md5)
            else:
                update_game_state(data['players'],data['table'])
            #
            rnd  = data['table']['roundName'].lower()
            if game_state is not None and player_stats is not None:
                player_stats[(rnd,'rounds')] += (~game_state.folded & game_state.isSurvive).astype(int)
            #
            resp   = action(event_name,data)
        elif event_name == '__show_action':
            # Player action
            if game_board is None: game_board = data['table']['board']
            if game_state is None:
                init_game_state(data['players'],data['table'],name_md5=name_md5)
            update_game_state(data['players'],data['table'],data['action'])
            #
            if game_actions is None:
                game_actions  = pd.DataFrame(columns=('game_id','round_id','turn_id','roundName','playerName','chips','reloadCount','position','pot','bet','action','amount'))
            #
            turn_id += 1
            act   = pd.Series(data['action'])
            act['game_id']   = game_id
            act['round_id']  = round_id
            act['turn_id']   = turn_id
            act['roundName'] = data['table']['roundName']
            act['reloadCount'] = game_state.loc[act.playerName,'reloadCount']
            act['position']    = game_state.loc[act.playerName,'position']
            act['pot']       = game_state.roundBet.sum() + game_state.bet.sum()
            act['bet']       = game_state.loc[act.playerName,'bet']
            if 'amount' not in act: act['amount'] = 0
            game_actions = game_actions.append(act.copy(),ignore_index=True)
            #
            if player_stats is not None:
                roundName   = data['table']['roundName'].lower()
                playerName  = act.playerName
                if playerName not in player_stats.index:
                    player_stats.loc[playerName] = 0
                    player_stats.loc[playerName,('deal','prBet')]  = 0.33
                    player_stats.loc[playerName,('deal','amtBet')] = data['table']['bigBlind']['amount']
                    player_stats.loc[playerName,('deal','prCall')] = 0.33
                    player_stats.loc[playerName,('deal','amtCall')] = data['table']['smallBlind']['amount']
                    player_stats.loc[playerName,('deal','oddsCall')] = 1/(2*game_state.isSurvive.sum()+1)
                    player_stats.loc[playerName,('deal','prFold')] = 0.33
                    player_stats.loc[playerName,('deal','amtFold')] = data['table']['smallBlind']['amount']
                    player_stats.loc[playerName,('deal','lossFold')] = data['table']['smallBlind']['amount']
                #
                player_stats.loc[playerName,(roundName,'prBet')]  *= 1 - expsmo_alpha
                player_stats.loc[playerName,(roundName,'prCall')] *= 1 - expsmo_alpha
                player_stats.loc[playerName,(roundName,'prFold')] *= 1 - expsmo_alpha
                #
                if act.action == 'allin':
                    if act.amount > game_state.bet.max():
                        act['action'] = 'bet'
                    else:
                        act['action'] = 'call'
                if act.action in ('bet','raise'):
                    player_stats.loc[playerName,(roundName,'prBet')] += expsmo_alpha
                    player_stats.loc[playerName,(roundName,'amtBet')] = expsmo_alpha*act.amount + (1-expsmo_alpha)*player_stats.loc[playerName,(roundName,'amtBet')]
                elif act.action in ('check','call'):
                    player_stats.loc[playerName,(roundName,'prCall')] += expsmo_alpha
                    player_stats.loc[playerName,(roundName,'amtCall')] = expsmo_alpha*act.amount + (1-expsmo_alpha)*player_stats.loc[playerName,(roundName,'amtCall')]
                    player_stats.loc[playerName,(roundName,'oddsCall')] = expsmo_alpha*act.amount/(act.pot+act.amount) + (1-expsmo_alpha)*player_stats.loc[playerName,(roundName,'oddsCall')]
                elif act.action == 'fold':
                    player_stats.loc[playerName,(roundName,'prFold')] += expsmo_alpha
                    player_stats.loc[playerName,(roundName,'amtFold')] = expsmo_alpha*(game_state.bet.max() - act.bet) + (1-expsmo_alpha)*player_stats.loc[playerName,(roundName,'amtFold')]
                    player_stats.loc[playerName,(roundName,'lossFold')] = expsmo_alpha*(game_state.loc[playerName,'roundBet'] + act.bet) + (1-expsmo_alpha)*player_stats.loc[playerName,(roundName,'lossFold')]
            #
            resp   = action(event_name,data)
        elif event_name == '__round_end':
            if game_board is None: game_board = data['table']['board']
            if game_state is None:
                init_game_state(data['players'],data['table'],name_md5=name_md5)
            else:
                update_game_state(data['players'],data['table'])
            #
            round_results.append(record_round_results(game_state,round_id,data['players']))
            #
            if len(decisions) > 0:
                round_decisions[round_id] = pd.concat(decisions,1).transpose()
                decisions = []
            #
            resp   = action(event_name,data)
        elif event_name in ('__game_over','__game_stop'):
            try:
                if game_board is None: game_board = data['game']['board']
                if game_state is None:
                    init_game_state(data['players'],data['table'],name_md5=name_md5)
                else:
                    update_game_state(data['players'],data['table'])
            except:
                pass
            if game_state is not None and 'winners' in data:
                result  = record_game_results(game_state,data['winners'])
                result.index = [playerMD5[x] if x in playerMD5 else x for x in result.index]
                result.to_csv("game_%s.csv"%game_id,encoding='utf-8-sig')
            #
            if len(round_results) > 0:
                result  = pd.concat(round_results,0)
                round_results = []
                temp    = [playerMD5[x] if x in playerMD5 else x for x in result.index.get_level_values('playerName')]
                result.reset_index('playerName',drop=False,inplace=True)
                result['playerName'] = temp
                result.set_index('playerName',drop=True,append=True,inplace=True)
                result.to_csv("game_%s_rounds.csv"%game_id,encoding='utf-8-sig')
            #
            if len(round_decisions) > 0:
                result  = pd.concat(round_decisions,0).sort_index()
                round_decisions = {}
                result.index.names = ('round_id','turn')
                result.to_csv("game_%s_decisions.csv"%game_id,encoding='utf-8-sig')
            #
            if game_actions is not None:
                game_actions['playerName'] = [playerMD5[x] if x in playerMD5 else x for x in game_actions.playerName]
                game_actions.to_csv("game_%s_actions.csv"%game_id,index=False,encoding='utf-8-sig')
                game_actions = None
            #
            resp   = action(event_name,data)
        elif event_name == '__start_reload':
            resp   = action(event_name,data)
            if resp:
                ws.send(json.dumps({
                    'eventName': '__reload',
                    }))
                print("Action: Reload")
        elif event_name not in ('__left','__new_peer'):
            print("event received: %s\n" % event_name)
        #
        #-- Console Output --#
        if event_name in ('__new_round','__deal','__show_action','__round_end','__game_over'):
            try:
                if record:
                    output  = player_stats.copy()
                    output.index   = ['Me' if name_md5==x else (playerMD5[x] if x in playerMD5 else x) for x in output.index]
                    output.loc['Table Median'] = output.median(0)
                    print(output.loc[['Me','Table Median']])
                    print()
                    #-- Output Game State --#
                    print("Table %s: Game %s:\nRound %d-%s: Board [%s]: Event %s" % (data['table']['tableNumber'],game_id,round_id,data['table']['roundName'],pkr_to_str(game_board),event_name))
                    output  = game_state.copy()
                    output.index = ['Me' if name_md5==x else (playerMD5[x] if x in playerMD5 else x) for x in output.index]
                    output.loc[output.allIn,'action'] = 'allin'
                    output.loc[output.folded,'cards'] = 'fold'
                    print(output[['chips','reloadCount','roundBet','bet','position','cards','action','amount']].rename(columns={'reloadCount':'reld','roundBet':'pot','position':'pos','action':'act','amount':'amt'}).fillna(''))
                    print()
                else:
                    print("event received: %s\n" % event_name)
            except:
                pass
