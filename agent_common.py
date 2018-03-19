from common import *
import json,hashlib
from websocket import create_connection
from datetime import datetime
import multiprocessing as mp

pd.set_option('display.width',200)
pd.set_option('display.unicode.east_asian_width',True)

ws  = None
pc  = []
pq  = None
prWin_samples = []
game_id    = None
round_id   = 0
turn_id    = 0
game_state = None
game_board = None
game_actions = None
player_stats = None

playerNames = ['jyp','jyp0','jyp1','jyp2','jyp3','jyp4','jyp5','twice','Samuel','steven','465fc773c4']

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
    state  = pd.DataFrame(players)
    state['cards']  = state.cards.fillna('').apply(pkr_to_str)
    state.set_index('playerName',inplace=True)
    state.loc[table['smallBlind']['playerName'],'position'] = 'SB'
    state.loc[table['bigBlind']['playerName'],'position']   = 'BB'
    state['action'] = np.nan
    state['amount'] = np.nan
    state['me']     = state.index==name_md5
    # print(state)
    return state

def update_game_state(state,players,table,action=None):
    for x in players:
        idx  = x['playerName']
        for col in ('allIn','bet','chips','folded','isHuman','isOnline','isSurvive','reloadCount','roundBet'):
            state.loc[idx,col]  = x[col]
        state.loc[idx,'cards'] = pkr_to_str(x['cards']) if 'cards' in x else ''
    if action is not None:
        idx  = action['playerName']
        state.loc[idx,'action'] = action['action']
        state.loc[idx,'amount'] = action['amount'] if 'amount' in action else np.nan

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

def get_player_stats():
    global player_stats
    return player_stats

def record_round_results(state,round_id,players,bets=None):
    result  = state.copy().drop(['chips','reloadCount'],'columns')
    if bets is not None:
        result  = pd.concat([result,bets],1)
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
    result  = result.reindex(columns=['chips','reloadCount','cards','hand','position','act_deal','bet_deal','act_flop','bet_flop','act_turn','bet_turn','act_river','bet_river','allIn','folded','rank','message','winMoney'])
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

def read_win_prob(N,hole):
    # "Normalize" two card combinations
    hole.sort_values('o',ascending=False,inplace=True)
    if hole.s.iloc[0] == hole.s.iloc[1]:
        hole['s'] = '♠'
    else:
        hole['s'] = ['♠','♥']
    hole['c'] = [(x,y) for x,y in hole[['s','o']].values]
    #
    res  = pd.read_csv("sim_prob/sim2_N10_h[%s].csv.gz" % cards_to_str(hole).replace(' ',''))
    #
    if N < 10:
        res['prWin'] = 0
        mask = res['rank'] <= 11 - N
        res.loc[mask,'prWin'] = 1
        for i in range(N-1):
            res.loc[mask,'prWin'] *= (10 - res.loc[mask,'rank'] - i)/(9 - i)
        return res.prWin.mean(),res.prWin.std()
    elif N == 10:
        return res.pot.mean(),res.pot.std()

def calculate_win_prob(N,hole,board=(),Nsamp=100):
    deck  = new_deck()
    deck  = deck[~deck.c.isin(hole.c)]
    if len(board) > 0:
        deck  = deck[~deck.c.isin(board.c)]
    #
    pre_flop  = len(board) < 3
    pre_turn  = len(board) < 4
    pre_river = len(board) < 5
    #
    if not pre_flop:  flop  = board.iloc[:3]
    if not pre_turn:  turn  = board.iloc[3:4]
    if not pre_river:
        river  = board.iloc[4:5]
        score  = score_hand(pd.concat([hole,flop,turn,river]))
    #
    t0      = time.clock()
    pot_hat = np.zeros(Nsamp)
    for j in range(Nsamp):
        if pre_flop:
            cards = deck.sample(5 + (N-1)*2)
            flop  = cards[:3]
            turn  = cards[3:4]
            river = cards[4:5]
            holes_op = cards[5:]
        elif pre_turn:
            cards = deck.sample(2 + (N-1)*2)
            turn  = cards[:1]
            river = cards[1:2]
            holes_op = cards[2:]
        elif pre_river:
            cards = deck.sample(1 + (N-1)*2)
            river = cards[:1]
            holes_op = cards[1:]
        else:
            holes_op = deck.sample((N-1)*2)
        #
        if pre_river:
            score  = score_hand(pd.concat([hole,flop,turn,river]))
        #
        Nrank1     = 1
        pot_hat[j] = 1
        for i in range(N-1):
            resi  = compare_hands(score[0],pd.concat([holes_op[(2*i):(2*i+2)],flop,turn,river]))
            if resi < 0: # score[0] < scorei
                pot_hat[j] = 0
                break
            elif resi == 0: # score[0] == scorei
                Nrank1 += 1
            # scoresi = score_hand(pd.concat([holes_op[(2*i):(2*i+2)],flop,turn,river]))
            # if score[0] < scoresi[0]:
            #     pot_hat[j] = 0
            #     break
            # elif score[0] == scoresi[0]:
            #     Nrank1 += 1
        #
        if pot_hat[j] > 0:
            pot_hat[j] = (1/Nrank1) if Nrank1>1 else 1
    #
    print(time.clock() - t0)
    return pot_hat.mean(),pot_hat.std()

def calculate_win_prob_mp(q,N,hole,board=()):
    deck  = new_deck()
    deck  = deck[~deck.c.isin(hole.c)]
    if len(board) > 0:
        deck  = deck[~deck.c.isin(board.c)]
    #
    pre_flop  = len(board) < 3
    pre_turn  = len(board) < 4
    pre_river = len(board) < 5
    #
    if not pre_flop:  flop  = board.iloc[:3]
    if not pre_turn:  turn  = board.iloc[3:4]
    if not pre_river:
        river  = board.iloc[4:5]
        score  = score_hand(pd.concat([hole,flop,turn,river]))
    #
    # pot_hat  = []
    hole_str  = cards_to_str(hole)
    board_str = cards_to_str(board)
    while not q.full():
        if pre_flop:
            cards = deck.sample(5 + (N-1)*2)
            flop  = cards[:3]
            turn  = cards[3:4]
            river = cards[4:5]
            holes_op = cards[5:]
        elif pre_turn:
            cards = deck.sample(2 + (N-1)*2)
            turn  = cards[:1]
            river = cards[1:2]
            holes_op = cards[2:]
        elif pre_river:
            cards = deck.sample(1 + (N-1)*2)
            river = cards[:1]
            holes_op = cards[1:]
        else:
            holes_op = deck.sample((N-1)*2)
        #
        if pre_river:
            score  = score_hand(pd.concat([hole,flop,turn,river]))
        #
        Nrank1   = 1
        pot_hatj = 1
        for i in range(N-1):
            resi  = compare_hands(score[0],pd.concat([holes_op[(2*i):(2*i+2)],flop,turn,river]))
            if resi < 0: # score[0] < scorei
                pot_hatj = 0
                break
            elif resi == 0: # score[0] == scorei
                Nrank1 += 1
        #
        if pot_hatj > 0:
            pot_hatj = (1/Nrank1) if Nrank1>1 else 1
        #
        q.put({
            'N':     N,
            'hole':  hole_str,
            'board': board_str,
            'prWin': pot_hatj,
            },block=True,timeout=None)
        #
        # pot_hat.append(pot_hatj)
        # #
        # if len(pot_hat) >= 5:
        #     q.put(pot_hat,block=True,timeout=None)
        #     pot_hat = []

def calculate_win_prob_mp_start(N,hole,board=(),n_jobs=1):
    global pc
    global pq
    global prWin_samples
    #
    for pcc in pc:
        if pcc.is_alive(): pcc.terminate()
    #
    prWin_samples = [] #pd.DataFrame(columns=('N','hole','board','prWin'))
    pq  = mp.Queue(maxsize=0)
    pc  = []
    for _ in range(n_jobs):
        pc.append(mp.Process(target=calculate_win_prob_mp,args=(pq,N,hole,board)))
        pc[-1].start()

def calculate_win_prob_mp_get():
    global pc
    global pq
    global prWin_samples
    #
    while not pq.empty(): prWin_samples.append(pq.get_nowait())
    return prWin_samples

def calculate_win_prob_mp_stop():
    global pc
    global pq
    global prWin_samples
    #
    for pcc in pc:
        if pcc.is_alive(): pcc.terminate()
    if pq is not None:
        try:
            while not pq.empty(): prWin_samples.append(pq.get_nowait())
        except:
            pass

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
    #
    m   = hashlib.md5()
    m.update(name.encode('utf8'))
    name_md5   = m.hexdigest()
    decisions  = []
    round_results   = []
    round_decisions = {}
    round_bets      = None # All player's bets
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
                ws.close()
                ws   = None
                msg  = ''
        #
        t0         = time.time()
        event_name = msg['eventName']
        data       = msg['data']
        if event_name in ('__action','__bet'):
            if game_board is None: game_board = data['game']['board']
            if game_state is None:
                game_state = init_game_state(data['game']['players'],data['game'],name_md5=name_md5)
            else:
                update_game_state(game_state,data['game']['players'],data['game'])
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
            game_id   = datetime.now().strftime('%Y%m%d%H%M%S')
            game_state = None
            game_actions = None
            round_id  = 0
            turn_id   = 0
            if record:
                print("Table %s: Game %s start!!!\n"%(data['tableNumber'],game_id))
            else:
                print("Table %s: Game start!!!\n"%data['tableNumber'])
        elif event_name == '__new_round':
            round_id   += 1
            turn_id     = 0
            game_board  = data['table']['board']
            game_state  = init_game_state(data['players'],data['table'],name_md5=name_md5)
            round_bets  = game_state[['chips','reloadCount']].copy()
            round_bets['chips'] += game_state.bet
            #
            if player_stats is None:
                player_stats  = pd.DataFrame(0,columns=('rounds','bet/raise','amount','check/call','fold'),index=game_state.index)
            #
            resp   = action(event_name,data)
        elif event_name == '__deal':
            # Deal hole cards
            game_board  = data['table']['board']
            if game_state is None:
                game_state  = init_game_state(data['players'],data['table'],name_md5=name_md5)
            else:
                # Record last round's bets
                if round_bets is None:
                    round_bets  = pd.DataFrame(index=game_state.index)
                if data['table']['roundName'] == 'Flop':
                    col  = '_deal'
                elif data['table']['roundName'] == 'Turn':
                    col  = '_flop'
                else: #if data['table']['roundName'] == 'River':
                    col  = '_turn'
                round_bets['act' + col] = game_state.action
                round_bets['bet' + col] = game_state.bet
                #
                update_game_state(game_state,data['players'],data['table'])
            #
            resp   = action(event_name,data)
        elif event_name == '__show_action':
            # Player action
            if game_board is None: game_board = data['table']['board']
            if game_state is None:
                game_state  = init_game_state(data['players'],data['table'],name_md5=name_md5)
            update_game_state(game_state,data['players'],data['table'],data['action'])
            #
            if game_actions is None:
                game_actions  = pd.DataFrame(columns=('game_id','round_id','turn_id','roundName','playerName','chips','reloadCount','position','pot','bet','action','amount'))
            #
            turn_id += 1
            act   = data['action']
            act['game_id']   = game_id
            act['round_id']  = round_id
            act['turn_id']   = turn_id
            act['roundName'] = data['table']['roundName']
            act['reloadCount'] = game_state.loc[act['playerName'],'reloadCount']
            act['position']    = game_state.loc[act['playerName'],'position']
            act['pot']       = game_state.roundBet.sum() + game_state.bet.sum()
            act['bet']       = game_state.loc[act['playerName'],'bet']
            game_actions = game_actions.append(act,ignore_index=True)
            #
            if player_stats is not None:
                if act['playerName'] not in player_stats.index:
                    player_stats.loc[act['playerName']] = 0
                if act['action'] == 'allin':
                    if act['amount'] > game_state.bet.max():
                        player_stats.loc[act['playerName'],'bet/raise'] += 1
                        player_stats.loc[act['playerName'],'amount'] += act['amount']
                    else:
                        player_stats.loc[act['playerName'],'check/call'] += 1
                if act['action'] in ('bet','raise'):
                    player_stats.loc[act['playerName'],'bet/raise'] += 1
                    player_stats.loc[act['playerName'],'amount']    += act['amount']
                elif act['action'] in ('check','call'):
                    player_stats.loc[act['playerName'],'check/call'] += 1
                elif act['action'] == 'fold':
                    player_stats.loc[act['playerName'],'fold'] += 1
            #
            resp   = action(event_name,data)
        elif event_name == '__round_end':
            if game_board is None: game_board = data['table']['board']
            if game_state is None:
                game_state  = init_game_state(data['players'],data['table'],name_md5=name_md5)
            else:
                # Record last round's bets
                if round_bets is None:
                    round_bets  = pd.DataFrame(index=game_state.index)
                #
                round_bets['act_river'] = game_state.action
                round_bets['bet_river'] = game_state.bet
                #
                update_game_state(game_state,data['players'],data['table'])
            #
            round_results.append(record_round_results(game_state,round_id,data['players'],round_bets))
            #
            # round_results[-1].to_csv("game_%s_round_%d.csv"%(game_id,round_id),encoding='utf-8-sig')
            if len(decisions) > 0:
                round_decisions[round_id] = pd.concat(decisions,1).transpose()
                # pd.concat(decisions,1).transpose().to_csv("game_%s_round_%d_actions.csv"%(game_id,round_id),encoding='utf-8-sig')
                decisions = []
            #
            if player_stats is not None:
                for playerName in game_state.index:
                    if playerName not in player_stats.index:
                        player_stats.loc[playerName] = 0
                player_stats.loc[game_state.index,'rounds'] += 1
            #
            resp   = action(event_name,data)
        elif event_name in ('__game_over','__game_stop'):
            try:
                if game_board is None: game_board = data['game']['board']
                if game_state is None:
                    game_state  = init_game_state(data['players'],data['table'],name_md5=name_md5)
                else:
                    update_game_state(game_state,data['players'],data['table'])
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
                    output.amount /= output['bet/raise']
                    output['prBet']  = output['bet/raise']/(output['bet/raise']+output['check/call']+output.fold)
                    output['prFold'] = output.fold/(output['bet/raise']+output['check/call']+output.fold)
                    output.index   = ['Me' if name_md5==x else (playerMD5[x] if x in playerMD5 else x) for x in output.index]
                    print(output)
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
