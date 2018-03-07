from common import *
import json,hashlib
from websocket import create_connection

def pkr_to_str(pkr):
    # Trend micro poker platform format to string
    return [suitmap[x[1].lower()]+(x[0] if x[0]!='T' else '10') for x in pkr]

def pkr_to_cards(pkr):
    # Trend micro poker platform format to pkrprb format
    cards  = [((suitmap[x[1].lower()],rankmap[x[0].lower()]),suitmap[x[1].lower()],rankmap[x[0].lower()]) for x in pkr]
    return pd.DataFrame(cards,columns=('c','s','o'))

def doListen(url,name,action):
    # try:
    ws  = create_connection(url)
    ws.send(json.dumps({
        'eventName': '__join',
        'data': {'playerName': name,},
        }))
    while True:
        msg  = ws.recv()
        msg  = json.loads(msg)
        event_name = msg['eventName']
        data       = msg['data']
        if event_name == '__game_prepare':
            print("Table %s: Game starts in %d sec(s)"%(data['tableNumber'],data['countDown']))
        elif event_name == '__game_start':
            print("Table %s: Game start!!!"%data['tableNumber'])
            print()
        # elif event_name in ('__new_round','_join','__deal','__round_end'):
        #     print("Table %s: Round %s:"%(data['table']['tableNumber'],data['table']['roundName']))
        #     print("Board [%s]" % ' '.join(cards_to_str(data['table']['board'])))
        #     SB_name = data['table']['smallBlind']['playerName']
        #     BB_name = data['table']['bigBlind']['playerName']
        #     for x in data['players']:
        #         print("%32s %6d %6d %s [%s]" % (
        #             'Me' if x['playerName']==name_md5 else x['playerName'],
        #             x['chips'],
        #             x['bet'],
        #             'SB' if SB_name==x['playerName'] else ('BB' if BB_name==x['playerName'] else '  '),
        #             ' '.join(cards_to_str(x['cards'])) if not x['folded'] else 'Folded',
        #             ))
        #     print()
        elif event_name == '__show_action':
            print("Table %s: Round %s: Player %32s (%6d) %s%s" % (data['table']['tableNumber'],data['table']['roundName'],data['action']['playerName'],data['action']['chips'],data['action']['action']," with %d" % data['action']['amount'] if 'amount' in data['action'] else ''))
            print()
        elif event_name == '__game_over':
            print("Table %s Game Over" % (data['table']['tableNumber']))
        # elif event_name not in ('__start_reload',): #'__bet','__action',
        #     print(event_name,'\n',json.dumps(data,indent=4))
        #
        action(ws,event_name,data)
    # except Exception as e:
    #     print("Exception: ",e)
    #     doListen(url,name,action)
