from common import *
import json

pd.set_option('display.max_rows',120)
pd.set_option('display.max_columns',None)
pd.set_option('display.width',90)

#---------------#
#-- Read Data --#
#---------------#
dt    = sys.argv[1] #'20180716'
# game    = pd.read_csv('data/game_log_'+dt+'.gz')
rnd   = pd.read_csv('data/round_log_'+dt+'.gz')
# action  = pd.read_csv('data/action_log_'+dt+'.gz')

#-- Calculate Hand Scores --#
mask  = rnd.cards.notnull()

#-- Deal --#
t0  = time.clock()
temp  = hole_texture_batch(rnd[mask].cards)
temp['cards_category']  = hole_texture_to_category_batch(temp)
rnd.loc[mask,'score_Deal'] = temp.apply(lambda x:json.dumps((9-x.cards_category,int(x.cards_pair),x.cards_rank1,0 if x.cards_pair else x.cards_rank2,)).replace(' ',''),axis=1)
print(time.clock() - t0)

#-- Post-Flop --#
t0  = time.clock()
hand  = (rnd[mask].cards + ' ' + rnd[mask].board).str.split()
rnd.loc[mask,'score_Flop']  = hand.str[:5].apply(lambda x:json.dumps(score_hand5(pkr_to_cards(x))).replace(' ',''))
rnd.loc[mask,'score_Turn']  = hand.str[:6].apply(lambda x:json.dumps(score_hand(pkr_to_cards(x))).replace(' ',''))
rnd.loc[mask,'score_River'] = hand.apply(lambda x:json.dumps(score_hand(pkr_to_cards(x))).replace(' ',''))
print(time.clock() - t0)

rnd.to_csv('round_log_'+dt+'.gz',index=False,compression='gzip')
