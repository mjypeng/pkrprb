from agent_common import *
import os,glob

# #-- Generate deal win prob table --#
# results   = pd.DataFrame()
# deck  = new_deck()
# deck  = deck[deck.s.isin(('♥','♠'))]
# for idx1,c1 in deck.iterrows():
#     for idx2,c2 in deck.iterrows():
#         if not c1.equals(c2) and c1.s == '♠' and c1.o >= c2.o:
#             hole  = pd.concat([c1,c2],1).transpose()
#             idx   = cards_to_str(hole)
#             results.loc[idx,'suit']  = int(c1.s==c2.s)
#             results.loc[idx,'rank']  = int(c1.o==c2.o)
#             results.loc[idx,'order'] = int(np.abs(c1.o-c2.o)==1)
#             results.loc[idx,'prob']  = (4 if c1.s==c2.s else (6 if c1.o==c2.o else 12))/(26*51)
#             for N in range(2,11):
#                 results.loc[cards_to_str(hole),N] = read_win_prob(N,hole)[0]
#             print(results.loc[idx],'\n')
#
# results.index.names = ('hole',)
# results.to_csv('deal_win_prob.csv',encoding='utf-8-sig')

#-- Generate score win rate --#
filelist  = glob.glob('sim_prob' + os.sep + 'sim*_*')
results   = None
for filename in filelist:
    print(filename)
    name  = filename.rsplit(os.sep,1)[1].split('_',1)[0]
    hole  = filename.rsplit(']')[0].rsplit('[')[1]
    res   = pd.read_csv(filename)
    res['score'] = res.score.apply(eval)
    res[10]      = res.pot
    for N in range(2,10):
        res[N] = 0
        mask = res['rank'] <= 11 - N
        res.loc[mask,N] = 1
        for i in range(N-1):
            res.loc[mask,N] *= (10 - res.loc[mask,'rank'] - i)/(9 - i)
    #
    wt    = 4 if hole[0]==hole[2] else (6 if hole[1]==hole[3] else 12)
    c     = wt*pd.concat([res.groupby('score').pot.count(),res.groupby('score')[list(range(2,11))].sum()],1).rename(columns={'pot':'wt'})
    if results is None:
        results  = c
    else:
        results  = results.add(c,fill_value=0)

temp  = results[results.index.str[0]==7].copy()
temp['score'] = temp.index.str[:2]
temp  = temp.groupby('score').sum()
results = pd.concat([results[results.index.str[0]!=7],temp],0)

temp  = results[results.index.str[0]==5].copy()
temp['score'] = temp.index.str[:3]
temp  = temp.groupby('score').sum()
results = pd.concat([results[results.index.str[0]!=5],temp],0)

results['prob']  = results.wt/results.wt.sum()
results[list(range(2,11))] = (results[list(range(2,11))].T/results.wt.T).T
results.to_csv('score_win_prob.csv',encoding='utf-8-sig')

# suit eq:  4 x (13 2)
# order eq: (4 2) x 13
# suit & order neq: (4 2) x (13 2) x 2
