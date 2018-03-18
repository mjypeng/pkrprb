from common import *
import os,glob

results   = pd.DataFrame()
deck  = new_deck()
deck  = deck[deck.s.isin(('♥','♠'))]
for idx1,c1 in deck.iterrows():
    for idx2,c2 in deck.iterrows():
        if not c1.equals(c2) and c1.s == '♠' and c1.o >= c2.o:
            hole  = pd.concat([c1,c2],1).transpose()
            idx   = cards_to_str(hole)
            results.loc[idx,'suit']  = int(c1.s==c2.s)
            results.loc[idx,'rank']  = int(c1.o==c2.o)
            results.loc[idx,'order'] = int(np.abs(c1.o-c2.o)==1)
            results.loc[idx,'prob']  = (4 if c1.s==c2.s else (6 if c1.o==c2.o else 12))/(26*51)
            for N in range(2,11):
                results.loc[cards_to_str(hole),N] = read_win_prob(N,hole)[0]
            print(results.loc[idx],'\n')

results.index.names = ('hole',)
results.to_csv('deal_win_prob.csv',encoding='utf-8-sig')





results   = pd.DataFrame(index=pd.MultiIndex(levels=[[],[]],labels=[[],[]]))
filelist  = glob.glob('sim_prob' + os.sep + 'sim2_*')
c         = None
for filename in filelist:
    name  = filename.rsplit(os.sep,1)[1].split('_',1)[0]
    hole  = filename.rsplit(']')[0].rsplit('[')[1]
    res   = pd.read_csv(filename)
    results.loc[(name,hole),'prWin']    = res.pot.mean()
    results.loc[(name,hole),'prWinStd'] = res.pot.std()
    if c is None:
        c  = res.groupby(['score','winner']).pot.count()
    else:
        c  = c.add(res.groupby(['score','winner']).pot.count(),fill_value=0)

results.to_csv('test.csv',encoding='utf-8-sig')

# suit eq:  4 x (13 2)
# order eq: (4 2) x 13
# suit & order neq: (4 2) x (13 2) x 2
