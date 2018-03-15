from common import *
import os,glob

results   = pd.DataFrame(index=pd.MultiIndex(levels=[[],[]],labels=[[],[]]))
filelist  = glob.glob('sim_prob' + os.sep + 'sim_*')
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
