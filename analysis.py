from agent_common import *
import glob

filelist = glob.glob('game_records/game_*.csv')
results  = {}
for filename in filelist:
    game_id = filename.rsplit('.',1)[0].rsplit('_',1)[1]
    res     = pd.read_csv(filename).rename(columns={'Unnamed: 0':'name'}).set_index('name')
    results[game_id] = res.amt.fillna(0)

results  = pd.concat(results,1).transpose()
