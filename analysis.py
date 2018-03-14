from agent_common import *
import os,glob

batch_name = sys.argv[1] #'basic_vs_human' #'basic_1+twice'
filelist = glob.glob('game_records' + os.sep + batch_name + os.sep + 'game_' + '?'*14 + '.csv')

results  = {}
for filename in filelist:
    game_id = filename.rsplit('.',1)[0].rsplit('_',1)[1]
    res     = pd.read_csv(filename).rename(columns={'Unnamed: 0':'playerName'}).set_index('playerName')
    # res     = pd.read_csv(filename,index_col='playerName')
    res.index = ['隨便' if x.startswith('隨便') else x for x in res.index]
    results[game_id] = res.score.fillna(0)
    #
    #-- basic_0 format --#
    # game_id = filename.rsplit('.',1)[0].rsplit('_',1)[1]
    # res     = pd.read_csv(filename).rename(columns={'Unnamed: 0':'name'}).set_index('name')
    # results[game_id] = res.amt.fillna(0)

results  = pd.concat(results,1).transpose()
results.rename(columns=playerMD5,inplace=True)

results  = results[results.notnull().all(1)]

#-- Player Ranking --#
player_ranking = pd.DataFrame(index=results.columns,columns=('survive','first','money'))
player_ranking['survive']  = (results>0).mean(0).sort_values(ascending=False)
player_ranking['first']    = results.idxmax(1).value_counts().sort_values(ascending=False)/len(results)
player_ranking['money']    = results.mean(0).sort_values(ascending=False)

# 3/13
# playerName      survive  first    money 
# (=^-Ω-^=)       30.00%   25.00%   2,484.63 
# FourSwordsMan   31.67%   20.00%   2,235.02 
# A_P_T^T         58.33%   20.00%   2,552.98 
# ヽ(=^･ω･^=)丿    78.33%   35.00%   4,727.30 
