from agent_common import *
import os,glob

batch_name = sys.argv[1] #'basic_vs_human' #'basic_1+twice'
filelist = glob.glob('game_records' + os.sep + batch_name + os.sep + 'game_' + '?'*14 + '.csv')

results  = {}
for filename in filelist:
    game_id = pd.to_datetime(filename.rsplit('.',1)[0].rsplit('_',1)[1])
    try:
        res = pd.read_csv(filename,index_col='playerName')
    except:
        res = pd.read_csv(filename).rename(columns={'Unnamed: 0':'playerName'}).set_index('playerName')
    res['playerName'] = [playerMD5[x] if x in playerMD5 else x for x in res.index]
    res.set_index('playerName',inplace=True)
    results[game_id] = res.score.fillna(0)

results  = pd.concat(results,1).transpose()
results.rename(columns=playerMD5,inplace=True)

#-- Player Ranking --#
player_ranking = pd.DataFrame(index=results.columns,columns=('games','survive','first','money'))
player_ranking['games']    = results.notnull().sum(0)
player_ranking['survive']  = (results>0).sum(0).sort_values(ascending=False)/player_ranking.games
player_ranking['first']    = results.idxmax(1).value_counts().sort_values(ascending=False)/player_ranking.games
player_ranking['money']    = results.mean(0).sort_values(ascending=False)
