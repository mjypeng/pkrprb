from agent_common import *
import os,glob

playerNames = ['jyp','jyp0','jyp1','jyp2','jyp3','jyp4','jyp5','twice']
playerMD5   = {}

for playerName in playerNames:
    m    = hashlib.md5()
    m.update(playerName.encode('utf8'))
    playerMD5[m.hexdigest()] = playerName

batch_name = 'basic_1+twice'
filelist = glob.glob('game_records' + os.sep + batch_name + os.sep + 'game_' + '?'*14 + '.csv')

results  = {}
for filename in filelist:
    game_id = filename.rsplit('.',1)[0].rsplit('_',1)[1]
    res     = pd.read_csv(filename,index_col='playerName')
    results[game_id] = res.score.fillna(0)
    #
    #-- basic_0 format --#
    # game_id = filename.rsplit('.',1)[0].rsplit('_',1)[1]
    # res     = pd.read_csv(filename).rename(columns={'Unnamed: 0':'name'}).set_index('name')
    # results[game_id] = res.amt.fillna(0)

results  = pd.concat(results,1).transpose()
results.rename(columns=playerMD5,inplace=True)

# results  = results[results.twice.notnull()]

#-- Player Ranking --#
player_ranking = pd.DataFrame(index=results.columns,columns=('survive','first','money'))
player_ranking['survive']  = (results>0).mean(0).sort_values(ascending=False)
player_ranking['first']    = results.idxmax(1).value_counts().sort_values(ascending=False)/len(results)
player_ranking['money']    = results.mean(0).sort_values(ascending=False)
