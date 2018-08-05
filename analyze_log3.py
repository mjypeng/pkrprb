from common import *
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.externals import joblib

pd.set_option('display.max_rows',120)
pd.set_option('display.max_columns',None)
pd.set_option('display.width',80)

#-- Maximum Bet/Chips by prWin Simulation --#
def sigmoid(w,alpha=12,beta=0.5):
    maxbet_chips  = 1/(1 + np.exp(-alpha*(w - beta)))
    return np.minimum(np.maximum(maxbet_chips,0),1).round(2)

#-- Read Data --#
DATE_START  = '20180723'
DATE_END    = '20180802'
dt_range    = pd.date_range(DATE_START,DATE_END,freq='B').strftime('%Y%m%d')
rnd     = pd.concat([pd.read_csv('data/round_log_'+dt+'.gz') for dt in dt_range],0,ignore_index=True,sort=False)
action  = pd.concat([pd.read_csv('data/action_proc_'+dt+'.gz') for dt in dt_range],0,ignore_index=True,sort=False)
action['timestamp']  = pd.to_datetime(action.timestamp)

#-- Round Hand Score Lookup Table --#
rnd.dropna(subset=['round_id','cards'],inplace=True)
rnd.drop_duplicates(subset=['round_id','playerName','cards','board'],inplace=True)
rnd.set_index(['round_id','playerName'],inplace=True)
score_lut  = pd.concat([rnd.score_Deal,rnd.score_Flop,rnd.score_Turn,rnd.score_River],0,keys=('Deal','Flop','Turn','River',)).apply(eval)

t0  = time.clock()
action['opponents']  = action.opponents.apply(eval)
action['op_playerName']  = action.opponents.apply(lambda x:[y['playerName'] for y in x])
print(time.clock() - t0)

action['win']  = action.apply(lambda x:score_lut.loc[x.roundName,x.round_id,x.playerName]>=max([score_lut.loc[(x.roundName,x.round_id,y)] for y in x.op_playerName]) if len(x.op_playerName) else True,axis=1)
action['winRiver']  = action.apply(lambda x:score_lut.loc['River',x.round_id,x.playerName]>=max([score_lut.loc[('River',x.round_id,y)] for y in x.op_playerName]) if len(x.op_playerName) else True,axis=1)

#-- Bet/Fold Lookup Table --#
action['fold']     = action.action=='fold'
action['totalBet'] = action.bet + action.amount
bet_lut  = action.groupby(['round_id','roundName','playerName'])[['fold','totalBet']].max().reset_index('playerName')

exit(0)

def sim_betting(Nsamp,N0,roundName,alpha,beta):
    results  = pd.DataFrame()
    for j in range(Nsamp):
        print(j)
        t0  = time.clock()
        smallBlind = 10
        SB_idx     = 8
        BB_idx     = 9
        SB_count   = 0
        players    = pd.DataFrame({'chips':3000,'pos':'','bet':0,'isSurvive':True},index=['p%d'%i for i in range(N0)])
        progress   = []
        while True:
            players['pos']  = ''
            players['bet']  = 0
            #
            if SB_count >= players.isSurvive.sum() and smallBlind < 640:
                smallBlind  *= 2
                SB_count     = 0
            #
            players.loc[players.index[SB_idx],'pos']    = 'SB'
            players.loc[players.index[SB_idx],'bet']    = min(smallBlind,players.loc[players.index[SB_idx],'chips'])
            players.loc[players.index[SB_idx],'chips'] -= players.loc[players.index[SB_idx],'bet']
            players.loc[players.index[BB_idx],'pos']    = 'BB'
            players.loc[players.index[BB_idx],'bet']    = min(2*smallBlind,players.loc[players.index[BB_idx],'chips'])
            players.loc[players.index[BB_idx],'chips'] -= players.loc[players.index[BB_idx],'bet']
            # print(players)
            # print()
            # time.sleep(1)
            #
            N     = players.isSurvive.sum()
            mask  = (action.N==N) & (action.roundName==roundName) & (action.prev_action=='none')
            samp  = action.loc[mask,['round_id','playerName','N','cards','prWin','winRiver']].sample().iloc[0]
            max_bet_chips  = sigmoid(samp.prWin,alpha,beta)
            bet   = int(players.loc['p0','chips']*max_bet_chips)
            bet   = max(bet,2*smallBlind) if bet>0 else 0
            print(players.loc[['p0']],'\t',N,smallBlind,samp.prWin)
            #
            temp  = bet_lut[bet_lut.playerName!=samp.playerName].loc[(samp.round_id,roundName)]
            Nnf   = N - temp.fold.sum()
            maxBet = min(temp.totalBet.max(),int(players.chips.median()))
            #
            op_idx = players[(players.index!='p0')&(players.isSurvive)].index.values
            op_nf  = np.random.choice(op_idx,Nnf - 1,replace=False)
            if bet:
                # print('p0 bet')
                bet1  = min(bet - players.loc['p0','bet'],players.loc['p0','chips'])
                players.loc['p0','bet']   += bet1
                players.loc['p0','chips'] -= bet1
                if Nnf == 1:
                    # print(players)
                    # print()
                    # time.sleep(1)
                    # print('Everybody else folded')
                    players.loc['p0','chips']  += players.bet.sum()
                else:
                    bet1  = np.minimum(bet - players.loc[op_nf,'bet'],players.loc[op_nf,'chips'])
                    players.loc[op_nf,'bet']   += bet1
                    players.loc[op_nf,'chips'] -= bet1
                    # print(players)
                    # print()
                    # time.sleep(1)
                    # print('Showdown')
                    win_idx  = 'p0' if samp.winRiver else np.random.choice(op_nf)
                    players.loc[win_idx,'chips'] += players.bet.sum()
            else:
                # print('p0 Fold')
                if Nnf == 1:
                    # print('Everybody else also folded')
                    players['chips']  += players.bet
                else:
                    # print('Randomly pick someone to receive the blinds')
                    win_idx  = np.random.choice(op_nf)
                    players.loc[win_idx,'chips'] += players.bet.sum()
            #
            players['bet']  = 0
            players.loc[players.chips==0,'isSurvive']  = False
            # print(players)
            # print()
            # time.sleep(1)
            #
            progress.append(players.loc['p0','chips'])
            #
            SB_count  += 1
            SB_idx  = (SB_idx + 1) % len(players)
            while not players.iloc[SB_idx].isSurvive:
                SB_idx  = (SB_idx + 1) % len(players)
            BB_idx  = (SB_idx + 1) % len(players)
            while not players.iloc[BB_idx].isSurvive:
                BB_idx  = (BB_idx + 1) % len(players)
            #
            if players.loc['p0','chips'] == 0 or players.loc['p0','chips'] == players.chips.sum() or players.isSurvive.sum()<np.ceil((len(players)+1)/2):
                break
        #
        print(time.clock() - t0)
        progress  = np.asarray(progress)
        results.loc[j,'alpha'] = alpha
        results.loc[j,'beta']  = beta
        results.loc[j,'chips'] = progress[-1]
        results.loc[j,'roundCount']  = len(progress)
        results.loc[j,'mean']  = np.mean(progress)
        results.loc[j,'std']   = np.std(progress)
        results.loc[j,'min']   = min(progress)
        results.loc[j,'max']   = max(progress)
    return results

#-- Simulation --#
Nsamp      = 100
N0         = 10     # Number of players
roundName  = 'Deal'
alpha_range = range(18,30,3) #range(6,30,3)
beta_range = np.arange(0.3,0.71,0.1) #np.arange(0.3,0.71,0.05)
results    = []
for alpha in alpha_range:
    for beta in beta_range:
        # ww  = np.arange(0.05,1,0.05)
        # print(pd.Series(sigmoid(ww,alpha,beta),index=ww))
        print(alpha,beta)
        if alpha==18 and beta==0.3: continue
        results.append(sim_betting(Nsamp,N0,roundName,alpha,beta))

results  = pd.concat(results,ignore_index=True)
