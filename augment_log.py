from common import *

pd.set_option('display.max_rows',120)
pd.set_option('display.max_columns',None)
pd.set_option('display.width',90)

dt      = sys.argv[1] #'20180716'
action  = pd.read_csv('data/action_log_'+dt+'.gz')

t0  = time.clock()
state  = action[[]].copy()
state['N']     = action.N
state['hole']  = action.cards.str.split().apply(pkr_to_cards)
state['board'] = action.board.fillna('').str.split().apply(pkr_to_cards)
print(time.clock() - t0)

MIN_PRWIN_SAMPLES  = 500
for idx,row in state.iterrows():
    if 'Nsim' in action and action.loc[idx,'Nsim']>0: continue
    if pd.isnull(action.loc[idx,'chips_final']): continue
    print(idx,row.N,cards_to_str(row.hole),cards_to_str(row.board),len(action),end=' ... ')
    t0 = time.time()
    if len(row.board) > 0:
        calculate_win_prob_mp_start(row.N,row.hole,row.board,n_jobs=2)
        res  = []
        while len(res) < MIN_PRWIN_SAMPLES:
            time.sleep(0.05)
            res = calculate_win_prob_mp_get()
        calculate_win_prob_mp_stop()
        res  = [x['prWin'] for x in res]
        action.loc[idx,'Nsim']     = len(res)
        action.loc[idx,'prWin']    = np.mean(res)
        action.loc[idx,'prWinStd'] = np.std(res)
    else:
        action.loc[idx,'Nsim'],action.loc[idx,'prWin'],action.loc[idx,'prWinStd'] = read_win_prob(row.N,row.hole)
    print(time.time()-t0)
