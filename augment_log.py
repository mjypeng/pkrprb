from common import *

if __name__ == '__main__':
    pd.set_option('display.max_rows',120)
    pd.set_option('display.max_columns',None)
    pd.set_option('display.width',90)

    dt      = sys.argv[1] #'20180716'
    action  = pd.read_csv('data/action_log_'+dt+'.gz')

    t0  = time.clock()
    hole   = action.cards.str.split().apply(sorted).str.join('')
    board  = action.board.fillna('').str.split().apply(lambda x: sorted(x[:3]) + x[3:]).str.join('')
    action['hashkey'] = action.N.astype(str) + '_' + hole + '_' + board
    print(time.clock() - t0)

    state  = pd.read_csv('precal_win_prob_temp.gz',index_col='hashkey')
    new_state  = action[action.winMoney.notnull()].hashkey.unique()
    MIN_PRWIN_SAMPLES  = 500
    for i,hashkey in enumerate(new_state):
        # temp  = action[(action.hashkey==hashkey)&(action.Nsim.notnull())]
        # if len(temp) > 0:
        #     state.loc[hashkey,'Nsim']      = temp.iloc[0].Nsim
        #     state.loc[hashkey,'prWin']     = temp.iloc[0].prWin
        #     state.loc[hashkey,'prWinStd']  = temp.iloc[0].prWinStd
        if 'Nsim' in state and hashkey in state.index and state.loc[hashkey,'Nsim']>0:
            continue
        print(i,len(new_state),hashkey,end=' ... ')
        t0  = time.time()
        temp = hashkey.split('_')
        N    = int(temp[0])
        hole = pkr_to_cards([temp[1][:2],temp[1][2:]])
        board = pkr_to_cards([temp[2][j:j+2] for j in range(0,len(temp[2]),2)])
        if len(board) > 0:
            calculate_win_prob_mp_start(N,hole,board,n_jobs=2)
            res  = []
            while len(res) < MIN_PRWIN_SAMPLES:
                time.sleep(0.05)
                res = calculate_win_prob_mp_get()
            calculate_win_prob_mp_stop()
            res  = [x['prWin'] for x in res]
            state.loc[hashkey,'Nsim']     = len(res)
            state.loc[hashkey,'prWin']    = np.mean(res)
            state.loc[hashkey,'prWinStd'] = np.std(res)
        else:
            state.loc[hashkey,'Nsim'],state.loc[hashkey,'prWin'],state.loc[hashkey,'prWinStd'] = read_win_prob(N,hole)
        print(time.time()-t0)

    state.to_csv('precal_win_prob_temp.gz',compression='gzip')
