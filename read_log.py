import os,sys,glob
import pandas as pd
import json

logs  = []
for f in glob.glob('user_logs_20180716.log/*'):
    log  = pd.read_csv(f,sep='>>>',header=None,names=(0,1,2),engine='python')
    log.dropna(subset=[2],inplace=True)
    if len(log) > 0:
        log  = log[log[1].str.split().str[0]=='event']
        log['timestamp']  = log[0].str.split(n=1).str[0].str.strip().str[1:-1]
        logs.append(log)

logs  = pd.concat(logs,0)
