import os,sys
import pandas as pd
import json

log  = pd.read_csv(sys.argv[1],sep='>>>',header=None,engine='python')
log.dropna(subset=[2],inplace=True)
log  = log[log[1].str.split().str[0]=='event']
log['timestamp']  = log[0].str.split(n=1).str[0].str.strip().str[1:-1]
