from common import *
import os,glob

filelist  = glob.glob('sim_prob' + os.sep + 'sim_*')
for filename in filelist:


list(zip(*filelist[0].rsplit(']')[0].rsplit('[')[1].split()))


# suit eq:  4 x (13 2)
# order eq: (4 2) x 13
# suit & order neq: (4 2) x (13 2) x 2
