from common import *
import os,glob

filelist  = glob.glob('sim_prob' + os.sep + 'sim_*')
for filename in filelist:


list(zip(*filelist[0].rsplit(']')[0].rsplit('[')[1].split()))
