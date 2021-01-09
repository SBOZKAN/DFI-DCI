import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as LA

# Inputs amber covariance matrices. Use mass-weighted covariance matrix
def parse_covar(files):
    f=open(files,'r')
    data=f.read()
    data=data.split("\n")
    del data[-1]
    f.close()
    covar=[]
    for f in data:
        covar.append(np.asarray(f.split(),float))
    return np.asarray(covar,float)

def calcperturbMat(cov,resnum):
    directions = np.vstack(([1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]))
    normL = np.apply_along_axis(np.linalg.norm, 1, directions)
    direct=directions/normL[:,None]
    perturbMat = np.zeros((resnum,resnum))
    for k in range(len(direct)):
        peturbDir = direct[k,:]
        for j in range(int(resnum)):
            delforce = np.zeros(3*resnum)
            delforce[3*j:3*j+3] = peturbDir 
            delXperbVex = np.dot(cov,delforce)
            delXperbMat = delXperbVex.reshape((resnum,3))
            delRperbVec = np.sqrt(np.sum(delXperbMat*delXperbMat,axis=1))
            perturbMat[:,j] += delRperbVec[:]
    perturbMat /= 7
    nrmlperturbMat = perturbMat/np.sum(perturbMat)
    return nrmlperturbMat

def get_dfi(nrmlperturbMat):
    dfi = np.sum(nrmlperturbMat,axis=1)
    return dfi

def get_dci(pos,nrmlperturbMat):
    dci=[]
    for p in pos:
        dci.append(nrmlperturbMat[:,p])
    dci=(np.sum(dci,axis=0)/len(pos))/((np.sum(nrmlperturbMat,axis=1)/len(nrmlperturbMat)))
    return dci

cov=sys.argv[1]
fname=sys.argv[2]

cov=parse_covar(files)
resnum=int(len(cov)/3)
pert_mat=calcperturbMat(cov,resnum)
dfi=get_dfi(pert_mat)
data=pd.DataFrame()
data["Renum_Res"]=list(range(1,resnum+1))
data["dfi"]=dfi

if len(sys.argv == 4):
    dci_pos=sys.argv[3]
    dci=get_dci(dci_pos,pert_mat)
    data["dci"]=dci

data.to_csv(fname,index=False)

# Usage: dfi_dci.py <cov> <fname> <pert res>