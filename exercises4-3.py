import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import FastICA
from PyEMD import EMD, EEMD

df=pd.read_csv('data_sheet4/ex3_signals.txt',sep=' ')
df.columns=['x','y']
fig,ax=plt.subplots(2,1,sharex=True)
ax[0].plot(df["x"][:500])
ax[0].set_title("Signal 1")
ax[1].plot(df['y'][:500])
ax[1].set_title("Signal 2")
fig.tight_layout()

model=FastICA(n_components=2)
X=pd.DataFrame(model.fit_transform(df),columns=["a","b"])
fig,ax=plt.subplots(2,1)
fig.suptitle("ICA Components")
ax[0].plot(X["a"][:500])
ax[0].set_title("Component A")
ax[1].plot(X["b"][:500])
ax[1].set_title("Component B")
fig.tight_layout()

emd=EMD()
t=np.linspace(0,1,len(df))
dat=df['x'].values
X=emd.emd(dat,t)

fig,ax=plt.subplots(6,1)
ax=ax.ravel()
for i,k in enumerate(X):
    ax[i].plot(k[:500])
fig.suptitle("EMD of Signal 1")

dat+=np.random.normal(scale=0.1,size=len(dat))
X=emd.emd(dat,t)
fig2,ax2=plt.subplots(len(X),1)
for i,k in enumerate(X):
    ax2[i].plot(k[:500])
fig2.suptitle("EMD of Signal 1 with noise")

noise_widths=[0.02,0.05,0.1,0.5,1,2]
for noise_width in noise_widths:
    eemd=EEMD(noise_width=noise_width)
    eIMFs=eemd.eemd(df['x'].values,t)
    nIMFs=eIMFs.shape[0]

    fig3,ax3=plt.subplots(nIMFs,1)
    for i,k in enumerate(eIMFs):
        ax3[i].plot(k[:500])
    fig3.suptitle(f"EEMD of Signal 1 with noise_width={noise_width}")



plt.show()