import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA,PCA
import seaborn as sns
toydata=pd.read_csv('data_sheet4/toydata.txt',sep=' ',header=None)
toydata.columns=['x','y']
ica=FastICA(n_components=2)
ica_toy=ica.fit(toydata)
pca=PCA(n_components=2)
pca_toy=pca.fit(toydata)
fig=sns.jointplot(x='x',y='y',data=toydata[:200], kind='scatter',label='samples')
for i, (comp, var) in enumerate(zip(pca_toy.components_, pca_toy.explained_variance_)):
    comp = comp * var  # scale component by its variance explanation power
    fig.ax_joint.plot([0, comp[0]], [0, comp[1]], label=f"Component {i} PCA", linewidth=5,
             color=f"C{i + 2}")
for i, comp in enumerate(ica_toy.components_):
    fig.ax_joint.plot([0,comp[0]],[0,comp[1]],label=f"Component {i} ICA", linewidth=5,
             color=f"C{i + 4}")
audiodata=pd.read_csv('data_sheet4/audiodata.txt',sep=' ',header=None)
audiodata.columns=['x','y']
fig.fig.suptitle('Toydata (sample)')
fig.fig.tight_layout()
fig.fig.legend()


toy_transformed_pca=pd.DataFrame(pca_toy.transform(toydata),columns=['x','y'])
toy_transformed_ica=pd.DataFrame(ica_toy.transform(toydata),columns=['x','y'])
g=sns.jointplot(data=toy_transformed_ica,x='x',y='y')
g.fig.suptitle('Projected samples - ICA (Toy)')
g.fig.tight_layout()
g=sns.jointplot(data=toy_transformed_pca,x='x',y='y')
g.fig.suptitle('Projected samples - PCA (Toy)')
g.fig.tight_layout()


ica_audio=ica.fit(audiodata)
pca_audio=pca.fit(audiodata)
ax2=sns.jointplot(data=audiodata[:400],x='x',y='y')
for i, (comp, var) in enumerate(zip(pca_audio.components_, pca_audio.explained_variance_)):
    comp = comp * var  # scale component by its variance explanation power
    ax2.ax_joint.plot([0, comp[0]], [0, comp[1]], label=f"Component {i} PCA", linewidth=5,
             color=f"C{i + 2}")
for i, comp in enumerate(ica_audio.components_):
    ax2.ax_joint.plot([0,comp[0]],[0,comp[1]],label=f"Component {i} ICA", linewidth=5,
             color=f"C{i + 4}")

ax2.fig.suptitle('Audiodata (sample)')
ax2.fig.tight_layout()
ax2.fig.legend()

audio_transformed_pca=pd.DataFrame(pca_audio.transform(audiodata),columns=['x','y'])
audio_transformed_ica=pd.DataFrame(ica_audio.transform(audiodata),columns=['x','y'])
g=sns.jointplot(data=audio_transformed_ica,x='x',y='y')
g.fig.suptitle('Projected samples - ICA (Audio)')
g.fig.tight_layout()
g=sns.jointplot(data=audio_transformed_pca,x='x',y='y')
g.fig.suptitle('Projected samples - PCA (Audio)')
g.fig.tight_layout()

plt.show()