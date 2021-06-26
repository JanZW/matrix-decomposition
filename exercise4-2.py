import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
import random
from PIL import Image
from sklearn.decomposition import NMF, PCA, FastICA
import pickle

def plot_gallery(title, images, cmap='gray'):
    n_col=4
    n_row=5
    image_shape=(64,64)
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=cmap,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)


files=glob.glob('lfwcrop_grey/faces/*.pgm')
selected_files=random.sample(files,1000)
fig,ax=plt.subplots(4,5)
ax=ax.ravel()
for i,f in enumerate(selected_files[:20]):
    img=Image.open(f).convert('L')
    ax[i].imshow(img,cmap='gray')


data=[]
for f in selected_files:
    img=np.array(Image.open(f).convert('L'))
    data.append(img.flatten())
data=pd.DataFrame(data)


avg=data.mean(axis=0).values.reshape(64,64)
avg_img=Image.fromarray(avg)


fig2,ax2=plt.subplots(1,1)
ax2.imshow(avg_img,cmap='gray')


mode=input('Train? (t)')
if mode=='t':
    model_nmf=NMF(n_components=20,max_iter=20000,init='nndsvda', tol=5e-3)
    W_nmf=model_nmf.fit_transform(data)
    pickle.dump(model_nmf,open('trained-model-nmf.sav','wb'))
    model_pca=PCA(n_components=20,whiten=True,svd_solver='randomized')
    model_pca.fit(data)
    pickle.dump(model_pca,open('trained-model-pca.sav','wb'))
    model_ica=FastICA(n_components=20,whiten=True)
    model_ica.fit(data)
    pickle.dump(model_ica,open('trained-model-ica.sav','wb'))
else:
    model_nmf=pickle.load(open('trained-model-nmf.sav','rb'))
    model_pca=pickle.load(open('trained-model-pca.sav','rb'))
    model_ica=pickle.load(open('trained-model-ica.sav','rb'))
H_nmf=model_nmf.components_
H_pca=model_pca.components_
H_ica=model_ica.components_

plot_gallery('NMF',H_nmf)
plot_gallery('PCA',H_pca)
plot_gallery('ICA',H_ica)

plt.show()