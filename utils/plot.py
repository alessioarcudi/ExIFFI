import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.pyplot import cm
from matplotlib.pyplot import *

from DIFFI_master.interpretability_module import local_diffi



def plt_bars(Importances, imp_alg, name, dim, f = 6):
    number_colours = 20
    color = plt.cm.get_cmap('tab20',number_colours).colors
    patterns = [None, "/" , "\\" , "|" , "-" , "+" , "x", "o", "O", ".", "*" ]
    by=0
    importances_matrix = np.c_[[np.array(x.sort_values(by = [by],ascending = False).index).T for x in Importances[imp_alg]["importances"]]]
    bars = [[(list(importances_matrix[:,j]).count(i)/len(importances_matrix))*100 for i in range(dim)] for j in range(dim)]
    
    bars = pd.DataFrame(bars)
                    
    barWidth = 0.85
    r=range(dim)

    for i in range(dim):
        plt.bar(r[:6], bars.T.iloc[i,:f].values, bottom=bars.T.iloc[:i,:f].sum().values ,color=color[i%number_colours], edgecolor='white', width=barWidth, label=str(i), hatch=patterns[i//number_colours])

    if imp_alg=="E-diffi" or imp_alg=="E-DIFFI":
        imp_alg="ExIFFI"
    elif imp_alg=="diffi":
        imp_alg="DIFFI"
    
    name = name.capitalize()
    plt.xlabel("rank", fontsize = 14)
    plt.xticks(range(6),[r'$1^{st}$', r'$2^{nd}$', r'$3^{rd}$', r'$4^{th}$', r'$5^{th}$', r'$6^{th}$', r'$7^{th}$', r'$8^{th}$', r'$9^{th}$'][:f])
    plt.ylabel("Percentage count", fontsize = 14)
    plt.yticks(range(10,101,10),[str(x)+"%" for x in range(10,101,10)])
    if dim >6:
        plt.title("{} {} \n percentage count of feature ranking \n in the first {} position".format(name,imp_alg,f), fontsize = 18)
    else:
        plt.title("{} {} \n percentage count of feature ranking".format(name,imp_alg,f), fontsize = 18)        
    plt.legend(bbox_to_anchor = (1.05,0.95),loc="upper left",ncol=2)
    plt.savefig('./results/compare_features/images/'+imp_alg+'_'+name+'.pdf',bbox_inches = "tight")
    plt.show()
    

def plot_importance_map(name,model, X_train, y_train, risoluzione,pwd,save=True,axes=True,m=None,factor=3,feats_plot = (0,1), plt=plt,isdiffi=False):
    mins = X_train.min(axis=0)[list(feats_plot)]
    maxs = X_train.max(axis=0)[list(feats_plot)]  
    mean = X_train.mean(axis = 0)
    mins = list(mins-(maxs-mins)*factor/10)
    maxs = list(maxs+(maxs-mins)*factor/10)
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], risoluzione), np.linspace(mins[1], maxs[1], risoluzione))
    mean = np.repeat(np.expand_dims(mean,0),len(xx)**2,axis = 0)
    mean[:,feats_plot[0]]=xx.reshape(len(xx)**2)
    mean[:,feats_plot[1]]=yy.reshape(len(yy)**2)

    importance_matrix = np.zeros_like(mean)
    if isdiffi:
        model.max_samples = len(X_train)
        for i in range(importance_matrix.shape[0]):
            importance_matrix[i] = local_diffi(model, mean[i])[0]
    else:
        importance_matrix = model.Local_importances(mean, True, False)
    
    sign = np.sign(importance_matrix[:,feats_plot[0]]-importance_matrix[:,feats_plot[1]])
    Score = sign*((sign>0)*importance_matrix[:,feats_plot[0]]+(sign<0)*importance_matrix[:,feats_plot[1]])
    x = X_train[:,feats_plot[0]].squeeze()
    y = X_train[:,feats_plot[1]].squeeze()
    
    Score = Score.reshape(xx.shape)

    if m is not None:
        cp = plt.pcolor(xx, yy, Score, cmap=cm.RdBu, vmin=-m, vmax=m, shading='nearest')
    else:
        cp = plt.pcolor(xx, yy, Score , cmap=cm.RdBu, shading='nearest',norm=colors.CenteredNorm())
    plt.contour(xx, yy, (importance_matrix[:,feats_plot[0]]+importance_matrix[:,feats_plot[1]]).reshape(xx.shape),levels = 7, cmap=cm.Greys,alpha=0.7)
    plt.scatter(x[y_train==0],y[y_train==0],s=40,c="tab:blue",marker="o",edgecolors="k",label="inliers")
    plt.scatter(x[y_train==1],y[y_train==1],s=60,c="tab:orange",marker="*",edgecolors="k",label="outliers")
    plt.legend()
    if save:
        plt.savefig(pwd+'\\results\\davide\\Local Scoremaps Plot\\Local_Importance_Scoremap_{}.pdf'
                .format(name),bbox_inches='tight')
    #plt.colorbar(cp,label="difference of feature importance between {} {}".format(feats_plot[0],feats_plot[1]), extend='max')
    #if axes: 
        #plt.xlabel("Feature " + str(feats_plot[0]),fontsize = 20)
        #plt.ylabel("Feature " + str(feats_plot[1]),fontsize = 20)
        #plt.title("Importance scoremap "+name+' '+isdiffi*"DIFFI"+(1-isdiffi)*"ExIFFI",fontsize = 18)

def plot_importance_map_2(EDIFFI, X_train, y_train, risoluzione, m=None,feats_plot = (0,1), plt=plt):
    mins = X_train.min(axis=0)[list(feats_plot)]
    maxs = X_train.max(axis=0)[list(feats_plot)]  
    mean = X_train.mean(axis = 0)
    mins = list(mins-(maxs-mins)*3/10)
    maxs = list(maxs+(maxs-mins)*3/10)
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], risoluzione), np.linspace(mins[1], maxs[1], risoluzione))
    mean = np.repeat(np.expand_dims(mean,0),len(xx)**2,axis = 0)
    mean[:,feats_plot[0]]=xx.reshape(len(xx)**2)
    mean[:,feats_plot[1]]=yy.reshape(len(yy)**2)
    
    importance_matrix = EDIFFI.Local_importances(mean, True, False)

    x = X_train[:,feats_plot[0]].squeeze()
    y = X_train[:,feats_plot[1]].squeeze()
    
    R=importance_matrix[:,feats_plot[0]].reshape(xx.shape)
    B=importance_matrix[:,feats_plot[1]].reshape(xx.shape)
    B=(B-np.min(B))/(m-np.min(B))
    B[B>1]=1
    R=(R-np.min(R))/(m-np.min(R))
    R[R>1]=1

    def screen(B,x,y,z):
        return np.array([B*x,B*y,B*z])

    R_screen = screen(R,0.88,0.88,0.50)
    B_screen = screen(B,0.50,0.88,0.88)

    image = 1-(R_screen**2+B_screen**2)/1.2

    plt.imshow(image.transpose(1,2,0), extent=[mins[0],maxs[0],mins[1],maxs[1]], origin='lower')
    plt.scatter(x,y,s=15,alpha=(y_train+1)/2,c=y_train,cmap=plt.cm.PuOr)
    plt.show()
    

def print_score_map(I,X,risoluzione):
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    mins = list(mins-(maxs-mins)*3/10)
    maxs = list(maxs+(maxs-mins)*3/10)
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], risoluzione), np.linspace(mins[1], maxs[1], risoluzione))

    S1 = I.Anomaly_Score(X_in=np.c_[xx.ravel(), yy.ravel()])
    S1 = S1.reshape(xx.shape)
    x= X.T[0]
    y= X.T[1]        

    plt.figure(figsize=(12,12)) 
    levels = np.linspace(np.min(S1),np.max(S1),10)
    CS = plt.contourf(xx, yy, S1, levels, cmap=plt.cm.YlOrRd)
    plt.scatter(x,y,s=15,c='None',edgecolor='k')
    plt.axis("equal")
    plt.show()
    return
