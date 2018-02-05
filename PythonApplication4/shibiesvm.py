import numpy as np
import pylab as pl
import neurolab as nl
import cv2
import mlpy
w_fg=10
h_fg=5
picflag=3
def getresult(simjg):
    jg=[]
    for j in range(0,len(simjg)):
        maxjg=-2
        nowii=0
        for i in range(0,len(simjg[0])):
            if simjg[j][i]>maxjg:
                maxjg=simjg[j][i]
                nowii=i
        jg.append(len(simjg[0])-nowii)
    return jg

def readpic(fn):
    fnimg=cv2.imread(fn)
    img=cv2.resize(fnimg,(400,200))
    w=img.shape[1]
    h=img.shape[0]
    w_interval=int(w/w_fg)
    h_interval=int(h/h_fg)
    alltz=[]
    for nowh in range(0,h,int(h_interval)):
        for noww in range(0,w,int(w_interval)):
            b=img[nowh:nowh+h_interval,noww:noww+w_interval,0]
            g=img[nowh:nowh+h_interval,noww:noww+w_interval,1]
            r=img[nowh:nowh+h_interval,noww:noww+w_interval,2]
            btz=np.mean(b)
            gtz=np.mean(g)
            rtz=np.mean(r)
        alltz.append([btz,gtz,rtz])
    result_alltz=np.array(alltz).T  
    pca = mlpy.PCA()   
    pca.learn(result_alltz)   
    result_alltz = pca.transform(result_alltz)
    #result_alltz = pca.transform(result_alltz, k=len(result_alltz)/2)     
    result_alltz =result_alltz.reshape(-1)  
    return result_alltz

train_x=[]
d=[]
sp_d=[]
sp_d.append([0,0,1])
sp_d.append([0,1,0])
sp_d.append([1,0,0])
for ii in range(1,3):
    smp_x=[]
    mytz=np.zeros((3,w_fg*h_fg))
    for jj in range(1,3):
        fn='F:\\'+str(ii)+'-'+str(jj)+'.jpg'
        pictz=readpic(fn)
        train_x.append(pictz.tolist())
        d.append(ii)
new=list()
for x in train_x:
    z=list()
    for y in x:
        y=int(y)
        z.append(y)
    new.append(z)
myinput=np.array(new)
new=list()
for x in d:
    x=int(x)
    new.append(x)
mytarget=np.array(new)
svm=mlpy.LibSvm(svm_type='c_svc',kernel_type='poly',gamma=50)
svm.learn(myinput,mytarget)
print(svm.pred(myinput))

fn='F:\\ts3.jpg'
testtz=np.array((readpic(fn)))
nowi=svm.pred(testtz)
print(u"%s属于第%d类"%(fn,nowi))

fn='F:\\ts2.jpg'
testtz=np.array((readpic(fn)))
nowi=svm.pred(testtz)
print(u"%s属于第%d类"%(fn,nowi))

fn='F:\\ts1.jpg'
testtz=np.array((readpic(fn)))
nowi=svm.pred(testtz)
print(u"%s属于第%d类"%(fn,nowi))
