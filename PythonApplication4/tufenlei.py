import numpy as np
import cv2
import mlpy
w_fg=20
h_fg=15
picflag=3

def readpicPCA(fn):
    fnimg=cv2.imread(fn)
    img=cv2.resize(fnimg,(500,400))
    w=img.shape[1]
    h=img.shape[0]
    w_interval=int(w/20)
    h_interval=int(h/10)
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
    pca=mlpy.PCA()
    pca.learn(result_alltz)
    result_alltz=pca.transform(result_alltz,k=len(result_alltz)/2)
    result_alltz=result_alltz.reshape(len(result_alltz))
    return result_alltz
def readpic(fn):
    fnimg=cv2.imread(fn)
    img=cv2.resize(fnimg,(800,600))
    w=img.shape[1]
    h=img.shape[0]
    w_interval=int(w/w_fg)
    h_interval=int(h/h_fg)
    alltz=[]
    alltz.append([])
    alltz.append([])
    alltz.append([])
    for nowh in range(0,h,int(h_interval)):
        for noww in range(0,w,int(w_interval)):
            b=img[nowh:nowh+h_interval,noww:noww+w_interval,0]
            g=img[nowh:nowh+h_interval,noww:noww+w_interval,1]
            r=img[nowh:nowh+h_interval,noww:noww+w_interval,2]
            btz=np.mean(b)
            gtz=np.mean(g)
            rtz=np.mean(r)
            alltz[0].append(btz)
            alltz[1].append(gtz)
            alltz[2].append(rtz)
    return alltz

def get_cossimi(x,y):
    myx=np.array(x)
    myy=np.array(y)
    cos1=np.sum(myx*myy)
    cos21=np.sqrt(sum(myx*myx))
    cos22=np.sqrt(sum(myy*myy))
    return cos1/float(cos21*cos22)

train_x=[]
d=[]

for ii in range(1,picflag+1):
    smp_x=[]
    b_tz=np.array([0,0,0])
    g_tz=np.array([0,0,0])
    r_tz=np.array([0,0,0])
    mytz=np.zeros((3,w_fg*h_fg))
    for jj in range(1,3):
        fn='F:\\'+str(ii)+'-'+str(jj)+'.jpg'
        tmptz=readpic(fn)
        mytz+=np.array(tmptz)
    mytz/=3
    train_x.append(mytz[0].tolist()+mytz[1].tolist()+mytz[2].tolist())

fn='F:\\ts1.jpg'
testtz=np.array((readpic(fn)))
simtz=testtz[0].tolist()+testtz[1].tolist()+testtz[2].tolist()
maxtz=0
nowi=0
for i in range(0,picflag):
    nowsim=get_cossimi(train_x[i],simtz)
    if nowsim>maxtz:
        maxtz=nowsim
        nowi=i
print(u"%s属于第%d类"%(fn,nowi+1))

fn='F:\\ts2.jpg'
testtz=np.array((readpic(fn)))
simtz=testtz[0].tolist()+testtz[1].tolist()+testtz[2].tolist()
maxtz=0
nowi=0
for i in range(0,picflag):
    nowsim=get_cossimi(train_x[i],simtz)
    if nowsim>maxtz:
        maxtz=nowsim
        nowi=i
print(u"%s属于第%d类"%(fn,nowi+1))

fn='F:\\ts3.jpg'
testtz=np.array((readpic(fn)))
simtz=testtz[0].tolist()+testtz[1].tolist()+testtz[2].tolist()
maxtz=0
nowi=0
for i in range(0,picflag):
    nowsim=get_cossimi(train_x[i],simtz)
    if nowsim>maxtz:
        maxtz=nowsim
        nowi=i
print(u"%s属于第%d类"%(fn,nowi+1))