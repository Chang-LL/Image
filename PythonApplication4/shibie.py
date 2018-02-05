import numpy as np
import pylab as pl
import neurolab as nl
import cv2
import mlpy

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
    for jj in range(1,3):
        fn='F:\\'+str(ii)+'-'+str(jj)+'.jpg'
        pictz=readpic(fn)
        train_x.append(pictz)
        d.append(sp_d[ii-1])
myinput=np.array(train_x)
mytarget=np.array(d)
mymax=np.max(myinput)
netminmax=[]
for i in range(0,len(myinput[0])):
    netminmax.append([0,mymax])

print(u"\n正在建立神经网络")
bpnet=nl.net.newff(netminmax,[5,3])

print(u"\n训练神经网络中")
err=bpnet.train(myinput,mytarget,epochs=800,show=5,goal=0.2)
if err[len(err)-1]>0.4:
    print(u"\n神经网络训练失败")
else:
    print(u"\n神经网络训练完毕")
    pl.subplot(111)
    pl.plot(err)
    pl.xlabel("Epoch number")
    pl.ylabel("error (default SSE")
    print(u"对样本进行测试")
    simd=bpnet.sim(myinput)
    mysimd=getresult(simd)
    print (mysimd)
    print(u"进行仿真")

    testpictz=np.array([readpic('F:\\ts3.jpg')])
    simtest=bpnet.sim(testpictz)
    mysimtest=getresult(simtest)
    print(u"ts3.jpg")
    print(simtest)
    print(mysimtest)

    testpictz=np.array([readpic('F:\\ts2.jpg')])
    simtest=bpnet.sim(testpictz)
    mysimtest=getresult(simtest)
    print(u"ts2.jpg")
    print(simtest)
    print(mysimtest)

    testpictz=np.array([readpic('F:\\ts1.jpg')])
    simtest=bpnet.sim(testpictz)
    mysimtest=getresult(simtest)
    print(u"ts1.jpg")
    print(simtest)
    print(mysimtest)

    testpictz=np.array([readpic('F:\\ts1.jpg')])
    simtest=bpnet.sim(testpictz)
    mysimtest=getresult(simtest)
    print(u"ts1.jpg")
    print(simtest)
    print(mysimtest)

    pl.show()