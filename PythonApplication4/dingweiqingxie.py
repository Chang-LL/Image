import cv2
import numpy as np

def showpicloction(img,findimg):
    w=img.shape[1]
    h=img.shape[0]
    fw=findimg.shape[1]
    fh=findimg.shape[0]
    findpt=None
    print("...")   
    return findpic(img,findimg,h,fh,w,fw)

def findpic(img,findimg,h,fh,w,fw):
    minds=1e8
    mincb_h=0
    mincb_w=0
    for nowh in range(0,h-fh):
        for noww in range(0,w-fw):
            my_img=img[nowh:nowh+fh,noww:noww+fw,]
            my_findimg=findimg
            dis=get_EuclideanDistance(my_img,my_findimg)
            if dis<minds:
                mincb_h=nowh
                mincb_w=noww
                minds=dis
        print('.',end='')
    findpt=mincb_h,mincb_w
    cv2.rectangle(img,findpt,(findpt[0]+fw,findpt[1]+fh),(0,0,255))
    return img

def addnoise(img):
    count=50000
    for k in range(0,count):
        xi =int(np.random.uniform(0,img.shape[1]))
        xj=int(np.random.uniform(0,img.shape[0]))
        img[xj,xi,0]=255*np.random.rand()
        img[xj,xi,1]=255*np.random.rand()
        img[xj,xi,2]=255*np.random.rand()

def get_EuclideanDistance(x,y):
    myx=np.array(x)
    myy=np.array(y)
    return np.sqrt(np.sum((myx-myy)*(myx-myy)))

fn='F:\\33.png'
fn1='F:\\30.png'
fn2='F:\\31.png'
myimg=cv2.imread(fn)
myimg1=cv2.imread(fn1)
myimg2=cv2.imread(fn2)
#addnoise(myimg)
myimg=showpicloction(myimg,myimg1)
myimg=showpicloction(myimg,myimg2)
cv2.namedWindow('img2')
cv2.imshow('img2',myimg)
cv2.waitKey()
cv2.destroyAllWindows()

