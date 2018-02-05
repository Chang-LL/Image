import cv2
import numpy as np
def showpicloction(img,findimg):
    w=img.shape[1]
    h=img.shape[0]
    fw=findimg.shape[1]
    fh=findimg.shape[0]
    findpt=None
    print("...")
    for nowh in range(0,h-fh):
        for noww in range(0,w-fw):
            comp_tz=img[nowh:nowh+fh,noww:noww+fw,:]-findimg
            if np.sum(comp_tz)<1:
                findpt=noww,nowh
        print('.',end='')
    if findpt!=None:
        cv2.rectangle(img,findpt,(findpt[0]+fw,findpt[1]+fh),(255,0,0))
    return img

fn='F:\\33.png'
fn1='F:\\30.png'
fn2='F:\\31.png'
myimg=cv2.imread(fn)
myimg1=cv2.imread(fn1)
myimg2=cv2.imread(fn2)
myimg=showpicloction(myimg,myimg1)
myimg=showpicloction(myimg,myimg2)
cv2.namedWindow('img2')
cv2.imshow('img2',myimg)
cv2.waitKey()
cv2.destroyAllWindows()