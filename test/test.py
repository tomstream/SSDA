import numpy as np
import cv2
import cv
from matplotlib import pyplot as plt
import os
import h5py
from PIL import Image
import caffe

def patchfy(img, patch_shape,stride):
    img = np.ascontiguousarray(img)  # won't make a copy if not needed
    X, Y = img.shape
    x, y = patch_shape
    shape = ((X-x)/stride+1, (Y-y)/stride+1, x, y) # number of patches, patch_shape
    ret = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            ret[i][j]=img[i*stride:i*stride+x,j*stride:j*stride+y]

    return ret

def unpatchfy(imgs, shape, stride):
    x,y,X,Y = imgs.shape
    index = np.zeros(shape)
    unpatch = np.zeros(shape)
    patchIndex = np.zeros((X,Y))+1
    for i in range(x):
        for j in range(y):
           index[i*stride:i*stride+X,j*stride:j*stride+Y]+=patchIndex
           unpatch[i*stride:i*stride+X,j*stride:j*stride+Y]+=imgs[i,j]
    unpatch/=index
    return unpatch

def generateTmpTestFile(num):
    filein = open('da1_template.prototxt')
    fileout = open('da1.prototxt','w')
    fileout.write('name : "DA"\ninput: "data"\ninput_dim:' +str(num)+'\ninput_dim: 1\ninput_dim: 10\ninput_dim: 10\n\n')
    fileout.write(filein.read())
    fileout.close()
    filein.close()

    filein = open('sda_template.prototxt')
    fileout = open('sda.prototxt','w')
    fileout.write('name : "SDA"\ninput: "data"\ninput_dim:'+str(num)+'\ninput_dim: 1\ninput_dim: 1\ninput_dim: 100\n\n')
    fileout.write(filein.read())
    fileout.close()
    filein.close()

    filein = open('da2_template.prototxt')
    fileout = open('da2.prototxt','w')
    fileout.write('name : "DA"\ninput: "sigmoid1"\ninput_dim:'+str(num)+'\ninput_dim: 1\ninput_dim: 1\ninput_dim: 100\n')
    fileout.write(filein.read())
    fileout.close()
    filein.close()


def testForNet(img,patchshape,stride):
    patch=patchfy(img,patchshape,stride)
    x,y,X,Y=patch.shape
    patch = patch.reshape((x*y,1,X,Y))
    generateTmpTestFile(x*y)
    caffe.set_mode_cpu()
    net = caffe.Net('da1.prototxt','da_iter_10000.caffemodel',caffe.TEST)
    net.blobs['data'].reshape(*patch.shape)
    net.blobs['data'].data[...]=patch
    tmp1 = net.forward()['sigmoid1']
    print tmp1.shape

    net = caffe.Net('sda.prototxt','sda_iter_5000.caffemodel',caffe.TEST)
    net.blobs['data'].reshape(*tmp1.shape)
    net.blobs['data'].data[...] = tmp1
    tmp2 = net.forward()['sigmoid32']

    net = caffe.Net('da2.prototxt','da_iter_10000.caffemodel',caffe.TEST)
    net.blobs['sigmoid1'].reshape(*tmp2.shape)
    net.blobs['sigmoid1'].data[...]=tmp2
    tmp3 =  net.forward()['sigmoid2']

    tmp3=tmp3.reshape((x,y,X,Y))
    imgout=unpatchfy(tmp3,img.shape,stride)
    print imgout,img
    return imgout

img=np.array(Image.open('275520.jpg'))
img=img/255.
out = testForNet(img,(10,10),3)
out = out.reshape((out.shape[0],out.shape[1]))
cv2.imwrite('out.png',(out)*255)
