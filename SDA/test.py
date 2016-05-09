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
    shape = ((X-x+1)/stride, (Y-y+1)/stride, x, y) # number of patches, patch_shape
    # The right strides can be thought by:
    # 1) Thinking of `img` as a chunk of memory in C order
    # 2) Asking how many items through that chunk of memory are needed when indices
    #    i,j,k,l are incremented by one
    strides = img.itemsize*np.array([Y, 1, Y, 1])

    return np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)

def unpatchfy(imgs, shape, stride):
    x,y,X,Y = imgs.shape
    index = np.zeros(shape)
    unpatch = np.zeros(shape)
    patchIndex = np.zeros((X,Y))+1
    for i in range(x):
        for j in range(y):
            index[i:i+X,j:j+Y]+=patchIndex
            unpatch[i:i+X,j:j+Y]+=imgs[i,j]
    unpatch/=index
    return unpatch

def generateTmpTestFile(num):
    filein = open('da1_template.prototxt')
    fileout = open('da1.prototxt')
    fileout.write('name : "DA1"\ninput: "data"\ninput_dim:' +str(num)+'\ninput_dim: 1\ninput_dim: 10\ninput_dim: 10\ninput: "label"\ninput_dim:'+str(num)+'input_dim: 1\ninput_dim: 10\ninput_dim: 10\n\n\n')
    
    fileout.write(filein.read())
    fileout.close()
    filein.close()

    filein = open('sda_template.prototxt')
    fileout = open('sda.prototxt')
    fileout.write('name : "SDA1"\ninput: "data"\ninput_dim:'+str(num)+'input_dim: 100\ninput: "label"\ninput_dim: '+str(num)+'\ninput_dim: 100\n\n')
    fileout.write(filein.read())
    fileout.close()
    filein.close()

    filein = open('da2_template.prototxt')
    fileout = open('da2.prototxt')
    fileout.write('name : "DA2"\ninput: "sigmoid1"\ninput_dim:'+str(num)+'input_dim: 100\n\n')
    fileout.write(filein.read())
    fileout.close()
    filein.close()


def testForNet(img,patchshape,stride):
    patch=patchfy(img,patchshape,stride)
    x,y,X,Y=patch.shape
    generateTmpTestFile(x*y)
    caffe.set_mode_cpu()
    net = caffe.Net('da1.prototxt','da_iter_10000.caffemodel',caffe.TEST)
    net.blobs['data'].data[...]=img
    tmp1 = net.forward()['sigmoid1']
    print 'step1 finished'

    net = caffe.Net('sda.prototxt','sda_iter_5000.caffemodel',caffe.TEST)
    net.blobs['data'].data[...] = tmp1
    tmp2 = net.forward()['sigmoid32']
    print 'step2 finished'

    net = caffe.Net('da2.prototxt','da_iter_10000.caffemodel',caffe.TEST)
    net.blobs['sigmoid1'].data[...]=tmp2
    tmp3 =  net.forward()['sigmoid2']
    print 'step3 finished'

    tmp3.reshape((x,y,X,Y))
    imgout=unpatchfy(tmp3,img.shape,stride)
    return imgout

img = np.array(Image.open('275520.jpg'))
img = img/255.
out = testForNet(img,(10,10),3)
np.save('out.npy',out)