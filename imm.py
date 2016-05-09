import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image 
import caffe
import h5py

caffe.set_mode_cpu()
net=caffe.Net('net.prototxt','da_iter_10000.caffemodel',caffe.TEST)

data=np.load('data.npy')
label=np.load('label.npy')

net.blobs['data'].data[...]=data
data_imm=net.forward()['sigmoid1']


net.blobs['data'].data[...]=label
label_imm=net.forward()['sigmoid1']
print label_imm.shape

file1 = h5py.File('imm.h5','w')
file1.create_dataset('data',data=data_imm.reshape((data_imm.shape[0],1,1,data_imm.shape[1])))
file1.create_dataset('label',data=label_imm.reshape((label_imm.shape[0],1,1,label_imm.shape[1])))
file1.close()
