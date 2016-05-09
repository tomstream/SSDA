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

file1 = h5py.File('imm.h5','w')
file1.create_dataset('data',data=data_imm)
file1.create_dataset('label',data=label_imm)
file1.close()
