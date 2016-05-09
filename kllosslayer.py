import caffe
import numpy as np


class kllosslayer(caffe.Layer):
    """
    Compute the Euclidean Loss in the same manner as the C++ EuclideanLossLayer
    to demonstrate the class interface for developing layers in Python.
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.thou = 0.001
        self.diff[...] = bottom[0].data - bottom[1].data
        self.thous = np.average(bottom[0].data, axis=0)
        top[0].data[...] = np.sum(self.thou*np.log(self.thou/self.thous)+(1-self.thou)*np.log((1-self.thou)/(1-self.thous)))

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = np.repeat([(-self.thou/self.thous-(1-self.thou)/(1-self.thous))/bottom[0].num],bottom[0].num,axis=0)
        bottom[1].diff[...] = np.zeros(bottom[0].data.shape)
